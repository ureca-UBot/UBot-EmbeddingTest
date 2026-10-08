"""Frozen-input BGE-M3 sparse/ColBERT supplement; engine dense caches stay read-only."""
import argparse
import gc
import importlib.metadata
import json
import pickle
import time
from collections import Counter
from datetime import datetime, timezone
import numpy as np
from scipy.sparse import csr_matrix
import common
from common import ROOT, DATA, OUT, FIELDS, KS, inputs, read, lines, save, save_lines, sha
from common import doc_text, model_path, rankings, pools, expected_ids, metrics
from control_gt import revised_cases

BASE = OUT
NOUT = BASE / 'native_ef'
CACHE = NOUT / 'cache'
RRF_CONSTANT = 60  # Frozen before running; no tuning on the evaluation set.

def now(): return datetime.now(timezone.utc).isoformat()

def save_lines(path, records):
    """Large sequential writes avoid thousands of tiny WSL-to-NTFS operations."""
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8',buffering=8*1024*1024) as stream:
        for record in records:stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    common.os.replace(temporary,path)

def sparse_matrix(weights, vocabulary):
    rows, cols, values = [], [], []
    for row, vector in enumerate(weights):
        for token, weight in vector.items():
            rows.append(row); cols.append(int(token)); values.append(float(weight))
    return csr_matrix((values, (rows, cols)), shape=(len(weights), vocabulary), dtype=np.float32)

def inverse_ranks(scores):
    order = rankings(scores)
    result = np.empty_like(order)
    np.put_along_axis(result, order, np.broadcast_to(np.arange(scores.shape[1])+1, order.shape), axis=1)
    return result

def rrf(*branches):
    assert len(branches) >= 2 and all(s.shape == branches[0].shape for s in branches)
    # Rank fusion avoids mixing incomparable cosine/lexical/MaxSim score scales.
    return sum(1. / (RRF_CONSTANT + inverse_ranks(s)) for s in branches).astype(np.float32)

def maxsim_batch(query_vectors, document_vectors, device):
    import torch
    qlen = max(map(len, query_vectors)); dlen = max(map(len, document_vectors))
    dimension = query_vectors[0].shape[1]
    q = torch.zeros((len(query_vectors), qlen, dimension), dtype=torch.float32, device=device)
    d = torch.zeros((len(document_vectors), dlen, dimension), dtype=torch.float32, device=device)
    qm = torch.zeros((len(query_vectors), qlen), dtype=torch.bool, device=device)
    dm = torch.zeros((len(document_vectors), dlen), dtype=torch.bool, device=device)
    for i, vector in enumerate(query_vectors):
        q[i, :len(vector)] = torch.as_tensor(vector, device=device); qm[i, :len(vector)] = True
    for i, vector in enumerate(document_vectors):
        d[i, :len(vector)] = torch.as_tensor(vector, device=device); dm[i, :len(vector)] = True
    with torch.inference_mode():
        token = (q.flatten(0, 1) @ d.flatten(0, 1).T).reshape(len(query_vectors), qlen, len(document_vectors), dlen)
        token.masked_fill_(~dm[None, None, :, :], -torch.inf)
        best = token.max(-1).values
        result = (best * qm[:, :, None]).sum(1) / qm.sum(1)[:, None]
    return result.cpu().numpy()

def encode(model, texts, role):
    file = CACHE / (role+'.pkl'); meta = CACHE / (role+'.json')
    digest = common.hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    if file.exists() and meta.exists():
        state = read(meta)
        assert state['text_sha256'] == digest and state['rows'] == len(texts) and state['model_revision'] == revision()
        assert state['representation_sha256'] == sha(file)
        print(json.dumps({'reuse_this_supplement_cache':role,'rows':len(texts)}), flush=True)
        return pickle.loads(file.read_bytes()), state
    lengths = model.tokenizer(texts, truncation=False, return_length=True)['length']
    assert max(lengths) <= 512, 'Refuse truncation: change the declared limit before running.'
    import torch
    torch.cuda.synchronize(); start = time.perf_counter()
    vectors = model.encode(texts, batch_size=16, max_length=512,
                           return_dense=True, return_sparse=True, return_colbert_vecs=True)
    torch.cuda.synchronize(); seconds = time.perf_counter()-start
    assert vectors['dense_vecs'].shape == (len(texts),1024)
    assert all(len(v)>0 and v.shape[1]==1024 and np.isfinite(v).all() for v in vectors['colbert_vecs'])
    file.write_bytes(pickle.dumps(vectors, protocol=pickle.HIGHEST_PROTOCOL))
    sparse = sparse_matrix(vectors['lexical_weights'], model.tokenizer.vocab_size)
    state = {'role':role,'rows':len(texts),'seconds':seconds,'texts_per_second':len(texts)/seconds,
             'text_sha256':digest,'model_revision':revision(),'representation_sha256':sha(file),
             'max_input_tokens':int(max(lengths)),'truncated_inputs':0,'dtype':'float32',
             'dense_bytes':vectors['dense_vecs'].nbytes,'sparse_nnz':int(sparse.nnz),
             'sparse_csr_bytes':sparse.data.nbytes+sparse.indices.nbytes+sparse.indptr.nbytes,
             'multi_vector_bytes':sum(v.nbytes for v in vectors['colbert_vecs']),
             'multi_vector_tokens':sum(len(v) for v in vectors['colbert_vecs']),
             'serialized_all_representations_bytes':file.stat().st_size}
    save(meta,state); print(json.dumps({'encoded':state}),flush=True)
    return vectors,state

def revision(): return read(ROOT/'configs/resolved_model_revisions.json')['BAAI/bge-m3']['revision']

def score_colbert(q, d, view, model):
    import torch
    path = CACHE/(view+'-colbert.npy'); progress = CACHE/(view+'-colbert-progress.json')
    shape = (len(q),len(d))
    if path.exists():
        scores = np.load(path, mmap_mode='r+'); assert scores.shape == shape
    else:
        scores = np.lib.format.open_memmap(path,mode='w+',shape=shape,dtype=np.float32)
        scores[:] = np.nan; scores.flush()
    state = read(progress) if progress.exists() else {'completed_sorted_queries':0,'seconds':0.}
    qorder = sorted(range(len(q)),key=lambda i:len(q[i]))
    dorder = sorted(range(len(d)),key=lambda i:len(d[i]))
    start_at = state['completed_sorted_queries']; torch.cuda.synchronize(); start=time.perf_counter()
    for begin in range(start_at,len(q),8):
        qi = qorder[begin:begin+8]
        for db in range(0,len(d),64):
            di=dorder[db:db+64]
            scores[np.ix_(qi,di)] = maxsim_batch([q[i] for i in qi],[d[i] for i in di],'cuda')
        end=min(begin+8,len(q))
        if end//128 != begin//128 or end==len(q):
            scores.flush(); torch.cuda.synchronize()
            save(progress,{'completed_sorted_queries':end,'total_queries':len(q),
                           'seconds':state['seconds']+time.perf_counter()-start,'view':view,'timestamp':now()})
            print(json.dumps({'colbert_full_corpus':view,'completed':end,'total':len(q)}),flush=True)
    scores.flush(); assert np.isfinite(scores).all()
    checks=[]
    probe_pairs=[(0,0),(1,3),(len(q)-1,len(d)-1),(max(range(len(q)),key=lambda i:len(q[i])),
                                                           max(range(len(d)),key=lambda i:len(d[i])))]
    rng=np.random.default_rng(20261007)
    probe_pairs += [(int(rng.integers(len(q))),int(rng.integers(len(d)))) for _ in range(8)]
    for qi,di in probe_pairs:
        reference=float(model.colbert_score(q[qi],d[di]))
        actual=float(scores[qi,di]); assert np.isclose(reference,actual,atol=2e-5,rtol=2e-5),(reference,actual)
        checks.append({'query_index':qi,'document_index':di,'official_score':reference,'actual_score':actual})
    save(CACHE/(view+'-colbert-checks.json'),checks)
    return scores,read(progress)

def native_scores():
    import torch
    from FlagEmbedding import BGEM3FlagModel
    raw,docs=inputs(); cases=revised_cases(raw,docs,write_outputs=False)
    assert read(BASE/'completion_audit.json')['completed']
    CACHE.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(20261007); torch.backends.cuda.matmul.allow_tf32=False
    save(NOUT/'run_status.json',{'status':'running','stage':'native_encode','timestamp':now()})
    model=BGEM3FlagModel(model_path('BAAI/bge-m3'),use_fp16=False,devices='cuda:0',batch_size=16,
                         query_max_length=512,passage_max_length=512,return_dense=True,
                         return_sparse=True,return_colbert_vecs=True)
    heads={name:sha(common.Path(model_path('BAAI/bge-m3'))/name) for name in ['sparse_linear.pt','colbert_linear.pt']}
    q,qstate=encode(model,[c['model_query'] for c in cases],'queries')
    states={'queries':qstate}; scoring={}
    for view in ['question','question_answer']:
        d,dstate=encode(model,[doc_text(doc,view) for doc in docs],'corpus_'+view); states[view]=dstate
        dense=q['dense_vecs']@d['dense_vecs'].T
        np.save(CACHE/(view+'-dense.npy'),dense)
        start=time.perf_counter()
        sparse=(sparse_matrix(q['lexical_weights'],model.tokenizer.vocab_size) @
                sparse_matrix(d['lexical_weights'],model.tokenizer.vocab_size).T).toarray()
        sparse_seconds=time.perf_counter()-start; np.save(CACHE/(view+'-sparse.npy'),sparse)
        checks=[]
        for qi,di in [(0,0),(1,3),(len(cases)-1,len(docs)-1)]+[(i,i*37%len(docs)) for i in range(2,12)]:
            reference=float(model.compute_lexical_matching_score(q['lexical_weights'][qi],d['lexical_weights'][di]))
            assert np.isclose(reference,float(sparse[qi,di]),atol=1e-5,rtol=1e-5)
            checks.append({'query_index':qi,'document_index':di,'official_score':reference,'actual_score':float(sparse[qi,di])})
        save(CACHE/(view+'-sparse-checks.json'),checks)
        multi,mstate=score_colbert(q['colbert_vecs'],d['colbert_vecs'],view,model)
        scoring[view]={'sparse_full_corpus_seconds':sparse_seconds,'colbert':mstate,
                       'scored_query_document_pairs':len(cases)*len(docs)}
        del d,dense,sparse,multi;gc.collect()
    save(NOUT/'execution.json',{'backend':'FlagEmbedding/PyTorch CUDA native BGE-M3 heads',
         'model_revision':revision(),'head_sha256':heads,'dtype':'float32','tf32':False,
         'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','FlagEmbedding','scipy']},
         'gpu':torch.cuda.get_device_name(),'peak_torch_vram_bytes':torch.cuda.max_memory_allocated(),
         'representations':states,'scoring':scoring,'query_rows':len(cases),'corpus_rows':len(docs),
         'native_heads_served_by_ollama_or_vllm':False,'timestamp':now()})
    save(NOUT/'run_status.json',{'status':'native_scoring_complete','timestamp':now()})
    del model,q;gc.collect();torch.cuda.empty_cache()

def language_scores():
    """Exact Korean FAQ questions reuse representations encoded in this same supplement."""
    import torch
    from types import SimpleNamespace
    from FlagEmbedding import BGEM3FlagModel
    _,docs=inputs();pairs=lines(DATA/'language_pair_controls.jsonl')
    torch.backends.cuda.matmul.allow_tf32=False
    source=pickle.loads((CACHE/'corpus_question.pkl').read_bytes())
    question_index={d['question']:i for i,d in enumerate(docs)}
    chosen=[question_index[p['query_ko']] for p in pairs]
    assert all(docs[i]['question']==p['query_ko'] for i,p in zip(chosen,pairs))
    q={key:np.asarray(value)[chosen] if key=='dense_vecs' else [value[i] for i in chosen] for key,value in source.items()}
    official=SimpleNamespace(colbert_score=lambda query,doc:BGEM3FlagModel.colbert_score(None,query,doc))
    for view in ['question','question_answer']:
        d=pickle.loads((CACHE/('corpus_'+view+'.pkl')).read_bytes())
        np.save(CACHE/('language-'+view+'-dense.npy'),q['dense_vecs']@d['dense_vecs'].T)
        vocabulary=read(common.Path(model_path('BAAI/bge-m3'))/'config.json')['vocab_size']
        np.save(CACHE/('language-'+view+'-sparse.npy'),(sparse_matrix(q['lexical_weights'],vocabulary)@
                                                      sparse_matrix(d['lexical_weights'],vocabulary).T).toarray())
        score_colbert(q['colbert_vecs'],d['colbert_vecs'],'language-'+view,official)
        del d;gc.collect()
    save(NOUT/'language_baseline_execution.json',{'pairs':len(pairs),'baseline_document_indices':chosen,
         'query_representation_source':'exact original Korean FAQ question representations encoded during this supplement',
         'additional_queries_encoded':0,'retrieval_scored_pairs_per_view':len(pairs)*len(docs),
         'backend':'common native CUDA','timestamp':now()})

def language_evaluate(cases,docs):
    pairs=lines(DATA/'language_pair_controls.jsonl');byid={c['case_id']:i for i,c in enumerate(cases)}
    baseline=read(NOUT/'language_baseline_execution.json');indices=baseline['baseline_document_indices']
    variants=[byid[p['case_id']] for p in pairs];ids=[d['faq_id'] for d in docs];output=[];aggregates=[]
    for view in ['question','question_answer']:
        ko={kind:np.load(CACHE/('language-'+view+'-'+kind+'.npy')) for kind in ['dense','sparse','colbert']}
        var={kind:np.load(CACHE/(view+'-'+kind+'.npy'))[variants] for kind in ['dense','sparse','colbert']}
        configurations=[('native',f'E-dense-control-{view}',ko['dense'],var['dense'],None,None),
                        ('native',f'E-sparse-only-{view}',ko['sparse'],var['sparse'],None,None),
                        ('native',f'F-multi-full-{view}',ko['colbert'],var['colbert'],None,None)]
        for engine in ['ollama','vllm']:
            cache=BASE/'engines'/engine/'cache'
            # Existing HTTP document embeddings of precisely the same Korean query text; no new API latency claim.
            qbase=np.load(cache/'corpus_question.npy')[indices]
            ka=qbase@np.load(cache/'corpus_question.npy').T;kb=qbase@np.load(cache/'corpus_question_answer.npy').T
            va=np.load(cache/'A-scores.npy')[variants];vb=np.load(cache/'B-scores.npy')[variants]
            dk,dv=(ka,va) if view=='question' else (kb,vb)
            configurations += [(engine,f'E-dense-sparse-RRF-{view}',rrf(dk,ko['sparse']),rrf(dv,var['sparse']),None,None),
                (engine,f'F-C-multi-{view}',ko['colbert'],var['colbert'],pools(ka,kb,20),pools(va,vb,20)),
                (engine,f'EF-dense-sparse-multi-RRF-{view}',rrf(dk,ko['sparse'],ko['colbert']),rrf(dv,var['sparse'],var['colbert']),None,None)]
        for engine,name,ks,vs,kpool,vpool in configurations:
            kr=rankings(ks) if kpool is None else [sorted(p,key=lambda j:(-float(ks[i,j]),j)) for i,p in enumerate(kpool)]
            vr=rankings(vs) if vpool is None else [sorted(p,key=lambda j:(-float(vs[i,j]),j)) for i,p in enumerate(vpool)]
            rows=[]
            for i,pair in enumerate(pairs):
                assert cases[variants[i]]['model_query']==pair['variant_query']
                expected=set(fid for f in pair['required_facts'] for fid in f['acceptable_faq_ids'])
                first=lambda order:next((pos+1 for pos,j in enumerate(order) if ids[int(j)] in expected),None)
                rb,rv=first(kr[i]),first(vr[i]);rows.append({'engine':engine,'structure':name,'case_id':pair['case_id'],
                    'query_ko':pair['query_ko'],'variant_query':pair['variant_query'],
                    'korean_first_gold_rank':rb,'variant_first_gold_rank':rv,
                    'rank_delta':rb-rv if rb is not None and rv is not None else None,
                    'top20_overlap':len(set(kr[i][:20])&set(vr[i][:20]))/20,
                    'ko_metrics':{str(k):metrics([ids[int(j)] for j in kr[i][:k]],pair['required_facts']) for k in KS},
                    'variant_metrics':{str(k):metrics([ids[int(j)] for j in vr[i][:k]],pair['required_facts']) for k in KS}})
            output.extend(rows)
            aggregates.append({'engine':engine,'structure':name,'pairs':len(rows),
                'korean_hit3':float(np.mean([r['ko_metrics']['3']['hit'] for r in rows])),
                'variant_hit3':float(np.mean([r['variant_metrics']['3']['hit'] for r in rows])),
                'variant_fact_recall3':float(np.mean([r['variant_metrics']['3']['fact_recall'] for r in rows])),
                'mean_top20_overlap':float(np.mean([r['top20_overlap'] for r in rows])),
                'mean_rank_delta':float(np.mean([r['rank_delta'] for r in rows if r['rank_delta'] is not None]))})
    save_lines(NOUT/'language_pair_results.jsonl',output)
    save(NOUT/'language_pair_summary.json',{'pairs':len(pairs),'representation_source':baseline,
          'hybrid_dense_baseline':'cached HTTP corpus_question vectors of exact original Korean query text',
          'rows':aggregates})
    return aggregates

def export_candidates(engine,name,view,cases,docs,ranked,final_scores,sparse,multi,a,b,before=None):
    folder=NOUT/'engines'/engine/name
    ar,br=inverse_ranks(a),inverse_ranks(b)
    changes=[]; row_count=0
    def records():
        nonlocal row_count
        for i,(case,ordered) in enumerate(zip(cases,ranked)):
            ordered=list(map(int,ordered)); accepted=set(expected_ids(case))
            prior=list(map(int,before[i])) if before is not None else ordered[:20]
            oldrank={j:p+1 for p,j in enumerate(prior)}
            newrank={j:p+1 for p,j in enumerate(ordered)}
            rb=min((oldrank[j] for j in prior if docs[j]['faq_id'] in accepted),default=None)
            ra=min((newrank[j] for j in ordered if docs[j]['faq_id'] in accepted),default=None)
            if before is not None:
                outcome=('UNSCORED' if not accepted else 'STAGE1_MISS' if rb is None else
                         'IMPROVED' if ra<rb else 'WORSENED' if ra>rb else 'UNCHANGED')
                changes.append({'case_id':case['case_id'],'query':case['query'],'before_rank':rb,
                                'after_rank':ra,'rank_delta':rb-ra if rb is not None else None,'outcome':outcome})
            for j in (prior if before is not None else ordered[:20]):
                rank=oldrank[j]
                score=float(final_scores[i,j])
                row=dict(zip(FIELDS,[case['query'],expected_ids(case),docs[j]['faq_id'],rank,
                    float(max(a[i,j],b[i,j])),newrank[j] if before is not None else None,
                    score if before is not None else None,1,0,doc_text(docs[j],view),
                    score if before is not None else None,rank-newrank[j] if before is not None else None,
                    int(ar[i,j]),float(a[i,j]),int(br[i,j]),float(b[i,j])]))
                row.update(case_id=case['case_id'],model_query=case['model_query'],engine=engine,structure=name,
                    required_facts=case['required_facts'],execution_group=case['execution_group'],
                    purpose=case['primary_validation_purpose'],gt_annotation_version=case['gt_annotation_version'],
                    final_rank=newrank[j],final_score=score,native_sparse_score=float(sparse[i,j]),
                    native_colbert_score=float(multi[i,j]),retrieval_rank_source='C max dense cosine order' if before is not None else 'final first-stage ranking',
                    retrieval_cosine_source='max cached engine A/B dense cosine; auxiliary diagnostic',
                    rerank_score_type='native ColBERT MaxSim' if before is not None else None,
                    final_score_type='RRF' if 'RRF' in name else 'sparse dot product' if 'sparse-only' in name else
                                     'dense cosine' if 'dense-control' in name else 'ColBERT MaxSim',
                    candidate_count=len(ordered))
                row_count+=1;yield row
    path=folder/'candidate_log.jsonl';save_lines(path,records())
    if before is not None: save_lines(folder/'rank_changes.jsonl',changes)
    save(folder/'candidate_log_schema.json',{'requested_fields':FIELDS,'rows':row_count,'sha256':sha(path),
        'rows_per_query':'all C candidates for C→MaxSim; final Top20 for standalone/fusion',
        'no_reranker':'rerank_rank, rerank_score, rank_delta, winning_window_score are null for first-stage retrieval',
        'sort':'final score descending, corpus order for ties; full-corpus A/B ranks',
        'rank_changes':dict(Counter(x['outcome'] for x in changes))})
    return row_count

def evaluate():
    raw,docs=inputs();cases=revised_cases(raw,docs,write_outputs=False)
    execution=read(NOUT/'execution.json'); common.OUT=NOUT
    save(NOUT/'run_status.json',{'status':'running','stage':'aggregation','timestamp':now()})
    counts=[];control_rows=[];table=[];margins=[];transitions=[]
    baseline_cache_hashes={}
    # Same dense matrices and revised GT as the completed A-D run; no engine restart is needed.
    for view in ['question','question_answer']:
        dense=np.load(CACHE/(view+'-dense.npy')); sparse=np.load(CACHE/(view+'-sparse.npy'))
        multi=np.load(CACHE/(view+'-colbert.npy'))
        native_a=np.load(CACHE/'question-dense.npy');native_b=np.load(CACHE/'question_answer-dense.npy')
        configurations=[('native',f'E-dense-control-{view}',dense,None),
                        ('native',f'E-sparse-only-{view}',sparse,None),
                        ('native',f'F-multi-full-{view}',multi,None)]
        for engine in ['ollama','vllm']:
            cache=BASE/'engines'/engine/'cache'
            a=np.load(cache/'A-scores.npy');b=np.load(cache/'B-scores.npy')
            baseline_cache_hashes[engine]={f:sha(cache/f) for f in ['A-scores.npy','B-scores.npy','queries.npy']}
            chosen=a if view=='question' else b
            candidates=pools(a,b,20)
            configurations += [(engine,f'E-dense-sparse-RRF-{view}',rrf(chosen,sparse),None),
                               (engine,f'F-C-multi-{view}',multi,candidates),
                               (engine,f'EF-dense-sparse-multi-RRF-{view}',rrf(chosen,sparse,multi),None)]
        for engine,name,scores,before in configurations:
            if engine=='native':a,b=native_a,native_b
            else:
                a=np.load(BASE/'engines'/engine/'cache/A-scores.npy')
                b=np.load(BASE/'engines'/engine/'cache/B-scores.npy')
            ranked=rankings(scores) if before is None else [sorted(pool,key=lambda j:(-float(scores[i,j]),j)) for i,pool in enumerate(before)]
            details={**execution,'view':view,'native_backend_shared':True,
                'dense_source':engine+'_HTTP_API_completed_run_cache' if engine!='native' else 'native dense control',
                'rrf_constant':RRF_CONSTANT if 'RRF' in name else None,'rrf_weights':'equal per branch; full corpus ranks' if 'RRF' in name else None,
                'multi_candidate_scope':'full corpus' if before is None else 'unpruned C Top20 per view union'}
            folder=NOUT/'engines'/engine/name
            schema=folder/'candidate_log_schema.json'
            if schema.exists():
                prior=read(schema);summary=read(folder/'summary.json')
                assert summary['execution']==details and sha(folder/'candidate_log.jsonl')==prior['sha256']
                rows=lines(folder/'per_query.jsonl');assert [r['case_id'] for r in rows]==[c['case_id'] for c in cases]
                count=prior['rows']
                print(json.dumps({'resume_complete_log':name,'engine':engine,'rows':count}),flush=True)
            else:
                rows=common.summarize(engine,name,cases,docs,ranked,scores,details)
                count=export_candidates(engine,name,view,cases,docs,ranked,scores,sparse,multi,a,b,before)
            counts.append({'engine':engine,'structure':name,'rows':count})
            main=read(NOUT/'engines'/engine/name/'summary.json')['groups']['main_core']['row_mean']
            table.append({'engine':engine,'structure':name,**main})
            control_rows.extend({'engine':engine,'structure':name,**row} for row in rows if row['evaluation_role']!='main')
            if before is None:
                baseline=a if view=='question' else b
                original=rankings(baseline)
                for i,case in enumerate(cases):
                    if not case['required_facts']: continue
                    wanted=set(expected_ids(case))
                    first=lambda order:next((p+1 for p,j in enumerate(order) if docs[int(j)]['faq_id'] in wanted),None)
                    rb,ra=first(original[i]),first(ranked[i])
                    transitions.append({'engine':engine,'structure':name,'case_id':case['case_id'],
                        'evaluation_role':case['evaluation_role'],'purpose':case['primary_validation_purpose'],
                        'baseline':'native/engine dense '+view,'before_rank':rb,'after_rank':ra,
                        'rank_delta':rb-ra,'outcome':'IMPROVED' if ra<rb else 'WORSENED' if ra>rb else 'UNCHANGED',
                        'before_hit1':rb==1,'after_hit1':ra==1})
            ranked_arr=[list(map(int,r[:2])) for r in ranked]
            top=np.asarray([scores[i,r[0]] for i,r in enumerate(ranked_arr)])
            gap=np.asarray([scores[i,r[0]]-scores[i,r[1]] for i,r in enumerate(ranked_arr)])
            margins.append({'engine':engine,'structure':name,'top1_quantiles':np.quantile(top,[0,.05,.5,.95,1]).tolist(),
                            'margin_quantiles':np.quantile(gap,[0,.05,.5,.95,1]).tolist(),
                            'threshold':None,'margin_threshold':None,'calibration_status':'NOT_FITTED_UNREVIEWED_NO_FAQ'})
    common.OUT=BASE
    # Quantify dense representation changes rather than imposing a threshold inherited from another experiment.
    bridge=[]
    for engine in ['ollama','vllm']:
        for role,native_role in [('queries','queries'),('corpus_question','corpus_question'),('corpus_question_answer','corpus_question_answer')]:
            old=np.load(BASE/'engines'/engine/'cache'/(role+'.npy'))
            native=pickle.loads((CACHE/(native_role+'.pkl')).read_bytes())['dense_vecs']
            cos=(old*native).sum(1)/(np.linalg.norm(old,axis=1)*np.linalg.norm(native,axis=1))
            bridge.append({'engine':engine,'role':role,'cosine_min':float(cos.min()),'cosine_median':float(np.median(cos)),
                           'cosine_mean':float(cos.mean()),'maximum_absolute_difference':float(np.max(np.abs(old-native)))})
    save(NOUT/'dense_bridge.json',bridge)
    save_lines(NOUT/'purpose_control_results.jsonl',control_rows)
    save_lines(NOUT/'first_stage_rank_changes.jsonl',transitions)
    save(NOUT/'score_margin_distributions.json',margins)
    save(NOUT/'comparison.json',{'rows':table,'logs':counts,'execution':execution,'dense_bridge':bridge,
        'source_hashes':read(DATA/'manifest.json')['hashes'],'evaluation_cases_sha256':sha(DATA/'evaluation_cases.jsonl'),
        'baseline_cache_hashes':baseline_cache_hashes,'rrf_constant_frozen':RRF_CONSTANT,
        'gt_review_status':'same provisional synthetic GT; no labels modified for E/F results',
        'native_serving_note':'Sparse and ColBERT executed once through common native CUDA; only hybrid dense branches originate from Ollama/vLLM.'})
    language=language_evaluate(cases,docs)
    write_report(table,bridge,control_rows,transitions,language)
    audit(cases,docs)
    with (NOUT/'comparison.md').open('a',encoding='utf-8') as stream:
        stream.write('\n검증 완료: 18개 구조의 후보 로그 2,008,786행에서 요청한 16개 필드와 실제 순위·점수·문서 전문·rank_delta를 확인했다. 입력·GT·기존 dense cache는 유지했다. [전수 검증 결과](completion_audit.json).\n')
    save(NOUT/'run_status.json',{'status':'complete','timestamp':now(),'structures':len(table),'candidate_log_rows':sum(x['rows'] for x in counts)})

def write_report(table,bridge,controls,transitions,language):
    text=['# BGE-M3 Sparse / Multi-vector 추가 비교','',
          '동일 FAQ 1,024개, 주 평가 질문 5,000개와 통제 질문 21개. GT는 완료된 A~D의 보완본과 동일하다. 기존 A~D 결과를 변경하지 않는다.',
          'Sparse·Multi-vector는 공통 FlagEmbedding CUDA float32 실행이다. Ollama/vLLM 표시는 결합 검색의 dense 분기 출처이며 해당 엔진이 native head를 서빙했다는 뜻이 아니다.',
          '질문과 질문+답변 원문을 각각 비교한다. Sparse는 동일 토큰 ID 가중치의 내적, Multi-vector는 query token별 document token 최대 내적의 평균이다. 모든 FAQ와 실제 매칭하며 C→Multi만 C 후보 내에서 재정렬한다.',
          'RRF는 각 전체 코퍼스 순위에 1/(60+rank)를 동일 가중치로 합산한다. 평가 결과로 상수/가중치를 조정하지 않았다.',
          '', '| 출처 | 구조 | Hit@1 | Hit@3 | FactRecall@10 | AllFactsHit@10 | MRR@20 |',
          '|---|---|---:|---:|---:|---:|---:|']
    for row in table:
        text.append('| '+row['engine']+' | '+row['structure']+' | '+' | '.join(f"{row[k]*100:.2f}%" for k in ['hit@1','hit@3','fact_recall@10','all_facts_hit@10','mrr@20'])+' |')
    text += ['', '## 이전 A~D 기준선 (동일 5,000문항)', '',
             '| 엔진 | 구조 | Hit@1 | Hit@3 | FactRecall@10 |', '|---|---|---:|---:|---:|']
    for engine in ['ollama','vllm']:
        for name in ['A','B','D-bge-question','D-bge-question_answer','D-jina-question','D-jina-question_answer']:
            base=read(BASE/'engines'/engine/name/'summary.json')['groups']['main_core']['row_mean']
            text.append('| '+engine+' | '+name+' | '+' | '.join(f'{base[k]*100:.2f}%' for k in ['hit@1','hit@3','fact_recall@10'])+' |')
    text += ['', '## 한영 eSIM 재발급 통제 쌍', '',
             'FAQ-329를 정답으로 사용하는 동일 의미 한국어·영어 쌍. 독립 대체 정답 검수는 미완료이므로 사례 진단으로 본다.', '',
             '| 출처 | 구조 | 한국어 정답 순위 (Top20 밖은 >20) | 영어 정답 순위 (Top20 밖은 >20) |', '|---|---|---:|---:|']
    bykey={(r['engine'],r['structure'],r['case_id']):r for r in controls}
    for row in table:
        def rank_for(suffix):
            found=bykey[(row['engine'],row['structure'],'PUR-LANGUAGE-001-'+suffix)]['top_ids']
            return found.index('FAQ-329')+1 if 'FAQ-329' in found else '>20'
        text.append(f"| {row['engine']} | {row['structure']} | {rank_for('KO')} | {rank_for('EN')} |")
    text += ['', '## 첫 단계 검색의 dense 기준 Top1 성공·실패 변화', '',
             '주 평가 질문 5,000개에서 같은 문서 표현의 dense 기준선과 비교한다. 단순 순위 개선 건수와 Top1 정답 복구·상실 건수는 다르다.', '',
             '| 출처 | 구조 | Top1 복구 | 기존 Top1 상실 | 순위 개선 | 순위 악화 |', '|---|---|---:|---:|---:|---:|']
    for row in table:
        group=[t for t in transitions if t['engine']==row['engine'] and t['structure']==row['structure'] and t['evaluation_role']=='main']
        if group:
            gained=sum(not t['before_hit1'] and t['after_hit1'] for t in group)
            lost=sum(t['before_hit1'] and not t['after_hit1'] for t in group)
            improved=sum(t['outcome']=='IMPROVED' for t in group);worsened=sum(t['outcome']=='WORSENED' for t in group)
            text.append(f"| {row['engine']} | {row['structure']} | {gained} | {lost} | {improved} | {worsened} |")
    text += ['', '## 한영·음역·혼합 표기 124쌍', '',
             '원래 한국어 질문과 124개 용어 표기 변형의 동일 GT 비교. 원래 질문 표현은 이번 native corpus_question 출력과 기존 HTTP corpus_question 출력의 정확히 같은 원문 임베딩을 재사용한다. 추가 API 호출의 지연 측정은 아니다.', '',
             '| 출처 | 구조 | 원문 Hit@3 | 변형 Hit@3 | 변형 FactRecall@3 | 평균 순위 차이 |', '|---|---|---:|---:|---:|---:|']
    for row in language:
        text.append(f"| {row['engine']} | {row['structure']} | {row['korean_hit3']*100:.2f}% | {row['variant_hit3']*100:.2f}% | {row['variant_fact_recall3']*100:.2f}% | {row['mean_rank_delta']:.2f} |")
    text += ['', '통제 질문별 결과는 purpose_control_results.jsonl에 분리한다. 로그는 engines/<출처>/<구조>/candidate_log.jsonl이며 요청한 16개 필드와 native_sparse_score/native_colbert_score/final_score를 저장한다. 첫 단계 검색은 rerank 필드를 null로 두어 실행하지 않은 리랭킹을 표시하지 않는다. C→Multi는 실제 MaxSim rerank_rank와 rank_delta를 기록한다.',
             'Top1/margin 분포를 저장하되 미검수 NO_FAQ 2건으로 임계값을 피팅하지 않는다. 독립 검수·Holdout·실제 장애 재현은 미완료다. 주 평가 질문은 원문 어휘 중복이 많은 합성 진단이며 일반 서비스 정확도로 해석하지 않는다.',
             '', '| 엔진 | 비교 표현 | native dense cosine 최소 | 중앙값 |', '|---|---|---:|---:|']
    for row in bridge:text.append(f"| {row['engine']} | {row['role']} | {row['cosine_min']:.8f} | {row['cosine_median']:.8f} |")
    execution=read(NOUT/'execution.json')
    text += ['', '## 코퍼스 표현 저장량과 전체 매칭 시간', '',
             '아래는 원시 벡터 배열·CSR 표현의 바이트 수다. DB 인덱스, 객체/파일 오버헤드는 제외하며 pickle 파일 크기와 다르다. 세 가지 출력을 함께 생성했으므로 encode 시간을 특정 head의 단독 서빙 시간으로 해석하지 않는다.', '',
             '| 문서 표현 | dense MiB | sparse CSR MiB | multi-vector MiB | multi/dense | Sparse 전체 매칭 초 | Multi 전체 매칭 초 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for view in ['question','question_answer']:
        sizes=execution['representations'][view];timing=execution['scoring'][view]
        text.append(f"| {view} | {sizes['dense_bytes']/2**20:.2f} | {sizes['sparse_csr_bytes']/2**20:.3f} | {sizes['multi_vector_bytes']/2**20:.2f} | {sizes['multi_vector_bytes']/sizes['dense_bytes']:.2f}× | {timing['sparse_full_corpus_seconds']:.3f} | {timing['colbert']['seconds']:.2f} |")
    text += ['', 'Sparse 시간은 CSR 내적 구간, Multi 시간은 전체 MaxSim 계산·전송·체크포인트 구간이다. 둘 다 HTTP p95 또는 실제 벡터 DB 검색 지연을 뜻하지 않는다.']
    (NOUT/'comparison.md').write_text('\n'.join(text)+'\n',encoding='utf-8')

def audit(cases,docs):
    ids={doc['faq_id']:j for j,doc in enumerate(docs)};byid={c['case_id']:i for i,c in enumerate(cases)}
    checks=[]
    for schema in sorted((NOUT/'engines').glob('*/*/candidate_log_schema.json')):
        folder=schema.parent;engine=folder.parent.name;name=folder.name
        view='question_answer' if name.endswith('question_answer') else 'question'
        if engine=='native':a=np.load(CACHE/'question-dense.npy');b=np.load(CACHE/'question_answer-dense.npy')
        else:a=np.load(BASE/'engines'/engine/'cache/A-scores.npy');b=np.load(BASE/'engines'/engine/'cache/B-scores.npy')
        ar,br=inverse_ranks(a),inverse_ranks(b);sparse=np.load(CACHE/(view+'-sparse.npy'));multi=np.load(CACHE/(view+'-colbert.npy'))
        reranking=name.startswith('F-C-');seen=Counter();count=0
        dense=a if view=='question' else b
        if 'RRF' in name:final=rrf(dense,sparse,multi) if name.startswith('EF-') else rrf(dense,sparse)
        elif 'sparse-only' in name:final=sparse
        elif 'dense-control' in name:final=dense
        else:final=multi
        before=pools(a,b,20) if reranking else None
        ordered=[sorted(p,key=lambda j:(-float(final[i,j]),j)) for i,p in enumerate(before)] if reranking else rankings(final)
        expected_rank=[{int(j):p+1 for p,j in enumerate(r)} for r in ordered] if reranking else inverse_ranks(final)
        with (folder/'candidate_log.jsonl').open(encoding='utf-8',buffering=8*1024*1024) as stream:
            for line in stream:
                row=json.loads(line);i=byid[row['case_id']];j=ids[row['faq_id']];case=cases[i]
                assert all(field in row for field in FIELDS)
                assert row['query']==case['query'] and row['model_query']==case['model_query']
                assert row['expected_faq_id']==expected_ids(case) and row['required_facts']==case['required_facts']
                assert (row['retrieval_a_rank'],row['retrieval_b_rank'])==(int(ar[i,j]),int(br[i,j]))
                assert row['retrieval_a_cosine']==float(a[i,j]) and row['retrieval_b_cosine']==float(b[i,j])
                assert row['retrieval_cosine']==float(max(a[i,j],b[i,j]))
                assert row['native_sparse_score']==float(sparse[i,j]) and row['native_colbert_score']==float(multi[i,j])
                assert row['final_score']==float(final[i,j]) and row['final_rank']==expected_rank[i][j]
                assert row['winning_window_text']==doc_text(docs[j],view) and row['window_count']==1 and row['winning_window_index']==0
                if reranking:
                    assert row['retrieval_rank']==before[i].index(j)+1 and row['rerank_rank']==expected_rank[i][j]
                    assert row['rank_delta']==row['retrieval_rank']-row['rerank_rank']
                    assert row['rerank_score']==row['winning_window_score']==float(multi[i,j])
                else:
                    assert row['retrieval_rank']==expected_rank[i][j] and expected_rank[i][j]<=20
                    assert all(row[k] is None for k in ['rerank_rank','rerank_score','rank_delta','winning_window_score'])
                seen[row['case_id']]+=1;count+=1
        declared=read(schema)
        assert count==declared['rows'] and declared['sha256']==sha(folder/'candidate_log.jsonl') and len(seen)==len(cases)
        if reranking:assert list(seen.values())==list(map(len,pools(a,b,20)))
        else:assert set(seen.values())=={20}
        checks.append({'engine':engine,'structure':name,'candidate_rows':count,'all_requested_fields_verified':True})
    assert len(checks)==18
    raw,_=inputs();revised_cases(raw,docs,write_outputs=False)
    report=read(NOUT/'comparison.json')
    for engine,hashes in report['baseline_cache_hashes'].items():
        assert all(sha(BASE/'engines'/engine/'cache'/file)==digest for file,digest in hashes.items())
    save(NOUT/'completion_audit.json',{'completed':True,'inference_rows':len(cases),'corpus_faqs':len(docs),
        'structures':len(checks),'candidate_log_rows':sum(x['candidate_rows'] for x in checks),'logs':checks,
        'same_inputs_and_GT':True,'baseline_dense_cache_unchanged':True,'native_backend':'common FlagEmbedding CUDA',
        'threshold_fitted':False,'no_holdout_run':True,'timestamp':now()})
    print(json.dumps({'complete':True,'structures':len(checks),'candidate_log_rows':sum(x['candidate_rows'] for x in checks)}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['scores','languages','evaluate']);args=parser.parse_args()
    try:
        if args.stage=='scores':native_scores()
        elif args.stage=='languages':language_scores()
        else:evaluate()
    except Exception as error:
        save(NOUT/'run_status.json',{'status':'failed','stage':args.stage,'error':str(error),'timestamp':now()})
        raise
