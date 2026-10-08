"""Execute matched new-FAQ engine matrices; full input, no inference dedupe."""
import argparse
import gc
import hashlib
import json
import time
from collections import Counter,defaultdict
import numpy as np
from common import ROOT,DATA,OUT,N,MAIN_N,REFERENCE_N,MAX_LENGTH,RERANK_MAX_LENGTH,NAMES,inputs,read,save,save_lines,sha,model_path,doc_text,rankings,pools,summarize,calibrate_policy,lines

def folder(engine):return OUT/'engines'/engine
def torch_setup():
    import torch
    assert torch.cuda.is_available()
    torch.manual_seed(20261006)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return torch
def lengths(tokenizer,texts):
    result=[]
    for start in range(0,len(texts),256):
        result.extend(tokenizer(texts[start:start+256],truncation=False,return_length=True)['length'])
    return np.asarray(result,dtype=np.int32)
def mmap(path,shape,fill=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        values=np.load(path,mmap_mode='r+')
        assert values.shape==shape and values.dtype==np.float32
    else:
        values=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=shape)
        if fill is not None:values[:]=fill;values.flush()
    return values

def validate():
    from transformers import AutoTokenizer
    cases,docs=inputs()
    manifest=read(DATA/'manifest.json')
    main=[c for c in cases if c['evaluation_role']=='main']
    parents=defaultdict(list)
    for c in main:parents[c['parent_case_id']].append(c)
    assert len(parents)==500 and all(len(v)==10 and len({c['query'] for c in v})==10 for v in parents.values())
    assert Counter(c['type'] for c in main)==manifest['type_counts']
    assert all(c['query']==c['model_query'] and not c['no_faq_truth'] for c in cases)
    assert all(all(sid in {d['faq_id'] for d in docs} for sid in c['source_ids']) for c in cases)
    tokenizer=AutoTokenizer.from_pretrained(model_path('BAAI/bge-m3'),trust_remote_code=False)
    ql=lengths(tokenizer,[c['model_query'] for c in cases])
    dl=lengths(tokenizer,[doc_text(d,'question_answer') for d in docs])
    assert max(ql.max(),dl.max())<=MAX_LENGTH
    chunks=lines(DATA/'reranker_chunks.jsonl')
    assert {c['faq_id'] for c in chunks}=={d['faq_id'] for d in docs}
    np.save(OUT/'query_token_lengths.npy',ql)
    report={'main_rows':MAIN_N,'reference_rows':REFERENCE_N,'corpus_faqs':len(docs),
        'queries_max_tokens':int(ql.max()),'full_qa_docs_max_tokens':int(dl.max()),
        'dense_input_limit':MAX_LENGTH,'reranker_input_limit':RERANK_MAX_LENGTH,
        'variant_input_sha256':manifest['source_variants_sha256'],'frozen_case_sha256':sha(DATA/'cases.jsonl'),
        'truncation':False,'gt_or_answers_in_query':False,'previous_data_imported':False,
        'holdout_used':False,'independent_annotation_complete':False,'no_faq_rows':0,
        'reranker_qa_policy':manifest['protocol']['reranker_qa_input'],'repeat_rows':0}
    save(OUT/'input_validation.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)

def embeddings(a):
    import requests
    from transformers import AutoTokenizer
    cases,docs=inputs()
    base=folder(a.engine)
    cache=base/'cache';cache.mkdir(parents=True,exist_ok=True)
    assert not (base/'C-M20/summary.json').exists()
    session=requests.Session()
    logpath=base/'embedding_requests.jsonl'
    def encode(texts,role,token_lengths):
        values=mmap(cache/f'{role}.npy',(len(texts),1024))
        statepath=cache/f'{role}-progress.json'
        state=read(statepath) if statepath.exists() else {'completed':0,'seconds':0.}
        begin=state['completed']
        while begin<len(texts):
            end=min(begin+16,len(texts))
            while end>begin+1 and (end-begin)*int(max(token_lengths[begin:end]))>4096:end-=1
            chunk=texts[begin:end]
            assert max(token_lengths[begin:end])<=MAX_LENGTH
            if a.engine=='ollama':
                endpoint='/api/embed'
                # BERT embeddings require one whole input to fit the physical batch.
                # Preserve the original 2048 setting for short inputs and cached queries.
                physical_batch=4096 if max(token_lengths[begin:end])>2048 else 2048
                payload={'model':'bge-m3','input':chunk,'truncate':False,'keep_alive':'30m','options':{'num_ctx':MAX_LENGTH,'num_batch':physical_batch}}
            else:
                endpoint='/v1/embeddings'
                payload={'model':'BAAI/bge-m3','input':chunk,'encoding_format':'float'}
            start=time.perf_counter()
            response=None
            try:
                response=session.post(a.url.rstrip('/')+endpoint,json=payload,timeout=300)
                response.raise_for_status();body=response.json()
                vv=body['embeddings'] if a.engine=='ollama' else [v['embedding'] for v in sorted(body['data'],key=lambda v:v['index'])]
                vv=np.asarray(vv,dtype=np.float32)
                assert vv.shape==(len(chunk),1024) and np.isfinite(vv).all()
                norm=np.linalg.norm(vv,axis=1)
                assert np.all(norm>0)
            except Exception as exc:
                with logpath.open('a',encoding='utf-8') as stream:stream.write(json.dumps({'role':role,'first_row':begin,'texts':len(chunk),'success':False,'error':str(exc),'request_options':payload.get('options'),'http_status':response.status_code if response is not None else None,'response_error':response.text[:4000] if response is not None else None})+'\n')
                raise
            elapsed=time.perf_counter()-start
            values[begin:end]=vv/norm[:,None];values.flush()
            record={'role':role,'first_row':begin,'texts':len(chunk),'endpoint':endpoint,'success':True,
                'input_sha256':hashlib.sha256(json.dumps(chunk,ensure_ascii=False).encode()).hexdigest(),
                'latency_seconds':elapsed,'status':response.status_code,'max_input_tokens':int(max(token_lengths[begin:end])),
                'truncation_requested':False,'request_options':payload.get('options'),
                'processed_tokens':body.get('prompt_eval_count') if a.engine=='ollama' else body.get('usage',{}).get('prompt_tokens')}
            with logpath.open('a',encoding='utf-8') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            state={'completed':end,'seconds':state['seconds']+elapsed};save(statepath,state)
            if end//512!=begin//512 or begin==0 or end==len(texts):
                print(json.dumps({'engine':a.engine,'api_role':role,'completed_texts':end,'total':len(texts)}),flush=True)
            begin=end
        return values,state['seconds']
    tokenizer=AutoTokenizer.from_pretrained(model_path('BAAI/bge-m3'),trust_remote_code=False)
    ql=np.load(OUT/'query_token_lengths.npy')
    qm,qt=encode([c['model_query'] for c in cases[:MAIN_N]],'query_main_5000',ql[:MAIN_N])
    qr,qrt=encode([c['model_query'] for c in cases[MAIN_N:]],'paired_reference_500',ql[MAIN_N:])
    q=np.vstack([qm,qr]);np.save(cache/'query_all_5500.npy',q)
    dq,dqt=encode([doc_text(d,'question') for d in docs],'corpus_question',lengths(tokenizer,[doc_text(d,'question') for d in docs]))
    dqa,dqat=encode([doc_text(d,'question_answer') for d in docs],'corpus_question_answer',lengths(tokenizer,[doc_text(d,'question_answer') for d in docs]))
    sq=q@dq.T;sqa=q@dqa.T
    np.save(cache/'A-scores.npy',sq);np.save(cache/'B-scores.npy',sqa)
    requests_log=lines(logpath)
    success=[r for r in requests_log if r.get('success') and r['role']=='query_main_5000']
    references=[r for r in requests_log if r.get('success') and r['role']=='paired_reference_500']
    assert {i for r in success for i in range(r['first_row'],r['first_row']+r['texts'])}==set(range(MAIN_N))
    assert {i for r in references for i in range(r['first_row'],r['first_row']+r['texts'])}==set(range(REFERENCE_N))
    execution={'dense_backend':a.engine+'_HTTP_API','main_question_rows_sent':sum(r['texts'] for r in success),
        'reference_question_rows_sent':sum(r['texts'] for r in references),'query_rows_deduplicated':0,
        'http_inference_seconds':qt+qrt+dqt+dqat,'main_query_http_seconds':qt,'reference_query_http_seconds':qrt,
        'main_queries_per_second':MAIN_N/qt,'document_http_seconds':dqt+dqat,
        'document_views':['question','question_answer'],'hf_dense_fallback':False,'raw_vector_dimensions':1024,
        'truncated_inputs':0,'dense_input_limit':MAX_LENGTH,'request_failures':sum(not r.get('success') for r in requests_log),
        'batch_latency_quantiles':{str(p):float(np.quantile([r['latency_seconds'] for r in success],p)) for p in [.5,.95,.99]},
        'latency_scope':'sequential variable-size batches, not a concurrency/load benchmark'}
    execution['ollama_physical_batch_policy']='2048 for inputs <=2048 tokens; 4096 for longer full documents; num_ctx=8192; truncate=false' if a.engine=='ollama' else None
    save(base/'embedding_execution.json',execution)
    if not (base/'A/summary.json').exists():summarize(a.engine,'A',cases,docs,rankings(sq),sq,execution)
    if not (base/'B/summary.json').exists():summarize(a.engine,'B',cases,docs,rankings(sqa),sqa,execution)
    candidates=pools(sq,sqa,20);save(cache/'C-pools.json',candidates)
    summarize(a.engine,'C-M20',cases,docs,candidates,np.maximum(sq,sqa),{**execution,'candidate_budget':20,'merge_rule':'top20 each view, dedupe FAQ IDs, max cosine, top20'})

def rerank_documents(docs,view):
    if view=='question':
        return [doc_text(d,view) for d in docs],{i:[i] for i in range(len(docs))}
    windows=lines(DATA/'reranker_chunks.jsonl')
    positions={d['faq_id']:i for i,d in enumerate(docs)}
    mapping=defaultdict(list)
    for i,w in enumerate(windows):mapping[positions[w['faq_id']]].append(i)
    assert set(mapping)==set(range(len(docs)))
    return [w['text'] for w in windows],mapping

def merge_window_scores(scores,qi,di,values,winners=None,window_indices=None):
    # Several chunk logits may target one query/FAQ cell in the same batch.
    previous=scores[qi,di].copy() if winners is not None else None
    np.maximum.at(scores,(qi,di),values)
    if winners is not None:
        assert window_indices is not None
        improved=scores[qi,di]>previous
        winners[qi[improved],di[improved]]=np.iinfo(np.int16).max
        is_max=values==scores[qi,di]
        # Equal scores choose the earliest source window deterministically.
        np.minimum.at(winners,(qi[is_max],di[is_max]),window_indices[is_max])

def rerank(a):
    torch=torch_setup()
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    cases,docs=inputs();base=folder(a.engine)
    candidates=read(base/'cache/C-pools.json')
    name=f'D-{a.reranker}-{a.view}-M20'
    assert not (base/name/'summary.json').exists()
    modelname='BAAI/bge-reranker-v2-m3' if a.reranker=='bge' else 'jinaai/jina-reranker-v2-base-multilingual'
    path=model_path(modelname)
    model=AutoModelForSequenceClassification.from_pretrained(path,trust_remote_code=a.reranker=='jina',torch_dtype=torch.float32,
        **({'use_flash_attn':False} if a.reranker=='jina' else {})).to('cuda').eval()
    tokenizer=AutoTokenizer.from_pretrained(path,trust_remote_code=False)
    queries=[c['model_query'] for c in cases]
    documents,mapping=rerank_documents(docs,a.view)
    ql,dl=lengths(tokenizer,queries),lengths(tokenizer,documents)
    probe_cases=sorted(set(map(int,np.linspace(0,MAIN_N-1,15)))|set(map(int,np.argsort(ql[:MAIN_N])[-3:])))
    probe_pairs=[(i,mapping[j][0]) for i in probe_cases for j in candidates[i][:2]]
    def probe_values():
        values=[];torch.cuda.synchronize();start=time.perf_counter()
        for b in range(0,len(probe_pairs),4):
            pairs=probe_pairs[b:b+4]
            encoded=tokenizer([[queries[i],documents[j]] for i,j in pairs],padding=True,truncation=False,return_tensors='pt').to('cuda')
            assert encoded['input_ids'].shape[1]<=RERANK_MAX_LENGTH
            with torch.inference_mode():values.extend(model(**encoded).logits.reshape(-1).float().cpu().tolist())
        torch.cuda.synchronize()
        return np.asarray(values),time.perf_counter()-start
    fp32,fp32_seconds=probe_values()
    model.half();fp16,fp16_seconds=probe_values()
    assert np.isfinite(fp32).all() and np.isfinite(fp16).all()
    probe={'pairs':len(probe_pairs),'max_absolute_logit_difference':float(np.max(np.abs(fp32-fp16))),
        'mean_absolute_logit_difference':float(np.mean(np.abs(fp32-fp16))),
        'top2_representative_window_order_flips':int(np.sum(np.argmax(fp32.reshape(-1,2),axis=1)!=np.argmax(fp16.reshape(-1,2),axis=1))),
        'fp32_seconds':fp32_seconds,'fp16_seconds':fp16_seconds,
        'scope':'representative first windows, numerical diagnostic only; not proof of equal accuracy or final FAQ max-window rank'}
    save(base/'cache'/f'{name}-precision-probe.json',probe)
    print(json.dumps({'engine':a.engine,'structure':name,'precision_probe':probe}),flush=True)
    pairs=[(i,j,w) for i,ids in enumerate(candidates) for j in ids for w in mapping[j]]
    qi,di,wi=np.asarray(pairs,dtype=np.int32).T;del pairs
    estimated=ql[qi]+dl[wi];order=np.argsort(estimated,kind='stable')
    scores=mmap(base/'cache'/f'{name}-scores.npy',(N,len(docs)),fill=-1e20)
    winners=None
    if a.view=='question_answer':
        winnerpath=base/'cache'/f'{name}-winning-windows.npy'
        if winnerpath.exists():
            winners=np.load(winnerpath,mmap_mode='r+')
            assert winners.shape==(N,len(docs)) and winners.dtype==np.int16
        else:
            winners=np.lib.format.open_memmap(winnerpath,mode='w+',dtype=np.int16,shape=(N,len(docs)))
            winners[:]=np.iinfo(np.int16).max;winners.flush()
    statepath=base/'cache'/f'{name}-progress.json'
    state=read(statepath) if statepath.exists() else {'completed':0,'seconds':0.,'max_pair_tokens':0,'batches':0}
    identity=hashlib.sha256(order.tobytes()+qi.tobytes()+di.tobytes()+wi.tobytes()).hexdigest()
    assert state.get('order_hash',identity)==identity,'Resume order/window identity changed'
    begin=state['completed'];last_report=time.perf_counter();start=time.perf_counter()
    while begin<len(order):
        end=min(begin+96,len(order))
        while end>begin+1 and (end-begin)*int(max(estimated[order[begin:end]]))>8192:end-=1
        ix=order[begin:end]
        encoded=tokenizer([[queries[i],documents[w]] for i,w in zip(qi[ix],wi[ix])],padding=True,truncation=False,return_tensors='pt')
        width=int(encoded['input_ids'].shape[1]);assert width<=RERANK_MAX_LENGTH
        with torch.inference_mode():values=model(**encoded.to('cuda')).logits.reshape(-1).float().cpu().numpy()
        assert values.shape==(len(ix),) and np.isfinite(values).all()
        merge_window_scores(scores,qi[ix],di[ix],values,winners,wi[ix] if winners is not None else None)
        state['max_pair_tokens']=max(state['max_pair_tokens'],width);state['batches']+=1
        if end//4800!=begin//4800 or end==len(order):
            torch.cuda.synchronize();scores.flush()
            if winners is not None:winners.flush()
            now=time.perf_counter()
            state.update(completed=end,seconds=state['seconds']+now-start,order_hash=identity,total_pairs=len(order));save(statepath,state)
            start=now
            if now-last_report>25 or end==len(order):
                print(json.dumps({'engine':a.engine,'structure':name,'completed_window_pairs':end,'total_window_pairs':len(order),'candidate_faq_pairs':N*20,'seconds':round(state['seconds'],1),'max_pair_tokens':state['max_pair_tokens']}),flush=True)
                last_report=now
        begin=end
    ranked=[sorted(map(int,ids),key=lambda j:(-float(scores[i,j]),j)) for i,ids in enumerate(candidates)]
    assert np.all(scores[np.repeat(np.arange(N),20),np.asarray(candidates).reshape(-1)]>-1e19)
    extra={'dense_backend':a.engine+'_HTTP_API','reranker_backend':'common_HF_PyTorch_CUDA_float16',
        'reranker_model':modelname,'reranker_view':a.view,'main_rows_scored':MAIN_N,'reference_rows_scored':REFERENCE_N,
        'candidate_faq_pairs':N*20,'window_pairs_scored':len(order),'pairs_deduplicated':0,'repeat_rows':0,
        'input_limit':RERANK_MAX_LENGTH,'max_pair_tokens':state['max_pair_tokens'],'truncated_pairs':0,
        'window_aggregation':'max raw logit per FAQ' if a.view=='question_answer' else 'one question view per FAQ',
        'batching':'stable length buckets; <=96 pairs and <=8192 estimated padded tokens',
        'reranker_seconds':state['seconds'],'peak_torch_vram_bytes':int(torch.cuda.max_memory_allocated()),
        'all_components_served_by_engine':False,'score_adapter':'raw_logit_identity','resume_order_hash':identity,'precision_probe':probe}
    summarize(a.engine,name,cases,docs,ranked,scores,extra)
    from candidate_logs import export_candidate_log
    export_candidate_log(a.engine,name,a.view,cases,docs)
    del model;gc.collect();torch.cuda.empty_cache()

def sparse_matrix(weights):
    from scipy.sparse import csr_matrix
    rows,cols,values=[],[],[]
    for i,weight in enumerate(weights):
        for token,score in weight.items():rows.append(i);cols.append(int(token));values.append(float(score))
    return csr_matrix((values,(rows,cols)),shape=(len(weights),250002),dtype=np.float32)

def maxsim_batch(torch,queries,buckets,ndocs):
    result=np.empty((len(queries),ndocs),dtype=np.float32)
    order=sorted(range(len(queries)),key=lambda i:len(queries[i]))
    with torch.inference_mode():
        for begin in range(0,len(order),8):
            ix=order[begin:begin+8]
            ql=[len(queries[i]) for i in ix];dim=queries[ix[0]].shape[1]
            qq=torch.zeros((len(ix),max(ql),dim),device='cuda');qmask=torch.zeros((len(ix),max(ql)),device='cuda')
            for b,i in enumerate(ix):qq[b,:ql[b]]=torch.as_tensor(queries[i],device='cuda');qmask[b,:ql[b]]=1
            for dx,dd,dmask in buckets:
                sim=torch.einsum('bqd,ctd->bcqt',qq,dd).masked_fill(~dmask[None,:,None,:],-torch.inf)
                vals=(sim.max(dim=-1).values*qmask[:,None,:]).sum(dim=-1)/torch.as_tensor(ql,device='cuda')[:,None]
                result[np.ix_(ix,dx)]=vals.cpu().numpy()
    return result

def native(a):
    torch=torch_setup()
    from FlagEmbedding import BGEM3FlagModel
    from transformers import AutoTokenizer
    cases,docs=inputs();base=folder(a.engine);cache=base/'cache'
    assert not (base/'F-dense-pool-M20/summary.json').exists()
    model=BGEM3FlagModel(model_path('BAAI/bge-m3'),use_fp16=False,devices='cuda:0',batch_size=8,query_max_length=MAX_LENGTH,passage_max_length=MAX_LENGTH)
    tokenizer=AutoTokenizer.from_pretrained(model_path('BAAI/bge-m3'),trust_remote_code=False)
    texts=[doc_text(d,'question_answer') for d in docs];dl=lengths(tokenizer,texts)
    assert dl.max()<=MAX_LENGTH
    feature_docs={'lexical_weights':[None]*len(docs),'colbert_vecs':[None]*len(docs)}
    doc_order=np.argsort(dl,kind='stable');begin=0;start=time.perf_counter()
    while begin<len(docs):
        end=min(begin+16,len(docs))
        while end>begin+1 and (end-begin)*int(max(dl[doc_order[begin:end]]))>4096:end-=1
        ix=doc_order[begin:end]
        encoded=model.encode([texts[i] for i in ix],batch_size=len(ix),max_length=MAX_LENGTH,return_dense=False,return_sparse=True,return_colbert_vecs=True)
        for key in feature_docs:
            for j,i in enumerate(ix):feature_docs[key][int(i)]=encoded[key][j]
        begin=end
    torch.cuda.synchronize();doc_seconds=time.perf_counter()-start
    doc_sparse=sparse_matrix(feature_docs['lexical_weights'])
    buckets=[];order=sorted(range(len(docs)),key=lambda i:len(feature_docs['colbert_vecs'][i]));begin=0
    while begin<len(docs):
        end=min(begin+32,len(docs))
        while end>begin+1 and (end-begin)*max(len(feature_docs['colbert_vecs'][i]) for i in order[begin:end])>4096:end-=1
        ix=order[begin:end];length=max(len(feature_docs['colbert_vecs'][i]) for i in ix);dim=feature_docs['colbert_vecs'][ix[0]].shape[1]
        vv=torch.zeros((len(ix),length,dim),device='cuda');mask=torch.zeros((len(ix),length),device='cuda',dtype=torch.bool)
        for j,i in enumerate(ix):v=feature_docs['colbert_vecs'][i];vv[j,:len(v)]=torch.as_tensor(v,device='cuda');mask[j,:len(v)]=True
        buckets.append((ix,vv,mask));begin=end
    sparse=mmap(cache/'native-sparse.npy',(N,len(docs)))
    scores=mmap(cache/'native-colbert.npy',(N,len(docs)))
    statepath=cache/'native-progress.json'
    state=read(statepath) if statepath.exists() else {'completed':0,'encode_seconds':0.,'maxsim_seconds':0.}
    for begin in range(state['completed'],N,128):
        end=min(begin+128,N);start=time.perf_counter()
        queries=model.encode([c['model_query'] for c in cases[begin:end]],batch_size=8,max_length=MAX_LENGTH,return_dense=False,return_sparse=True,return_colbert_vecs=True)
        torch.cuda.synchronize();state['encode_seconds']+=time.perf_counter()-start
        sparse[begin:end]=(sparse_matrix(queries['lexical_weights'])@doc_sparse.T).toarray()
        start=time.perf_counter();scores[begin:end]=maxsim_batch(torch,queries['colbert_vecs'],buckets,len(docs))
        torch.cuda.synchronize();state['maxsim_seconds']+=time.perf_counter()-start
        if begin==0:
            for qi in range(3):
                for di in range(3):
                    assert np.isclose(sparse[qi,di],model.compute_lexical_matching_score(queries['lexical_weights'][qi],feature_docs['lexical_weights'][di]),atol=1e-5)
                    assert np.isclose(scores[qi,di],model.colbert_score(queries['colbert_vecs'][qi],feature_docs['colbert_vecs'][di]),atol=2e-5)
            state['official_reference_3x3_passed']=True
        sparse.flush();scores.flush();state['completed']=end;save(statepath,state)
        print(json.dumps({'engine':a.engine,'structure':'E/F','completed_queries':end,'main_queries':MAIN_N,'reference_queries':REFERENCE_N,'encode_seconds':round(state['encode_seconds'],1),'maxsim_seconds':round(state['maxsim_seconds'],1)}),flush=True)
        del queries
    assert state['official_reference_3x3_passed'] and np.isfinite(sparse).all() and np.isfinite(scores).all()
    extra={'dense_backend':a.engine+'_HTTP_API','native_backend':'common_FlagEmbedding_CUDA_float32',
        'all_components_served_by_engine':False,'main_rows_scored':MAIN_N,'reference_rows_scored':REFERENCE_N,
        'native_encode_seconds':doc_seconds+state['encode_seconds'],'late_interaction_seconds':state['maxsim_seconds'],
        'native_features_reused_from_other_engine':False,'truncated_inputs':0,'input_limit':MAX_LENGTH,
        'official_reference_3x3_passed':True,'peak_torch_vram_bytes':int(torch.cuda.max_memory_allocated())}
    sr=rankings(sparse);dense=np.load(cache/'B-scores.npy',mmap_mode='r');dr=rankings(dense)
    if not (base/'E-sparse/summary.json').exists():summarize(a.engine,'E-sparse',cases,docs,sr,sparse,{**extra,'dense_backend':None,'scope':'common native sparse control'})
    rrf=np.zeros_like(dense);ranks=1/(60+np.arange(1,len(docs)+1))
    for i in range(N):rrf[i,dr[i]]+=ranks;rrf[i,sr[i]]+=ranks
    ranked=[sorted(set(map(int,dr[i,:20]))|set(map(int,sr[i,:20])),key=lambda j:(-float(rrf[i,j]),j))[:20] for i in range(N)]
    if not (base/'E-dense-sparse-RRF-M20/summary.json').exists():summarize(a.engine,'E-dense-sparse-RRF-M20',cases,docs,ranked,rrf,{**extra,'rrf_constant':60,'candidate_budget':20})
    del rrf;gc.collect()
    if not (base/'F-full/summary.json').exists():summarize(a.engine,'F-full',cases,docs,rankings(scores),scores,{**extra,'dense_backend':None,'scope':'common native full-corpus control; excluded from selection'})
    ranked=[sorted(map(int,dr[i,:20]),key=lambda j:(-float(scores[i,j]),j)) for i in range(N)]
    summarize(a.engine,'F-dense-pool-M20',cases,docs,ranked,scores,{**extra,'candidate_budget':20,'candidate_source':'this engine question-answer dense'})

def finish(a):
    cases,docs=inputs();base=folder(a.engine);summaries={};policies={}
    for name in NAMES:
        summaries[name]=read(base/name/'summary.json')
        assert summaries[name]['main_case_count']==MAIN_N and summaries[name]['case_count']==N
        rows=lines(base/name/'per_query.jsonl')
        assert [r['case_id'] for r in rows]==[c['case_id'] for c in cases]
        policies[name]=calibrate_policy(rows,cases)
    allowed=[n for n in NAMES if n not in ['E-sparse','F-full']]
    def key(name):
        s=summaries[name]['cohort']
        return (-s['general']['parent_mean']['all_sources@3'],-s['multi_intent']['parent_mean']['all_sources@3'],
                -s['condition']['parent_mean']['all_sources@3'],-s['general']['parent_mean']['hit@1'],name)
    selected=sorted(allowed,key=key)[0]
    save(base/'policies.json',policies)
    save(base/'selection.json',{'selected':selected,'policy':policies[selected],
        'rule':read(DATA/'manifest.json')['protocol']['selection'],'all_5000_main_rows_executed':True,
        'reference_rows':REFERENCE_N,'stages':len(NAMES),'source_manifest_sha256':sha(DATA/'manifest.json'),
        'ranking_diagnostics_only':True,'holdout_used':False,'prior_results_or_thresholds_used':False,
        'no_faq_threshold_fitted':False,'all_components_served_by_engine':False})
    print(json.dumps({'engine':a.engine,'main_rows':MAIN_N,'reference_rows':REFERENCE_N,'stages':len(NAMES),'selected':selected,'threshold_status':policies[selected]['status']}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validate','embeddings','D','native','finish'],required=True)
    p.add_argument('--engine',choices=['ollama','vllm']);p.add_argument('--url')
    p.add_argument('--reranker',choices=['bge','jina']);p.add_argument('--view',choices=['question','question_answer'])
    a=p.parse_args()
    if a.stage=='validate':validate()
    else:
        assert a.engine
        {'embeddings':embeddings,'D':rerank,'native':native,'finish':finish}[a.stage](a)
if __name__=='__main__':main()
