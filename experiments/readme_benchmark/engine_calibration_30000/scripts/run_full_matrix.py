"""Full 30k matched matrices, no input truncation or inference deduplication."""
import argparse,gc,hashlib,json,time
from collections import Counter
import numpy as np
from common import ROOT,OUT,N,MAX_LENGTH,NAMES,inputs,read,save,save_lines,sha,model_path,doc_text,rankings,pools,summarize,calibrate_policy,lines

def folder(engine):return OUT/'engines'/engine
def torch_setup():
    import torch
    assert torch.cuda.is_available()
    torch.manual_seed(20261006);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return torch
def lengths(tok,texts):
    result=[]
    for begin in range(0,len(texts),256):
        batch=tok(texts[begin:begin+256],truncation=False,return_length=True)
        result.extend(batch['length'])
    return np.asarray(result,dtype=np.int32)
def mmap(path,shape,fill=None):
    if path.exists():
        v=np.load(path,mmap_mode='r+');assert v.shape==shape and v.dtype==np.float32
    else:
        v=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=shape)
        if fill is not None:v[:]=fill;v.flush()
    return v
def validate():
    from transformers import AutoTokenizer
    cs,ds=inputs();manifest=read(ROOT/'data/manifest.json')
    assert Counter(c['type'] for c in cs)==manifest['type_counts']
    parents={}
    for c in cs:parents.setdefault(c['parent_case_id'],[]).append(c)
    assert len(parents)==3000 and all(len(v)==10 and len({c['query'] for c in v})==10 for v in parents.values())
    assert len({c['case_id'] for c in cs})==N
    assert all(c['model_query'].endswith(c['query']) for c in cs)
    tok=AutoTokenizer.from_pretrained(model_path('BAAI/bge-m3'))
    ql=lengths(tok,[c['model_query'] for c in cs]);dl=lengths(tok,[doc_text(d,'question_answer') for d in ds])
    assert max(ql.max(),dl.max())<=MAX_LENGTH
    OUT.mkdir(parents=True,exist_ok=True);np.save(OUT/'query_token_lengths.npy',ql)
    value={'rows':N,'queries_max_tokens':int(ql.max()),'qa_docs_max_tokens':int(dl.max()),
           'over_old_512_limit':int((ql>512).sum()),'common_input_limit':MAX_LENGTH,'truncation':False,
           'full_variant_input_hash':sha(ROOT/'data/cases.jsonl'),'provided_context_is_not_model_input':True,
           'all_15_types_retained':True,'repeat_rows':2000,'holdout_used':False,
           'independent_annotation_complete':False,'tei_model_capacity':8192}
    save(OUT/'input_validation.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)

def embeddings(a):
    import requests
    cs,ds=inputs();base=folder(a.engine);cache=base/'cache';cache.mkdir(parents=True,exist_ok=True)
    assert not (base/'C-M20/summary.json').exists(),'Dense matrix already completed'
    session=requests.Session();logpath=base/'embedding_requests.jsonl'
    def encode(texts,role,lens):
        path=cache/f'{role}.npy';values=mmap(path,(len(texts),1024))
        statepath=cache/f'{role}-progress.json';state=read(statepath) if statepath.exists() else {'completed':0,'seconds':0.}
        begin=state['completed']
        while begin<len(texts):
            end=min(begin+16,len(texts))
            while (end-begin)*int(max(lens[begin:end]))>4096:end-=1
            chunk=texts[begin:end]
            if a.engine=='ollama':endpoint='/api/embed';payload={'model':'bge-m3','input':chunk,'truncate':False,'keep_alive':'30m','options':{'num_ctx':MAX_LENGTH}}
            elif a.engine=='tei':endpoint='/embed';payload={'inputs':chunk,'truncate':False}
            else:endpoint='/v1/embeddings';payload={'model':'BAAI/bge-m3','input':chunk,'encoding_format':'float'}
            start=time.perf_counter()
            try:
                r=session.post(a.url.rstrip('/')+endpoint,json=payload,timeout=300);r.raise_for_status();body=r.json()
                vv=body['embeddings'] if a.engine=='ollama' else body if a.engine=='tei' else [x['embedding'] for x in sorted(body['data'],key=lambda x:x['index'])]
                vv=np.asarray(vv,dtype=np.float32);assert vv.shape==(len(chunk),1024) and np.isfinite(vv).all()
                norm=np.linalg.norm(vv,axis=1);assert np.all(norm>0)
            except Exception as exc:
                with logpath.open('a',encoding='utf-8') as f:f.write(json.dumps({'role':role,'first_row':begin,'texts':len(chunk),'success':False,'error':str(exc)})+'\n')
                raise
            elapsed=time.perf_counter()-start
            values[begin:end]=vv/norm[:,None];values.flush()
            record={'role':role,'first_row':begin,'texts':len(chunk),'endpoint':endpoint,'success':True,
                    'input_sha256':hashlib.sha256(json.dumps(chunk,ensure_ascii=False).encode()).hexdigest(),
                    'latency_seconds':elapsed,'status':r.status_code,'max_input_tokens':int(max(lens[begin:end])),'truncation_requested':False}
            with logpath.open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            state={'completed':end,'seconds':state['seconds']+elapsed};save(statepath,state)
            if end//256!=begin//256 or begin==0 or end==len(texts):print(json.dumps({'engine':a.engine,'api_role':role,'completed_texts':end,'total':len(texts)}),flush=True)
            begin=end
        return values,state['seconds']
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model_path('BAAI/bge-m3'))
    q,qt=encode([c['model_query'] for c in cs],'query_all_30000',np.load(OUT/'query_token_lengths.npy'))
    dq,dqt=encode([doc_text(d,'question') for d in ds],'corpus_question',lengths(tok,[doc_text(d,'question') for d in ds]))
    dqa,dqat=encode([doc_text(d,'question_answer') for d in ds],'corpus_question_answer',lengths(tok,[doc_text(d,'question_answer') for d in ds]))
    sq=q@dq.T;sqa=q@dqa.T
    np.save(cache/'A-scores.npy',sq);np.save(cache/'B-scores.npy',sqa)
    records=lines(logpath);success=[r for r in records if r.get('success') and r['role']=='query_all_30000']
    covered={i for r in success for i in range(r['first_row'],r['first_row']+r['texts'])};assert covered==set(range(N))
    execution={'dense_backend':a.engine+'_HTTP_API','question_rows_sent':sum(r['texts'] for r in success),
               'distinct_execution_rows':len(covered),'query_rows_deduplicated':0,'provided_context_used':False,
               'http_inference_seconds':qt+dqt+dqat,'query_http_seconds':qt,'document_views':['question','question_answer'],
               'hf_dense_fallback':False,'raw_vector_dimensions':1024,'reranker':None,'truncated_inputs':0,
               'request_failures':sum(not r.get('success') for r in records),'batch_latency_quantiles':
               {str(x):float(np.quantile([r['latency_seconds'] for r in success],x)) for x in [.5,.95,.99]},
               'latency_scope':'sequential variable-size batches, not concurrency/load benchmark'}
    save(base/'embedding_execution.json',execution)
    if not (base/'A/summary.json').exists():summarize(a.engine,'A',cs,ds,rankings(sq),sq,execution)
    if not (base/'B/summary.json').exists():summarize(a.engine,'B',cs,ds,rankings(sqa),sqa,execution)
    pp=pools(sq,sqa,20);save(cache/'C-pools.json',pp)
    summarize(a.engine,'C-M20',cs,ds,pp,np.maximum(sq,sqa),{**execution,'candidate_budget':20,'merge_rule':'top20 each view, dedupe FAQ IDs, max cosine, top20'})

def rerank(a):
    torch=torch_setup()
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    cs,ds=inputs();base=folder(a.engine);pp=read(base/'cache/C-pools.json')
    name=f'D-{a.reranker}-{a.view}-M20';assert not (base/name/'summary.json').exists()
    modelname='BAAI/bge-reranker-v2-m3' if a.reranker=='bge' else 'jinaai/jina-reranker-v2-base-multilingual'
    path=model_path(modelname)
    model=AutoModelForSequenceClassification.from_pretrained(path,trust_remote_code=a.reranker=='jina',torch_dtype=torch.float32,
            **({'use_flash_attn':False} if a.reranker=='jina' else {})).to('cuda').eval()
    tok=AutoTokenizer.from_pretrained(path,trust_remote_code=a.reranker=='jina')
    queries=[c['model_query'] for c in cs];documents=[doc_text(d,a.view) for d in ds]
    ql=lengths(tok,queries);dl=lengths(tok,documents)
    # Diagnostic only: all actual matrix scores below use uniform FP16.
    probe_cases=sorted(set(map(int,np.linspace(0,N-1,15)))|set(map(int,np.argsort(ql)[-3:])))
    probe_pairs=[(i,j) for i in probe_cases for j in pp[i][:2]]
    def probe_values():
        values=[];torch.cuda.synchronize();started=time.perf_counter()
        for b in range(0,len(probe_pairs),4):
            ix=probe_pairs[b:b+4]
            encoded=tok([[queries[i],documents[j]] for i,j in ix],padding=True,truncation=False,return_tensors='pt').to('cuda')
            assert encoded['input_ids'].shape[1]<=MAX_LENGTH
            with torch.inference_mode():values.extend(model(**encoded).logits.reshape(-1).float().cpu().tolist())
        torch.cuda.synchronize();return np.array(values),time.perf_counter()-started
    fp32,fp32_seconds=probe_values();model.half();fp16,fp16_seconds=probe_values()
    assert np.isfinite(fp32).all() and np.isfinite(fp16).all()
    probe={'pairs':len(probe_pairs),'case_ids':[cs[i]['case_id'] for i in probe_cases],
           'max_absolute_logit_difference':float(np.max(np.abs(fp32-fp16))),
           'mean_absolute_logit_difference':float(np.mean(np.abs(fp32-fp16))),
           'top2_order_flips':int(np.sum(np.argmax(fp32.reshape(-1,2),axis=1)!=np.argmax(fp16.reshape(-1,2),axis=1))),
           'fp32_seconds':fp32_seconds,'fp16_seconds':fp16_seconds,
           'scope':'numerical diagnostic only; not proof of equal retrieval accuracy; not used to fit threshold'}
    save(base/'cache'/f'{name}-precision-probe.json',probe)
    print(json.dumps({'engine':a.engine,'structure':name,'precision_probe':probe}),flush=True)
    qi=np.repeat(np.arange(N,dtype=np.int32),20);di=np.asarray(pp,dtype=np.int32).reshape(-1)
    estimated=ql[qi]+dl[di];order=np.argsort(estimated,kind='stable')
    scores=mmap(base/'cache'/f'{name}-scores.npy',(N,len(ds)),fill=-1e20)
    statepath=base/'cache'/f'{name}-progress.json'
    state=read(statepath) if statepath.exists() else {'completed':0,'seconds':0.,'max_pair_tokens':0,'batches':0}
    identity=hashlib.sha256(order.tobytes()+qi.tobytes()+di.tobytes()).hexdigest()
    assert state.get('order_hash',identity)==identity,'Resume candidate/order identity changed'
    begin=state['completed'];last_report=time.perf_counter();start=time.perf_counter()
    while begin<len(order):
        end=min(begin+96,len(order))
        while (end-begin)*int(max(estimated[order[begin:end]]))>6144:end-=1
        ix=order[begin:end];texts=[[queries[i],documents[j]] for i,j in zip(qi[ix],di[ix])]
        encoded=tok(texts,padding=True,truncation=False,return_tensors='pt')
        width=int(encoded['input_ids'].shape[1]);assert width<=MAX_LENGTH,'Reranker pair would be truncated'
        encoded=encoded.to('cuda')
        with torch.inference_mode():vals=model(**encoded).logits.reshape(-1).float().cpu().numpy()
        assert vals.shape==(len(ix),) and np.isfinite(vals).all()
        scores[qi[ix],di[ix]]=vals
        state['max_pair_tokens']=max(state['max_pair_tokens'],width);state['batches']+=1
        if end//4800!=begin//4800 or end==len(order):
            torch.cuda.synchronize();scores.flush();now=time.perf_counter()
            state.update(completed=end,seconds=state['seconds']+now-start,order_hash=identity)
            save(statepath,state);start=now
            print(json.dumps({'engine':a.engine,'structure':name,'pairs_completed':end,'pairs_total':len(order),
                              'elapsed_seconds':round(state['seconds'],1),'max_pair_tokens':state['max_pair_tokens']}),flush=True);last_report=now
        elif time.perf_counter()-last_report>30:
            print(json.dumps({'engine':a.engine,'structure':name,'pairs_completed':end,'pairs_total':len(order)}),flush=True);last_report=time.perf_counter()
        begin=end
    rr=[sorted(p,key=lambda j:(-float(scores[i,j]),j)) for i,p in enumerate(pp)]
    assert np.all(scores[qi,di]>-1e19)
    extra={'dense_backend':a.engine+'_HTTP_API','reranker_backend':'common_HF_PyTorch_CUDA_float16',
           'reranker_model':modelname,'reranker_view':a.view,'rows_scored':N,'pairs_scored':len(order),
           'pairs_deduplicated':0,'repeat_rows_actually_rescored':2000,'input_limit':MAX_LENGTH,
           'max_pair_tokens':state['max_pair_tokens'],'truncated_pairs':0,'batching':'stable token-length buckets; <=96 pairs, <=6144 padded tokens',
           'reranker_seconds':state['seconds'],'peak_torch_vram_bytes':int(torch.cuda.max_memory_allocated()),
           'all_components_served_by_engine':False,'score_adapter':'raw_logit_identity','resume_order_hash':identity,'precision_probe':probe}
    summarize(a.engine,name,cs,ds,rr,scores,extra)
    del model;gc.collect();torch.cuda.empty_cache()

def sparse_matrix(weights):
    from scipy.sparse import csr_matrix
    rows=[];cols=[];values=[]
    for i,w in enumerate(weights):
        for token,score in w.items():rows.append(i);cols.append(int(token));values.append(float(score))
    return csr_matrix((values,(rows,cols)),shape=(len(weights),250002),dtype=np.float32)

def maxsim_batch(torch,queries,buckets,ndocs):
    """Batched MaxSim excludes document pads and averages only real query tokens."""
    result=np.empty((len(queries),ndocs),dtype=np.float32)
    order=sorted(range(len(queries)),key=lambda i:len(queries[i]))
    with torch.inference_mode():
        for begin in range(0,len(order),8):
            ix=order[begin:begin+8];ql=[len(queries[i]) for i in ix];dim=queries[ix[0]].shape[1]
            vv=torch.zeros((len(ix),max(ql),dim),device='cuda');qmask=torch.zeros((len(ix),max(ql)),device='cuda')
            for b,i in enumerate(ix):vv[b,:ql[b]]=torch.as_tensor(queries[i],device='cuda');qmask[b,:ql[b]]=1
            for dx,dd,dmask in buckets:
                sim=torch.einsum('bqd,ctd->bcqt',vv,dd).masked_fill(~dmask[None,:,None,:],-torch.inf)
                score=(sim.max(dim=-1).values*qmask[:,None,:]).sum(dim=-1)/torch.as_tensor(ql,device='cuda')[:,None]
                result[np.ix_(ix,dx)]=score.cpu().numpy()
    return result

def native(a):
    torch=torch_setup()
    from FlagEmbedding import BGEM3FlagModel
    cs,ds=inputs();base=folder(a.engine);cache=base/'cache'
    assert not (base/'F-dense-pool-M20/summary.json').exists(),'Native matrix completed'
    model=BGEM3FlagModel(model_path('BAAI/bge-m3'),use_fp16=False,devices='cuda:0',batch_size=8,
                         query_max_length=MAX_LENGTH,passage_max_length=MAX_LENGTH)
    start=time.perf_counter()
    docs=model.encode([doc_text(d,'question_answer') for d in ds],batch_size=16,max_length=MAX_LENGTH,
                      return_dense=False,return_sparse=True,return_colbert_vecs=True)
    torch.cuda.synchronize();doc_seconds=time.perf_counter()-start;doc_sparse=sparse_matrix(docs['lexical_weights'])
    buckets=[];order=sorted(range(len(ds)),key=lambda i:len(docs['colbert_vecs'][i]))
    for begin in range(0,len(ds),32):
        ix=order[begin:begin+32];length=max(len(docs['colbert_vecs'][i]) for i in ix);dim=docs['colbert_vecs'][ix[0]].shape[1]
        vv=torch.zeros((len(ix),length,dim),device='cuda');mask=torch.zeros((len(ix),length),device='cuda',dtype=torch.bool)
        for j,i in enumerate(ix):v=docs['colbert_vecs'][i];vv[j,:len(v)]=torch.as_tensor(v,device='cuda');mask[j,:len(v)]=True
        buckets.append((ix,vv,mask))
    sparse=mmap(cache/'native-sparse.npy',(N,len(ds)))
    scores=mmap(cache/'native-colbert.npy',(N,len(ds)))
    statepath=cache/'native-progress.json';state=read(statepath) if statepath.exists() else {'completed':0,'encode_seconds':0.,'maxsim_seconds':0.}
    for begin in range(state['completed'],N,128):
        end=min(begin+128,N);start=time.perf_counter()
        queries=model.encode([c['model_query'] for c in cs[begin:end]],batch_size=8,max_length=MAX_LENGTH,
                             return_dense=False,return_sparse=True,return_colbert_vecs=True)
        torch.cuda.synchronize();state['encode_seconds']+=time.perf_counter()-start
        sparse[begin:end]=(sparse_matrix(queries['lexical_weights'])@doc_sparse.T).toarray()
        start=time.perf_counter();scores[begin:end]=maxsim_batch(torch,queries['colbert_vecs'],buckets,len(ds))
        torch.cuda.synchronize();state['maxsim_seconds']+=time.perf_counter()-start
        if begin==0:
            for qi in range(3):
                for di in range(3):
                    assert np.isclose(sparse[qi,di],model.compute_lexical_matching_score(queries['lexical_weights'][qi],docs['lexical_weights'][di]),atol=1e-5)
                    assert np.isclose(scores[qi,di],model.colbert_score(queries['colbert_vecs'][qi],docs['colbert_vecs'][di]),atol=2e-5)
            state['official_reference_3x3_passed']=True
        sparse.flush();scores.flush();state['completed']=end;save(statepath,state)
        print(json.dumps({'engine':a.engine,'structure':'E/F','completed_queries':end,'total':N,
                          'encode_seconds':round(state['encode_seconds'],1),'maxsim_seconds':round(state['maxsim_seconds'],1)}),flush=True)
        del queries
    assert state['official_reference_3x3_passed'] and np.isfinite(sparse).all() and np.isfinite(scores).all()
    extra={'dense_backend':a.engine+'_HTTP_API','native_backend':'common_FlagEmbedding_CUDA_float32',
           'all_components_served_by_engine':False,'rows_scored':N,'repeat_rows_actually_rescored':2000,
           'native_encode_seconds':doc_seconds+state['encode_seconds'],'late_interaction_seconds':state['maxsim_seconds'],
           'native_feature_cache_reused_from_other_engine':False,'truncated_inputs':0,'input_limit':MAX_LENGTH,
           'official_reference_3x3_passed':True,'peak_torch_vram_bytes':int(torch.cuda.max_memory_allocated())}
    sr=rankings(sparse);dense=np.load(cache/'B-scores.npy',mmap_mode='r');dr=rankings(dense)
    if not (base/'E-sparse/summary.json').exists():summarize(a.engine,'E-sparse',cs,ds,sr,sparse,{**extra,'dense_backend':None,'scope':'common native sparse control'})
    rrf=np.zeros_like(dense);rankscore=1/(60+np.arange(1,len(ds)+1))
    for i in range(N):rrf[i,dr[i]]+=rankscore;rrf[i,sr[i]]+=rankscore
    er=[sorted(set(map(int,dr[i,:20]))|set(map(int,sr[i,:20])),key=lambda j:(-float(rrf[i,j]),j))[:20] for i in range(N)]
    if not (base/'E-dense-sparse-RRF-M20/summary.json').exists():summarize(a.engine,'E-dense-sparse-RRF-M20',cs,ds,er,rrf,{**extra,'rrf_constant':60,'candidate_budget':20})
    del rrf;gc.collect()
    if not (base/'F-full/summary.json').exists():summarize(a.engine,'F-full',cs,ds,rankings(scores),scores,{**extra,'dense_backend':None,'scope':'common native full-corpus control; excluded from selection'})
    fr=[sorted(map(int,dr[i,:20]),key=lambda j:(-float(scores[i,j]),j)) for i in range(N)]
    summarize(a.engine,'F-dense-pool-M20',cs,ds,fr,scores,{**extra,'candidate_budget':20,'candidate_source':'this engine question-answer dense'})

def finish(a):
    cs,ds=inputs();base=folder(a.engine);summaries={};policies={};repeat={}
    for name in NAMES:
        summaries[name]=read(base/name/'summary.json');assert summaries[name]['case_count']==N
        rows=lines(base/name/'per_query.jsonl');assert [r['case_id'] for r in rows]==[c['case_id'] for c in cs]
        policies[name]=calibrate_policy(rows,cs);groups={}
        for c,r in zip(cs,rows):
            if c['type']=='RT':groups.setdefault(c['variation_repeat_group_id'],[]).append(r)
        assert len(groups)==200 and all(len(rr)==10 and len({r['query_hash'] for r in rr})==1 for rr in groups.values())
        repeat[name]={k:{'executions':len(rr),'rank_hashes':len({r['rank_hash'] for r in rr}),
                         'query_hash_count':len({r['query_hash'] for r in rr}),
                         'top1_score_range':float(np.ptp([r['top_scores'][0] for r in rr]))} for k,rr in groups.items()}
        print(json.dumps({'engine':a.engine,'calibrated':name,'f1':policies[name]['no_faq_f1']}),flush=True)
    allowed=[n for n in NAMES if n not in ['E-sparse','F-full']]
    key=lambda n:(-summaries[n]['cohort']['general']['family_mean']['all_sources@3'],
                  -summaries[n]['cohort']['condition']['family_mean']['all_sources@3'],
                  -summaries[n]['cohort']['general']['family_mean']['hit@1'],n)
    selected=sorted(allowed,key=key)[0]
    save(base/'policies.json',policies);save(base/'repeat_stability.json',repeat)
    save(base/'selection.json',{'selected':selected,'policy':policies[selected],
          'rule':'general family AllSourcesHit@3; ties by condition AllSourcesHit@3, general source Hit@1',
          'all_30000_rows_executed':True,'stages':len(NAMES),'source_manifest_sha256':sha(ROOT/'data/manifest.json'),
          'calibration_only':True,'holdout_used':False,'prior_results_or_thresholds_used':False,
          'all_components_served_by_engine':False})
    print(json.dumps({'engine':a.engine,'all_30000_rows_executed':True,'stages':len(NAMES),'selected':selected}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validate','embeddings','D','native','finish'],required=True)
    p.add_argument('--engine',choices=['ollama','vllm','tei']);p.add_argument('--url');p.add_argument('--reranker',choices=['bge','jina']);p.add_argument('--view',choices=['question','question_answer']);a=p.parse_args()
    if a.stage=='validate':validate()
    else:
        assert a.engine
        {'embeddings':embeddings,'D':rerank,'native':native,'finish':finish}[a.stage](a)
if __name__=='__main__':main()
