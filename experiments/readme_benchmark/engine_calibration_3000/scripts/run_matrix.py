"""A-F per engine, using all original rows and explicit common auxiliary models."""
import argparse,gc,hashlib,json,time
import numpy as np
from common import ROOT,OUT,inputs,read,save,save_lines,sha,model_path,doc_text,rankings,pools,summarize,calibrate_policy,lines

def folder(engine):return OUT/'engines'/engine
def torch_setup():
    import torch
    assert torch.cuda.is_available()
    torch.manual_seed(20261006);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return torch
def validate():
    from transformers import AutoTokenizer
    cs,ds=inputs();tok=AutoTokenizer.from_pretrained(model_path('BAAI/bge-m3'))
    ql=[len(tok(c['model_query'],truncation=False)['input_ids']) for c in cs]
    dl=[len(tok(doc_text(d,'question_answer'),truncation=False)['input_ids']) for d in ds]
    assert max(ql+dl)<=512,'Unchanged inputs exceed the common 512-token limit'
    assert [c['query'] for c in cs]==[r['사용자 질문'] for r in lines(ROOT/'data/source_rows.jsonl')]
    value={'rows':len(cs),'queries_max_tokens':max(ql),'qa_docs_max_tokens':max(dl),'questions_identical':True,
           'provided_context_is_not_model_input':True,'all_15_types_retained':True,'repeat_rows':sum(c['type']=='RT' for c in cs)}
    save(OUT/'input_validation.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)
def embeddings(a):
    import requests
    cs,ds=inputs();base=folder(a.engine);cache=base/'cache';cache.mkdir(parents=True,exist_ok=True)
    assert not (base/'A/summary.json').exists(),'Embedding matrix already completed'
    session=requests.Session();api_log=[]
    def encode(texts,role,batch=16):
        values=[]
        for begin in range(0,len(texts),batch):
            chunk=texts[begin:begin+batch]
            if a.engine=='ollama':endpoint='/api/embed';payload={'model':'bge-m3','input':chunk,'truncate':False,'keep_alive':'30m'}
            elif a.engine=='tei':endpoint='/embed';payload={'inputs':chunk,'truncate':False}
            else:endpoint='/v1/embeddings';payload={'model':'BAAI/bge-m3','input':chunk,'encoding_format':'float'}
            start=time.perf_counter();r=session.post(a.url.rstrip('/')+endpoint,json=payload,timeout=180);r.raise_for_status();body=r.json()
            vv=body['embeddings'] if a.engine=='ollama' else body if a.engine=='tei' else [x['embedding'] for x in sorted(body['data'],key=lambda x:x['index'])]
            vv=np.asarray(vv,dtype=np.float32);assert vv.shape==(len(chunk),1024) and np.isfinite(vv).all()
            assert np.all(np.linalg.norm(vv,axis=1)>0)
            values.append(vv/np.linalg.norm(vv,axis=1)[:,None])
            api_log.append({'role':role,'first_row':begin,'texts':len(chunk),'endpoint':endpoint,
                            'input_sha256':hashlib.sha256(json.dumps(chunk,ensure_ascii=False).encode()).hexdigest(),
                            'latency_seconds':time.perf_counter()-start,'status':r.status_code})
            if begin%256==0:print(json.dumps({'engine':a.engine,'api_role':role,'completed_texts':begin+len(chunk),'total':len(texts)}),flush=True)
        return np.concatenate(values)
    start=time.perf_counter()
    q=encode([c['model_query'] for c in cs],'query_all_3000')
    dq=encode([doc_text(d,'question') for d in ds],'corpus_question')
    dqa=encode([doc_text(d,'question_answer') for d in ds],'corpus_question_answer')
    elapsed=time.perf_counter()-start
    for name,v in [('query',q),('doc-question',dq),('doc-question_answer',dqa)]:np.save(cache/f'{name}.npy',v)
    sq=q@dq.T;sqa=q@dqa.T
    np.save(cache/'A-scores.npy',sq);np.save(cache/'B-scores.npy',sqa)
    execution={'dense_backend':a.engine+'_HTTP_API','question_rows_sent':3000,'query_rows_deduplicated':0,
               'provided_context_used':False,'http_inference_seconds':elapsed,'document_views':['question','question_answer'],
               'hf_dense_fallback':False,'raw_vector_dimensions':1024,'reranker':None}
    save_lines(base/'embedding_requests.jsonl',api_log)
    save(base/'embedding_execution.json',execution)
    summarize(a.engine,'A',cs,ds,rankings(sq),sq,execution)
    summarize(a.engine,'B',cs,ds,rankings(sqa),sqa,execution)
    pp=pools(sq,sqa,20);save(cache/'C-pools.json',pp)
    summarize(a.engine,'C-M20',cs,ds,pp,np.maximum(sq,sqa),{**execution,'candidate_budget':20,'merge_rule':'top20 each view, dedupe, max cosine, top20'})
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
    # Every row is scored, including each of the 200 actual repeat executions.
    pairs=[(i,j) for i,p in enumerate(pp) for j in p]
    scores=np.full((len(cs),len(ds)),-1e20,dtype=np.float32);batch=24;trimmed=0
    torch.cuda.synchronize();start=time.perf_counter()
    for begin in range(0,len(pairs),batch):
        indexes=pairs[begin:begin+batch];texts=[[cs[i]['model_query'],doc_text(ds[j],a.view)] for i,j in indexes]
        lengths=tok(texts,truncation=False)['input_ids'];trimmed+=sum(len(t)>512 for t in lengths)
        encoded=tok(texts,padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
        with torch.inference_mode():vals=model(**encoded).logits.reshape(-1).float().cpu().numpy()
        assert np.isfinite(vals).all()
        for (i,j),v in zip(indexes,vals):scores[i,j]=v
        if begin%2400==0:print(json.dumps({'engine':a.engine,'structure':name,'pairs_completed':begin+len(indexes),'pairs_total':len(pairs)}),flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    rr=[sorted(p,key=lambda j:(-float(scores[i,j]),j)) for i,p in enumerate(pp)]
    extra={'dense_backend':a.engine+'_HTTP_API','reranker_backend':'common_HF_PyTorch_CUDA_float32',
           'reranker_model':modelname,'reranker_view':a.view,'rows_scored':len(cs),'pairs_scored':len(pairs),
           'pairs_deduplicated':0,'repeat_rows_actually_rescored':200,'max_pair_tokens':512,'truncated_pairs':trimmed,
           'reranker_seconds':elapsed,'peak_torch_vram_bytes':int(torch.cuda.max_memory_allocated()),
           'all_components_served_by_engine':False,'score_adapter':'raw_logit_identity'}
    summarize(a.engine,name,cs,ds,rr,scores,extra)
    del model;gc.collect();torch.cuda.empty_cache()
def sparse_matrix(weights):
    from scipy.sparse import csr_matrix
    rows=[];cols=[];values=[]
    for i,w in enumerate(weights):
        for token,score in w.items():rows.append(i);cols.append(int(token));values.append(float(score))
    return csr_matrix((values,(rows,cols)),shape=(len(weights),250002),dtype=np.float32)
def native(a):
    torch=torch_setup()
    from FlagEmbedding import BGEM3FlagModel
    cs,ds=inputs();base=folder(a.engine)
    assert not (base/'E-sparse/summary.json').exists(),'Native stages already completed'
    model=BGEM3FlagModel(model_path('BAAI/bge-m3'),use_fp16=False,devices='cuda:0',batch_size=16,query_max_length=512,passage_max_length=512)
    start=time.perf_counter()
    docs=model.encode([doc_text(d,'question_answer') for d in ds],batch_size=16,max_length=512,return_dense=False,return_sparse=True,return_colbert_vecs=True)
    queries=model.encode([c['model_query'] for c in cs],batch_size=16,max_length=512,return_dense=False,return_sparse=True,return_colbert_vecs=True)
    torch.cuda.synchronize();encoding=time.perf_counter()-start
    extra={'dense_backend':a.engine+'_HTTP_API','native_backend':'common_FlagEmbedding_CUDA_float32',
           'all_components_served_by_engine':False,'rows_scored':3000,'repeat_rows_actually_rescored':200,
           'native_encode_seconds':encoding,'native_feature_cache_reused_from_other_engine':False}
    sparse=(sparse_matrix(queries['lexical_weights'])@sparse_matrix(docs['lexical_weights']).T).toarray()
    for qi in range(3):
        for di in range(3):assert np.isclose(sparse[qi,di],model.compute_lexical_matching_score(queries['lexical_weights'][qi],docs['lexical_weights'][di]),atol=1e-5)
    sr=rankings(sparse);dense=np.load(base/'cache/B-scores.npy');dr=rankings(dense)
    summarize(a.engine,'E-sparse',cs,ds,sr,sparse,{**extra,'dense_backend':None,'scope':'common native sparse control; no engine-specific dense contribution'})
    rrf=np.zeros_like(dense);rankscore=1/(60+np.arange(1,len(ds)+1))
    for i in range(len(cs)):
        rrf[i,dr[i]]+=rankscore;rrf[i,sr[i]]+=rankscore
    er=[sorted(set(map(int,dr[i,:20]))|set(map(int,sr[i,:20])),key=lambda j:(-float(rrf[i,j]),j))[:20] for i in range(len(cs))]
    summarize(a.engine,'E-dense-sparse-RRF-M20',cs,ds,er,rrf,{**extra,'rrf_constant':60,'candidate_budget':20})
    buckets=[];order=sorted(range(len(ds)),key=lambda i:len(docs['colbert_vecs'][i]))
    for begin in range(0,len(ds),32):
        ix=order[begin:begin+32];length=max(len(docs['colbert_vecs'][i]) for i in ix);dim=docs['colbert_vecs'][ix[0]].shape[1]
        vv=torch.zeros((len(ix),length,dim),device='cuda');mask=torch.zeros((len(ix),length),device='cuda',dtype=torch.bool)
        for j,i in enumerate(ix):v=docs['colbert_vecs'][i];vv[j,:len(v)]=torch.as_tensor(v,device='cuda');mask[j,:len(v)]=True
        buckets.append((ix,vv,mask))
    scores=np.empty_like(dense);torch.cuda.synchronize();start=time.perf_counter()
    with torch.inference_mode():
        for i,q in enumerate(queries['colbert_vecs']):
            qt=torch.as_tensor(q,device='cuda')
            for ix,dd,mask in buckets:
                vv=torch.einsum('qd,btd->bqt',qt,dd).masked_fill(~mask[:,None,:],-torch.inf)
                scores[i,ix]=vv.max(dim=-1).values.mean(dim=-1).cpu().numpy()
            if i%200==0:print(json.dumps({'engine':a.engine,'structure':'F','completed_queries':i+1,'total':3000}),flush=True)
    torch.cuda.synchronize();extra['late_interaction_seconds']=time.perf_counter()-start
    for qi in range(3):
        for di in range(3):assert np.isclose(scores[qi,di],model.colbert_score(queries['colbert_vecs'][qi],docs['colbert_vecs'][di]),atol=2e-5)
    summarize(a.engine,'F-full',cs,ds,rankings(scores),scores,{**extra,'dense_backend':None,'scope':'common native full-corpus diagnostic; excluded from structure selection'})
    fr=[sorted(map(int,dr[i,:20]),key=lambda j:(-float(scores[i,j]),j)) for i in range(len(cs))]
    summarize(a.engine,'F-dense-pool-M20',cs,ds,fr,scores,{**extra,'candidate_budget':20,'candidate_source':'this engine question-answer dense'})
def finish(a):
    cs,ds=inputs();base=folder(a.engine)
    names=['A','B','C-M20','D-bge-question-M20','D-bge-question_answer-M20','D-jina-question-M20','D-jina-question_answer-M20','E-sparse','E-dense-sparse-RRF-M20','F-full','F-dense-pool-M20']
    summaries={};policies={};repeat={}
    for name in names:
        summaries[name]=read(base/name/'summary.json');assert summaries[name]['case_count']==3000
        rows=lines(base/name/'per_query.jsonl');assert [r['case_id'] for r in rows]==[c['case_id'] for c in cs]
        policies[name]=calibrate_policy(rows,cs)
        groups={}
        for c,r in zip(cs,rows):
            if c['type']=='RT':groups.setdefault(c['case_id'].rsplit('-R',1)[0],[]).append(r)
        for rr in groups.values():assert len({r['query_hash'] for r in rr})==1,'Repeat group contains different model inputs'
        repeat[name]={k:{'executions':len(rr),'rank_hashes':len({r['rank_hash'] for r in rr}),
                         'query_hash_count':len({r['query_hash'] for r in rr}),
                         'top1_score_range':float(np.ptp([r['top_scores'][0] for r in rr]))} for k,rr in groups.items()}
    allowed=[n for n in names if n not in ['E-sparse','F-full']]
    key=lambda n:(-summaries[n]['cohort']['general']['family_mean']['all_sources@3'],
                  -summaries[n]['cohort']['condition']['family_mean']['all_sources@3'],
                  -summaries[n]['cohort']['general']['family_mean']['hit@1'],n)
    selected=sorted(allowed,key=key)[0]
    save(base/'policies.json',policies);save(base/'repeat_stability.json',repeat)
    save(base/'selection.json',{'selected':selected,'policy':policies[selected],
          'rule':'general family AllSourcesHit@3; ties by condition AllSourcesHit@3, general source Hit@1',
          'all_3000_rows_executed':True,'stages':len(names),'source_manifest_sha256':sha(ROOT/'data/manifest.json'),
          'calibration_only':True,'holdout_used':False,'prior_603_case_results_used':False,
          'all_components_served_by_engine':False})
    print(json.dumps({'engine':a.engine,'all_3000_rows_executed':True,'stages':len(names),'selected':selected},ensure_ascii=False),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validate','embeddings','D','native','finish'],required=True)
    p.add_argument('--engine',choices=['ollama','vllm','tei']);p.add_argument('--url');p.add_argument('--reranker',choices=['bge','jina']);p.add_argument('--view',choices=['question','question_answer']);a=p.parse_args()
    if a.stage=='validate':validate()
    else:
        assert a.engine
        {'embeddings':embeddings,'D':rerank,'native':native,'finish':finish}[a.stage](a)

if __name__=='__main__':main()
