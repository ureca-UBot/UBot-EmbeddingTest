"""Actual HTTP engine inference + full-FAQ, common CUDA reranker comparisons."""
import argparse
import concurrent.futures
import gc
import hashlib
import json
import time
from collections import Counter, defaultdict
import numpy as np
from common import DATA, OUT, ROOT, FIELDS, read, lines, save, save_lines, sha, inputs, model_path, doc_text, rankings, pools, summarize, export_log, metrics

def lengths(tokenizer, texts):
    result=[]
    for start in range(0,len(texts),128):
        result.extend(tokenizer(texts[start:start+128],truncation=False,return_length=True)['length'])
    return np.asarray(result,dtype=np.int32)
def matrix(path, shape, fill=0):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        values=np.load(path,mmap_mode='r+')
        assert values.shape==shape and values.dtype==np.float32
    else:
        values=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=shape)
        values[:]=fill;values.flush()
    return values
def payload(engine, texts, max_tokens=2048):
    if engine=='ollama':
        return '/api/embed', {'model':'bge-m3','input':texts,'truncate':False,'keep_alive':'30m',
            'options':{'num_ctx':8192,'num_batch':4096 if max_tokens>2048 else 2048}}
    return '/v1/embeddings',{'model':'BAAI/bge-m3','input':texts,'encoding_format':'float'}
def request(session, engine, url, texts, max_tokens=2048):
    endpoint,body=payload(engine,texts,max_tokens)
    start=time.perf_counter()
    response=session.post(url+endpoint,json=body,timeout=300)
    response.raise_for_status()
    result=response.json()
    values=result['embeddings'] if engine=='ollama' else [v['embedding'] for v in sorted(result['data'],key=lambda v:v['index'])]
    values=np.asarray(values,dtype=np.float32)
    assert values.shape==(len(texts),1024) and np.isfinite(values).all()
    norms=np.linalg.norm(values,axis=1)
    assert np.all(norms>0)
    return values/norms[:,None], time.perf_counter()-start, result

def validate():
    from transformers import AutoTokenizer
    cases,docs=inputs()
    parents=Counter(c['parent_case_id'] for c in cases if c['evaluation_role']=='main')
    assert len(parents)==500 and set(parents.values())=={10}
    ids={d['faq_id'] for d in docs}
    assert all(set(fid for f in c['required_facts'] for fid in f['acceptable_faq_ids'])<=ids for c in cases)
    tokenizer=AutoTokenizer.from_pretrained(model_path('BAAI/bge-m3'))
    ql=lengths(tokenizer,[c['model_query'] for c in cases])
    dl=lengths(tokenizer,[doc_text(d,'question_answer') for d in docs])
    assert max(ql.max(),dl.max())<=8192
    np.save(OUT/'query_token_lengths.npy',ql)
    rerank_lengths={}
    for name in ['BAAI/bge-reranker-v2-m3','jinaai/jina-reranker-v2-base-multilingual']:
        tok=AutoTokenizer.from_pretrained(model_path(name))
        query_lengths=lengths(tok,[c['model_query'] for c in cases])
        document_lengths=lengths(tok,[doc_text(d,'question_answer') for d in docs])
        # Conservative cartesian maximum; all actual candidate pairs will be checked too.
        bound=int(query_lengths.max()+document_lengths.max()+4)
        assert bound<=1024, f'Full FAQ pair may exceed reranker length: {name} {bound}'
        rerank_lengths[name]={'conservative_pair_bound':bound,'limit':1024}
    save(OUT/'input_validation.json',{'main_rows':5000,'purpose_control_rows':len(cases)-5000,
        'corpus_faqs':len(docs),'query_max_tokens':int(ql.max()),'qa_max_tokens':int(dl.max()),
        'rerank_lengths':rerank_lengths,'truncation':False,'old_evaluation_queries_imported':0,
        'cases_sha256':sha(DATA/'cases.jsonl'),'corpus_sha256':sha(DATA/'corpus.jsonl'),
        'independent_semantic_review_complete':False,'source_faq_answers_in_model_query':False})
    print(json.dumps(read(OUT/'input_validation.json')),flush=True)

def embeddings(args):
    import requests
    from transformers import AutoTokenizer
    cases,docs=inputs()
    base=OUT/'engines'/args.engine;cache=base/'cache';cache.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(model_path('BAAI/bge-m3'))
    session=requests.Session();requests_path=base/'embedding_requests.jsonl'
    def encode(texts,role):
        lens=lengths(tokenizer,texts)
        values=matrix(cache/(role+'.npy'),(len(texts),1024))
        progress=cache/(role+'-progress.json')
        state=read(progress) if progress.exists() else {'completed':0,'seconds':0.}
        begin=state['completed']
        while begin<len(texts):
            end=min(begin+16,len(texts))
            while end>begin+1 and (end-begin)*int(max(lens[begin:end]))>4096: end-=1
            text_batch=texts[begin:end]
            record={'role':role,'first_row':begin,'texts':len(text_batch),
                    'input_sha256':hashlib.sha256(json.dumps(text_batch,ensure_ascii=False).encode()).hexdigest(),
                    'max_tokens':int(max(lens[begin:end])),'truncation':False}
            try:
                vector,seconds,body=request(session,args.engine,args.url,text_batch,record['max_tokens'])
                values[begin:end]=vector;values.flush()
                record.update(success=True,latency_seconds=seconds,
                              processed_tokens=body.get('prompt_eval_count') if args.engine=='ollama' else body.get('usage',{}).get('prompt_tokens'))
            except Exception as exc:
                record.update(success=False,error=str(exc))
                with requests_path.open('a',encoding='utf-8') as stream: stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                raise
            with requests_path.open('a',encoding='utf-8') as stream: stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            state.update(completed=end,seconds=state['seconds']+seconds);save(progress,state)
            if end//512!=begin//512 or begin==0 or end==len(texts):
                print(json.dumps({'engine':args.engine,'role':role,'completed':end,'total':len(texts)}),flush=True)
            begin=end
        return np.asarray(values),state['seconds']
    q,qt=encode([c['model_query'] for c in cases],'queries')
    dq,dqt=encode([doc_text(d,'question') for d in docs],'corpus_question')
    dqa,dqat=encode([doc_text(d,'question_answer') for d in docs],'corpus_question_answer')
    a,b=q@dq.T,q@dqa.T
    np.save(cache/'A-scores.npy',a);np.save(cache/'B-scores.npy',b)
    records=lines(requests_path)
    execution={'dense_backend':args.engine+'_HTTP_API','query_rows_sent':len(cases),'main_query_rows_sent':5000,
               'query_rows_deduplicated':0,'query_seconds':qt,'query_texts_per_second':len(cases)/qt,
               'document_seconds':dqt+dqat,'truncated_inputs':0,'vector_dimensions':1024,
               'request_failures':sum(not r['success'] for r in records),'old_cached_outputs_reused':False,
               'latency_scope':'sequential variable-size batching; separate serving load test also run'}
    save(base/'embedding_execution.json',execution)
    summarize(args.engine,'A',cases,docs,rankings(a),a,execution)
    summarize(args.engine,'B',cases,docs,rankings(b),b,execution)
    candidate_sets={k:pools(a,b,k) for k in [1,3,5,10,20]}
    candidates=candidate_sets[20]
    save(cache/'C-pools.json',candidates)
    summarize(args.engine,'C-union-K20',cases,docs,candidates,np.maximum(a,b),{
        **execution,'per_view_k':20,'max_union_size':max(map(len,candidates)),
        'mean_union_size':float(np.mean(list(map(len,candidates)))),'merge':'full deduplicated union; max cosine only orders, never prunes'},candidate_sets)

def serving(args):
    import requests
    cases,docs=inputs()
    base=OUT/'engines'/args.engine;cache=base/'cache'
    q=np.load(cache/'queries.npy');a=np.load(cache/'A-scores.npy')
    docvec=np.load(cache/'corpus_question.npy')
    # Same 160 query indices at every concurrency; every request contains one text.
    ix=list(map(int,np.linspace(0,4999,160)))
    control=cases.index(next(c for c in cases if c['case_id']=='PUR-KREN-001-KO')) if any(c['case_id']=='PUR-KREN-001-KO' for c in cases) else 0
    runs=[]
    for concurrency in [1,4,8,16,32]:
        def perform(i):
            try:
                with requests.Session() as session:
                    vector,seconds,body=request(session,args.engine,args.url,[cases[i]['model_query']])
                top=list(map(int,np.argsort(-(vector[0]@docvec.T),kind='stable')[:20]))
                baseline=list(map(int,np.argsort(-a[i],kind='stable')[:20]))
                return {'case_id':cases[i]['case_id'],'success':True,'latency_seconds':seconds,
                        'embedding_cosine_to_sequential':float(vector[0]@q[i]),
                        'max_absolute_component_delta':float(np.max(np.abs(vector[0]-q[i]))),
                        'top20_overlap':len(set(top)&set(baseline))/20,'top1_same':top[0]==baseline[0]}
            except Exception as exc: return {'case_id':cases[i]['case_id'],'success':False,'error':str(exc)}
        start=time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            records=list(pool.map(perform,ix))
        seconds=time.perf_counter()-start
        success=[r for r in records if r['success']]
        lat=[r['latency_seconds'] for r in success]
        result={'concurrency':concurrency,'client_batch_size':1,'requests':len(ix),'successes':len(success),
                'failures':len(records)-len(success),'wall_seconds':seconds,
                'requests_per_second':len(success)/seconds,'texts_per_second':len(success)/seconds,
                'latency_seconds':{f'p{p}':float(np.percentile(lat,p)) if lat else None for p in [50,95,99]},
                'embedding_cosine_min':min((r['embedding_cosine_to_sequential'] for r in success),default=None),
                'top1_consistency':float(np.mean([r['top1_same'] for r in success])) if success else None,
                'mean_top20_overlap':float(np.mean([r['top20_overlap'] for r in success])) if success else None,
                'input_indices':ix,'measurement':'warm, closed-loop client-side embedding HTTP latency; not end-to-end retrieval SLA'}
        save_lines(base/f'serving_requests_c{concurrency}.jsonl',records)
        runs.append(result);save(base/'serving_load.json',{'runs':runs,'corpus_inference_included':False,'warmup':'full sequential embedding stage completed first'})
        print(json.dumps({'engine':args.engine,'serving':result}),flush=True)
    # Repeated identical input is one case executed ten times, never ten new questions.
    repeats=[]
    with requests.Session() as session:
        for iteration in range(10):
            vector,seconds,body=request(session,args.engine,args.url,[cases[control]['model_query']])
            scores=vector[0]@docvec.T
            rr=list(map(int,np.argsort(-scores,kind='stable')[:20]))
            repeats.append({'case_id':cases[control]['case_id'],'run':iteration+1,'latency_seconds':seconds,
                            'embedding_cosine':float(vector[0]@q[control]),'top_ids':[docs[j]['faq_id'] for j in rr],
                            'context_hash':hashlib.sha256('\n'.join(doc_text(docs[j],'question_answer') for j in rr).encode()).hexdigest()})
    save_lines(base/'repeat_runs.jsonl',repeats)
    save(base/'repeat_summary.json',{'independent_case_count':1,'runs':10,
        'distinct_context_hashes':len({r['context_hash'] for r in repeats}),
        'distinct_top20_ranks':len({tuple(r['top_ids']) for r in repeats})})
    # Actual paired language controls: frozen raw Korean input, no GT injected.
    pair_cases=lines(DATA/'language_pair_controls.jsonl');positions={c['case_id']:i for i,c in enumerate(cases)}
    pair_rows=[]
    with requests.Session() as session:
        for start in range(0,len(pair_cases),16):
            chunk=pair_cases[start:start+16]
            vector,seconds,body=request(session,args.engine,args.url,[c['query_ko'] for c in chunk])
            for v,pair in zip(vector,chunk):
                i=positions[pair['case_id']]
                ko_scores=v@docvec.T;ko=list(map(int,np.argsort(-ko_scores,kind='stable')))
                mixed=list(map(int,np.argsort(-a[i],kind='stable')))
                ko_ids=[docs[j]['faq_id'] for j in ko];mix_ids=[docs[j]['faq_id'] for j in mixed]
                gold={fid for f in pair['required_facts'] for fid in f['acceptable_faq_ids']}
                kr=min(p+1 for p,fid in enumerate(ko_ids) if fid in gold)
                mr=min(p+1 for p,fid in enumerate(mix_ids) if fid in gold)
                pair_rows.append({'case_id':pair['case_id'],'query_ko':pair['query_ko'],'variant_query':pair['variant_query'],
                    'korean_first_gold_rank':kr,'variant_first_gold_rank':mr,'rank_delta':kr-mr,
                    'top20_overlap':len(set(ko[:20])&set(mixed[:20]))/20,
                    'ko_metrics':metrics(ko_ids[:3],pair['required_facts']),
                    'variant_metrics':metrics(mix_ids[:3],pair['required_facts'])})
    save_lines(base/'language_pair_results.jsonl',pair_rows)

def rerank(args):
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    torch.manual_seed(20261007)
    torch.backends.cuda.matmul.allow_tf32=False
    cases,docs=inputs();base=OUT/'engines'/args.engine
    candidates=read(base/'cache/C-pools.json')
    name=f'D-{args.reranker}-{args.view}'
    model_name='BAAI/bge-reranker-v2-m3' if args.reranker=='bge' else 'jinaai/jina-reranker-v2-base-multilingual'
    model=AutoModelForSequenceClassification.from_pretrained(model_path(model_name),trust_remote_code=args.reranker=='jina',
        torch_dtype=torch.float16,**({'use_flash_attn':False} if args.reranker=='jina' else {})).to('cuda').eval()
    tokenizer=AutoTokenizer.from_pretrained(model_path(model_name))
    queries=[c['model_query'] for c in cases];documents=[doc_text(d,args.view) for d in docs]
    ql,dl=lengths(tokenizer,queries),lengths(tokenizer,documents)
    pairs=np.asarray([(i,j) for i,ids in enumerate(candidates) for j in ids],dtype=np.int32)
    qi,di=pairs.T;estimated=ql[qi]+dl[di]+4;order=np.argsort(estimated,kind='stable')
    scores=matrix(base/'cache'/f'{name}-scores.npy',(len(cases),len(docs)),fill=-1e20)
    statepath=base/'cache'/f'{name}-progress.json'
    state=read(statepath) if statepath.exists() else {'completed':0,'seconds':0.,'batches':0,'max_pair_tokens':0}
    identity=hashlib.sha256(pairs.tobytes()+order.tobytes()).hexdigest()
    assert state.get('pair_hash',identity)==identity
    begin=state['completed'];checkpoint=time.perf_counter();report=checkpoint
    while begin<len(order):
        end=min(begin+96,len(order))
        while end>begin+1 and (end-begin)*int(max(estimated[order[begin:end]]))>8192: end-=1
        ix=order[begin:end]
        encoded=tokenizer([[queries[i],documents[j]] for i,j in zip(qi[ix],di[ix])],padding=True,truncation=False,return_tensors='pt')
        width=int(encoded['input_ids'].shape[1]);assert width<=1024
        with torch.inference_mode(): values=model(**encoded.to('cuda')).logits.reshape(-1).float().cpu().numpy()
        assert values.shape==(len(ix),) and np.isfinite(values).all()
        scores[qi[ix],di[ix]]=values
        state['batches']+=1;state['max_pair_tokens']=max(state['max_pair_tokens'],width)
        if end//5000!=begin//5000 or end==len(order):
            torch.cuda.synchronize();scores.flush();now=time.perf_counter()
            state.update(completed=end,seconds=state['seconds']+now-checkpoint,total_pairs=len(order),pair_hash=identity)
            save(statepath,state);checkpoint=now
            if now-report>25 or end==len(order):
                print(json.dumps({'engine':args.engine,'stage':name,'pairs_completed':end,'total':len(order),
                                  'seconds':round(state['seconds'],1),'max_pair_tokens':state['max_pair_tokens']}),flush=True);report=now
        begin=end
    assert np.all(scores[qi,di]>-1e19)
    ranked=[sorted(ids,key=lambda j:(-float(scores[i,j]),j)) for i,ids in enumerate(candidates)]
    execution={'dense_backend':args.engine+'_HTTP_API','reranker_backend':'common_HF_PyTorch_CUDA_float16',
        'reranker_model':model_name,'reranker_view':args.view,'raw_logit_sort':True,
        'candidate_faq_pairs':len(order),'scored_pairs':state['completed'],'reranker_seconds':state['seconds'],
        'max_pair_tokens':state['max_pair_tokens'],'truncated_pairs':0,'windowing':False,'window_count':1,
        'peak_torch_vram_bytes':int(torch.cuda.max_memory_allocated()),'pairs_deduplicated':0,
        'all_components_served_by_engine':False}
    summarize(args.engine,name,cases,docs,ranked,scores,execution)
    export_log(args.engine,name,args.view,cases,docs,candidates,scores,execution)
    del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['validate','embeddings','serving','D'])
    parser.add_argument('--engine',choices=['ollama','vllm'])
    parser.add_argument('--url');parser.add_argument('--reranker',choices=['bge','jina'])
    parser.add_argument('--view',choices=['question','question_answer'])
    args=parser.parse_args()
    if args.stage=='validate': validate()
    else: {'embeddings':embeddings,'serving':serving,'D':rerank}[args.stage](args)
