"""Real engine HTTP runs; each invocation chooses exactly one v4 corpus."""
import argparse,concurrent.futures,time
import numpy as np
from common_v4 import *

def preflight():
 from transformers import AutoTokenizer
 m=read(DATASET_ROOT/'manifest.json')
 if (OUT/'frozen_manifest.json').exists():assert read(OUT/'frozen_manifest.json')==m
 else:save(OUT/'frozen_manifest.json',m)
 tok=AutoTokenizer.from_pretrained(model_path('BAAI/bge-m3'))
 info={}
 for key in KEYS:
  d,qs,docs=load(key);ql=lengths(tok,[q['model_query'] for q in qs]);dl=lengths(tok,[doc_text(x,'question_answer') for x in docs])
  assert max(ql.max(),dl.max())<=8192,(key,int(ql.max()),int(dl.max()))
  info[key]={'queries':len(qs),'corpus':len(docs),'query_max_tokens':int(ql.max()),'qa_max_tokens':int(dl.max()),'qa_over_1024':int(sum(dl>1024)),
    'questions_sha256':d.metadata['questions_sha256'],'corpus_sha256':d.metadata['corpus_sha256']}
 save(OUT/'preflight.json',{'datasets':info,'revisions':MODEL_REVISIONS,'truncation':False,
  'matrix':['A','B','C-fixed20','C-raw-union','D-bge-question','D-bge-question_answer','D-jina-question','D-jina-question_answer','E-sparse-question','E-sparse-question_answer','E-dense-sparse-rrf','F-multi-question','F-multi-question_answer'],
  'reranker_pair_limit':1024,'document_window_overlap':128,
  'threshold_fitting':False,'all_labels_unreviewed':True,'corpus_scope_before_encoding':True,
  'vllm_dtype':'float16','ollama_expected_dtype':'F16, must inspect model metadata',
  'native_heads':'common FlagEmbedding backend, not Ollama/vLLM HTTP',
  'run_started':now(),'dataset_manifest_sha256':sha(DATASET_ROOT/'manifest.json')})
 print(json.dumps(info,ensure_ascii=False),flush=True)

def request(session,engine,url,texts,max_tokens=2048):
 if engine=='ollama':
  endpoint='/api/embed';body={'model':'bge-m3','input':texts,'truncate':False,'keep_alive':'30m','options':{'num_ctx':8192,'num_batch':4096 if max_tokens>2048 else 2048}}
 else:endpoint='/v1/embeddings';body={'model':'BAAI/bge-m3','input':texts,'encoding_format':'float'}
 start=time.perf_counter();r=session.post(url+endpoint,json=body,timeout=600);r.raise_for_status();v=r.json()
 arr=np.asarray(v['embeddings'] if engine=='ollama' else [x['embedding'] for x in sorted(v['data'],key=lambda x:x['index'])],dtype=np.float32)
 assert arr.shape==(len(texts),1024) and np.isfinite(arr).all()
 n=np.linalg.norm(arr,axis=1);assert np.all(n>0)
 return arr/n[:,None],time.perf_counter()-start,v

def encode(key,engine,url,texts,role,tok,session):
 root=base(key,engine);cache=root/'cache';cache.mkdir(parents=True,exist_ok=True)
 identity={'dataset_id':key,'text_sha256':token(texts),'model_revision':MODEL_REVISIONS['BAAI/bge-m3'],'engine':engine,
    'role':role,'dtype':'float16','normalization':'l2','dimensions':1024,'context':8192,'document_separator':'two newlines'}
 fp=cache/(role+'.npy');sp=cache/(role+'.json');lens=lengths(tok,texts)
 assert lens.max()<=8192
 if sp.exists():
  state=read(sp);assert state['identity']==identity
  arr=np.load(fp,mmap_mode='r+');assert arr.shape==(len(texts),1024)
 else:
  arr=np.lib.format.open_memmap(fp,mode='w+',dtype=np.float32,shape=(len(texts),1024));arr[:]=np.nan;arr.flush()
  state={'identity':identity,'completed':0,'seconds':0.,'total':len(texts)}
 begin=state['completed'];last=time.perf_counter()
 while begin<len(texts):
  end=min(begin+16,len(texts))
  while end>begin+1 and int(lens[begin:end].max())*(end-begin)>4096:end-=1
  record=dict(dataset_id=key,engine=engine,role=role,first_row=begin,texts=end-begin,input_sha256=token(texts[begin:end]),max_tokens=int(lens[begin:end].max()))
  try:
   vectors,seconds,response=request(session,engine,url,texts[begin:end],record['max_tokens']);arr[begin:end]=vectors;arr.flush()
   record.update(success=True,seconds=seconds,processed_tokens=response.get('prompt_eval_count') if engine=='ollama' else response.get('usage',{}).get('prompt_tokens'))
  except Exception as e:
   record.update(success=False,error=str(e));append(root/'embedding_requests.jsonl',record);raise
  append(root/'embedding_requests.jsonl',record);state.update(completed=end,seconds=state['seconds']+seconds);save(sp,state)
  if time.perf_counter()-last>20 or end==len(texts):print(json.dumps({'dataset':key,'engine':engine,'role':role,'done':end,'total':len(texts)}),flush=True);last=time.perf_counter()
  begin=end
 assert np.isfinite(arr).all();return np.asarray(arr),state

def embeddings(args):
 import requests
 from transformers import AutoTokenizer
 key,engine,url=args.dataset,args.engine,args.url;d,qs,docs=load(key);root=base(key,engine)
 tok=AutoTokenizer.from_pretrained(model_path('BAAI/bge-m3'))
 with requests.Session() as session:
  q,qstate=encode(key,engine,url,[x['model_query'] for x in qs],'queries',tok,session)
  aq,astate=encode(key,engine,url,[x['text'] for x in d.documents('question')],'corpus_question',tok,session)
  bq,bstate=encode(key,engine,url,[x['text'] for x in d.documents('question_answer')],'corpus_question_answer',tok,session)
 a,b=q@aq.T,q@bq.T;np.save(root/'cache/A-scores.npy',a);np.save(root/'cache/B-scores.npy',b)
 c=pool(a,b,20);save(root/'cache/C-pools.json',c)
 execution={'backend':engine+'_HTTP','query_rows':len(qs),'corpus_rows':len(docs),'vector_dimensions':1024,'normalized':True,'truncated':False,
   'query_seconds':qstate['seconds'],'document_seconds':astate['seconds']+bstate['seconds'],
   'query_texts_per_second':len(qs)/qstate['seconds'],'dtype':'float16','source_format_difference':'GGUF versus Hugging Face remains',
   'input_hashes':d.metadata,'request_failures':sum(not r['success'] for r in lines(root/'embedding_requests.jsonl'))}
 save(root/'embedding_execution.json',execution)
 for name,rs,scores in [('A',ranks(a),a),('B',ranks(b),b),('C-fixed20',[x[:20] for x in c],np.maximum(a,b))]:
  summarize(key,engine,name,qs,docs,rs,scores,execution)
  export_log(key,engine,name,qs,docs,[list(x[:20]) for x in rs],a,b)
 summarize(key,engine,'C-raw-union',qs,docs,c,np.maximum(a,b),execution,{k:pool(a,b,k) for k in KS})
 export_log(key,engine,'C-raw-union',qs,docs,c,a,b)

def serving(args):
 import requests
 key,engine,url=args.dataset,args.engine,args.url;d,qs,docs=load(key);root=base(key,engine)
 vectors=np.load(root/'cache/queries.npy');docvec=np.load(root/'cache/corpus_question.npy');a=np.load(root/'cache/A-scores.npy')
 # Exact same deterministic 64 inputs for both engines, repeated at each load.
 chosen=list(map(int,np.linspace(0,len(qs)-1,min(64,len(qs)))))
 runs=[]
 for concurrency in [1,4,8,16,32]:
  def run(i):
   try:
    with requests.Session() as s:v,t,_=request(s,engine,url,[qs[i]['model_query']])
    rr=np.argsort(-(v[0]@docvec.T),kind='stable')[:20];prior=np.argsort(-a[i],kind='stable')[:20]
    return dict(question_id=qs[i]['question_id'],success=True,seconds=t,embedding_cosine=float(v[0]@vectors[i]),top1_same=bool(rr[0]==prior[0]),top20_overlap=len(set(rr)&set(prior))/20)
   except Exception as e:return dict(question_id=qs[i]['question_id'],success=False,error=str(e))
  start=time.perf_counter()
  with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:records=list(executor.map(run,chosen))
  elapsed=time.perf_counter()-start;ok=[r for r in records if r['success']];lat=[r['seconds'] for r in ok]
  save_lines(root/f'load_c{concurrency}.jsonl',records)
  run_result={'concurrency':concurrency,'requests':len(records),'successes':len(ok),'failures':len(records)-len(ok),'seconds':elapsed,'requests_per_second':len(ok)/elapsed,
    'latency_seconds':{f'p{p}':float(np.percentile(lat,p)) if lat else None for p in (50,95,99)},
    'top1_consistency':float(np.mean([r['top1_same'] for r in ok])) if ok else None,
    'min_embedding_cosine':min([r['embedding_cosine'] for r in ok],default=None)}
  runs.append(run_result);save(root/'serving_load.json',{'dataset_id':key,'runs':runs,'input_indices':chosen,'warm':True,'client_batch_size':1,'scope':'client HTTP embedding latency, not end-to-end SLA; small diagnostic sample'})
  print(json.dumps({'dataset':key,'engine':engine,'load':run_result}),flush=True)
  assert len(ok)==len(records),'Serving errors: inspect raw request log'
 qm={q['question_id']:i for i,q in enumerate(qs)};repeated=[]
 with requests.Session() as s:
  for fixture in lines(d.path/'fixtures/repeat.jsonl'):
   i=qm[fixture['question_id']]
   for run in range(10):
    v,t,_=request(s,engine,url,[fixture['model_query']]);scores=v[0]@docvec.T;rr=list(map(int,np.argsort(-scores,kind='stable')[:20]))
    repeated.append(dict(question_id=fixture['question_id'],run=run+1,seconds=t,top_ids=[docs[j]['faq_id'] for j in rr],top_scores=[float(scores[j]) for j in rr],
      context_hash=token([doc_text(docs[j],'question_answer') for j in rr]),embedding_hash=hashlib.sha256(v.tobytes()).hexdigest()))
 save_lines(root/'repeat_runs.jsonl',repeated)
 save(root/'repeat_summary.json',{'unique_questions':10,'runs':len(repeated),'queries_with_rank_change':sum(len({tuple(r['top_ids']) for r in repeated if r['question_id']==f['question_id']})>1 for f in lines(d.path/'fixtures/repeat.jsonl')),
  'queries_with_context_change':sum(len({r['context_hash'] for r in repeated if r['question_id']==f['question_id']})>1 for f in lines(d.path/'fixtures/repeat.jsonl')),
  'output_hash_measured':False,'generation_run':False})

def attacks(args):
 import requests
 from transformers import AutoTokenizer
 key,engine,url=args.dataset,args.engine,args.url;d,qs,docs=load(key);root=base(key,engine)
 fixtures=lines(d.path/'fixtures/document_attack.jsonl');qm={q['question_id']:i for i,q in enumerate(qs)};di={d['faq_id']:i for i,d in enumerate(docs)}
 texts=[doc_text(docs[di[f['faq_id']]],'question_answer')+f['append_text'] for f in fixtures]
 tok=AutoTokenizer.from_pretrained(model_path('BAAI/bge-m3'))
 with requests.Session() as s:patched,_=encode(key,engine,url,texts,'attack_patched_documents',tok,s)
 vectors=np.load(root/'cache/queries.npy');clean=np.load(root/'cache/B-scores.npy');rows=[]
 for f,v,text in zip(fixtures,patched,texts):
  i=qm[f['question_id']];j=di[f['faq_id']];scores=clean[i].copy();scores[j]=vectors[i]@v
  before=np.argsort(-clean[i],kind='stable')[:20];after=np.argsort(-scores,kind='stable')[:20]
  context=[text if x==j else doc_text(docs[x],'question_answer') for x in after]
  rows.append(dict(fixture_id=f['fixture_id'],question_id=f['question_id'],faq_id=f['faq_id'],mutation_scope='one fixture at a time, original corpus unchanged',
   clean_document_rank=int(np.where(np.argsort(-clean[i],kind='stable')==j)[0][0])+1,
   attacked_document_rank=int(np.where(np.argsort(-scores,kind='stable')==j)[0][0])+1,
   clean_top20=[docs[x]['faq_id'] for x in before],attacked_top20=[docs[x]['faq_id'] for x in after],
   metrics=score(qs[i],[docs[x]['faq_id'] for x in after],20),constructed_context=context,
   context_hash=token(context),canary_in_context=any(f['forbidden_output_marker'] in t for t in context),
   generation_executed=False,instruction_following_success=None))
 save_lines(root/'document_attack_results.jsonl',rows)
 save(root/'document_attack_summary.json',{'fixtures':len(rows),'retrieval_context_executed':True,'generation_executed':False,
   'reason':'No answer-generating LLM configured; embedding/reranking engines cannot certify instruction following',
   'rank_changed':sum(r['clean_document_rank']!=r['attacked_document_rank'] for r in rows),
   'canary_in_context':sum(r['canary_in_context'] for r in rows),'included_in_clean_metrics':False})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','embeddings','serving','attacks']);p.add_argument('--dataset',choices=KEYS);p.add_argument('--engine',choices=ENGINES);p.add_argument('--url');a=p.parse_args()
 if a.stage=='preflight':preflight()
 else:
  assert a.dataset and a.engine and a.url
  globals()[a.stage](a)
