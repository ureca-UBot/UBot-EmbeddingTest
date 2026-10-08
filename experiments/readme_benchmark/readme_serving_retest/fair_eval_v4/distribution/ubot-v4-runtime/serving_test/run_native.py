"""Full-corpus native sparse/ColBERT, shared auxiliary to engine dense runs."""
import gc,pickle,time
import numpy as np
from scipy.sparse import csr_matrix
from common_v4 import *

def sparse_matrix(weights,vocab):
 rows=[];cols=[];values=[]
 for row,w in enumerate(weights):
  for t,v in w.items():rows.append(row);cols.append(int(t));values.append(float(v))
 return csr_matrix((values,(rows,cols)),shape=(len(weights),vocab),dtype=np.float32)

def maxsim_batch(qv,dv,device='cuda'):
 import torch
 ql=max(map(len,qv));dl=max(map(len,dv));dim=qv[0].shape[1]
 q=torch.zeros((len(qv),ql,dim),dtype=torch.float32,device=device);d=torch.zeros((len(dv),dl,dim),dtype=torch.float32,device=device)
 qm=torch.zeros((len(qv),ql),dtype=torch.bool,device=device);dm=torch.zeros((len(dv),dl),dtype=torch.bool,device=device)
 for i,v in enumerate(qv):q[i,:len(v)]=torch.as_tensor(v,device=device);qm[i,:len(v)]=True
 for i,v in enumerate(dv):d[i,:len(v)]=torch.as_tensor(v,device=device);dm[i,:len(v)]=True
 with torch.inference_mode():
  s=(q.flatten(0,1)@d.flatten(0,1).T).reshape(len(qv),ql,len(dv),dl)
  s.masked_fill_(~dm[None,None,:,:],-torch.inf);best=s.max(-1).values
  result=(best*qm[:,:,None]).sum(1)/qm.sum(1)[:,None]
 return result.cpu().numpy()

def encode(model,texts,root,role):
 file=root/(role+'.pkl');meta=root/(role+'.json');identity=token([texts,MODEL_REVISIONS['BAAI/bge-m3'],'fp16-native-v1'])
 if meta.exists():
  state=read(meta);assert state['identity']==identity and state['sha256']==sha(file)
  return pickle.loads(file.read_bytes()),state
 lens=lengths(model.tokenizer,texts);assert max(lens)<=8192
 order=np.argsort(lens,kind='stable');dense=np.zeros((len(texts),1024),np.float32);sparse=[None]*len(texts);multi=[None]*len(texts)
 start=time.perf_counter();last=start;begin=0
 while begin<len(order):
  end=min(begin+16,len(order))
  while end>begin+1 and int(lens[order[end-1]])*(end-begin)>4096:end-=1
  ii=list(map(int,order[begin:end]));vectors=model.encode([texts[i] for i in ii],batch_size=len(ii),max_length=8192,
     return_dense=True,return_sparse=True,return_colbert_vecs=True)
  for z,i in enumerate(ii):dense[i]=vectors['dense_vecs'][z];sparse[i]=vectors['lexical_weights'][z];multi[i]=np.asarray(vectors['colbert_vecs'][z],dtype=np.float32)
  if time.perf_counter()-last>25 or end==len(order):print(json.dumps({'native_role':role,'done':end,'total':len(order)}),flush=True);last=time.perf_counter()
  begin=end
 result={'dense_vecs':dense,'lexical_weights':sparse,'colbert_vecs':multi};elapsed=time.perf_counter()-start
 assert np.isfinite(dense).all() and all(v is not None and len(v)>0 and np.isfinite(v).all() for v in multi)
 file.write_bytes(pickle.dumps(result,protocol=pickle.HIGHEST_PROTOCOL));state={'identity':identity,'sha256':sha(file),'rows':len(texts),'max_tokens':int(lens.max()),'seconds':elapsed,'truncated':False,'dtype':'float16 inference, float32 scoring'}
 save(meta,state);return result,state

def score_multi(q,d,root,view,model):
 file=root/(view+'-multi.npy');progress=root/(view+'-multi-progress.json');shape=(len(q),len(d))
 qo=sorted(range(len(q)),key=lambda i:len(q[i]));do=sorted(range(len(d)),key=lambda i:len(d[i]))
 identity=token([qo,do,[len(x) for x in q],[len(x) for x in d],'mean_query_token_max_document_token-float32-v1'])
 if progress.exists():
  state=read(progress);assert state['identity']==identity;arr=np.load(file,mmap_mode='r+')
 else:
  arr=np.lib.format.open_memmap(file,mode='w+',shape=shape,dtype=np.float32);arr[:]=np.nan;arr.flush();state={'identity':identity,'completed':0,'seconds':0.}
 last=checkpoint=time.perf_counter()
 for begin in range(state['completed'],len(q),4):
  qi=qo[begin:begin+4]
  for start in range(0,len(d),32):
   di=do[start:start+32];arr[np.ix_(qi,di)]=maxsim_batch([q[i] for i in qi],[d[i] for i in di])
  end=min(begin+4,len(q))
  if end%32==0 or end==len(q):
   arr.flush();state.update(completed=end,seconds=state['seconds']+time.perf_counter()-checkpoint);save(progress,state);checkpoint=time.perf_counter()
   if time.perf_counter()-last>25 or end==len(q):print(json.dumps({'native_multivector':view,'done':end,'total':len(q),'seconds':round(state['seconds'],1)}),flush=True);last=time.perf_counter()
 assert np.isfinite(arr).all()
 probes=[(0,0),(len(q)-1,len(d)-1),(max(range(len(q)),key=lambda i:len(q[i])),max(range(len(d)),key=lambda i:len(d[i])))]
 rng=np.random.default_rng(20261007);probes += [(int(rng.integers(len(q))),int(rng.integers(len(d)))) for _ in range(5)]
 checked=[]
 for i,j in probes:
  expected=float(model.colbert_score(q[i],d[j]));actual=float(arr[i,j]);assert np.isclose(expected,actual,atol=3e-5,rtol=3e-5),(expected,actual)
  checked.append({'query':i,'document':j,'official':expected,'batched':actual})
 save(root/(view+'-multi-checks.json'),checked);return np.asarray(arr),state

def execute(model,key):
 import torch
 d,qs,docs=load(key);root=OUT/key/'native';root.mkdir(parents=True,exist_ok=True)
 q,qstate=encode(model,[x['model_query'] for x in qs],root,'queries');representations={};timings={}
 sq=sparse_matrix(q['lexical_weights'],model.tokenizer.vocab_size)
 for view in ('question','question_answer'):
  vectors,ds=encode(model,[doc_text(x,view) for x in docs],root,'corpus_'+view)
  sparse=(sq@sparse_matrix(vectors['lexical_weights'],model.tokenizer.vocab_size).T).toarray();np.save(root/(view+'-sparse.npy'),sparse)
  for i,j in [(0,0),(len(qs)-1,len(docs)-1)]:
   official=float(model.compute_lexical_matching_score(q['lexical_weights'][i],vectors['lexical_weights'][j]))
   assert np.isclose(official,sparse[i,j],atol=2e-5,rtol=2e-5)
  multi,ms=score_multi(q['colbert_vecs'],vectors['colbert_vecs'],root,view,model)
  representations['E-sparse-'+view]=sparse;representations['F-multi-'+view]=multi
  timings[view]={'document_encoding':ds,'multivector_scoring':ms};del vectors;gc.collect();torch.cuda.empty_cache()
 execution={'backend':'shared_FlagEmbedding_CUDA_native_heads','model':'BAAI/bge-m3','revision':MODEL_REVISIONS['BAAI/bge-m3'],
  'engine_serves_native_heads':False,'native_query_rows':len(qs),'native_corpus_rows':len(docs),'corpus_scope':key,
  'query_encoding':qstate,'views':timings,'scoring':'full corpus, no dense candidate restriction',
  'maxsim':'mean over query tokens of max over document tokens','dtype':'float16 encode, float32 score','tf32':False,
  'head_hashes':{x:sha(Path(model_path('BAAI/bge-m3'))/x) for x in ('sparse_linear.pt','colbert_linear.pt')}}
 save(root/'execution.json',execution)
 for engine in ENGINES:
  a=np.load(base(key,engine)/'cache/A-scores.npy');b=np.load(base(key,engine)/'cache/B-scores.npy')
  all_scores=dict(representations)
  all_scores['E-dense-sparse-rrf']=1./(60+inverse(b))+1./(60+inverse(representations['E-sparse-question_answer']))
  for name,scores in all_scores.items():
   ranked=ranks(scores)
   summarize(key,engine,name,qs,docs,ranked,scores,dict(execution,fusion='RRF k=60 fixed; engine B dense + shared sparse QA' if name.endswith('rrf') else None))
   export_log(key,engine,name,qs,docs,ranked[:,:20],a,b,representation_scores=scores)
 print(json.dumps({'native_dataset_complete':key}),flush=True)

if __name__=='__main__':
 import argparse,torch
 from FlagEmbedding import BGEM3FlagModel
 p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',choices=KEYS,default=list(KEYS));args=p.parse_args()
 torch.manual_seed(20261007);torch.backends.cuda.matmul.allow_tf32=False
 model=BGEM3FlagModel(model_path('BAAI/bge-m3'),use_fp16=True,devices='cuda:0',batch_size=16,
   query_max_length=8192,passage_max_length=8192,return_dense=True,return_sparse=True,return_colbert_vecs=True)
 for key in args.datasets:execute(model,key)
