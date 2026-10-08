"""One common reranker evaluates both engines' independently retrieved pools.

Identical query/document pairs are scored once per dataset/model/view. Their
scores are reused, never another carrier's corpus or an engine's dense vectors.
"""
import argparse,gc,time
import numpy as np
from common_v4 import *

def spans(length,budget,overlap=128):
 if budget<=overlap:raise ValueError('Query leaves insufficient document budget')
 if length<=budget:return [(0,length)]
 result=[];start=0
 while start<length:
  end=min(length,start+budget);result.append((start,end))
  if end==length:break
  start=end-overlap
 return result

def pair_ids(tokenizer,query_ids,document_ids):
 # Both pinned rerankers use XLM-R's four-token pair template. Verify it
 # against the installed tokenizer before scoring; v5 removed prepare_for_model.
 return [tokenizer.cls_token_id]+list(query_ids)+[tokenizer.sep_token_id,tokenizer.sep_token_id]+list(document_ids)+[tokenizer.sep_token_id]

def padded_batch(tokenizer,pairs):
 import torch
 width=max(map(len,pairs));ids=torch.full((len(pairs),width),tokenizer.pad_token_id,dtype=torch.long)
 mask=torch.zeros_like(ids)
 for i,pair in enumerate(pairs):ids[i,:len(pair)]=torch.tensor(pair);mask[i,:len(pair)]=1
 return {'input_ids':ids,'attention_mask':mask}

def execute(model,tokenizer,key,kind,view):
 import torch
 d,qs,docs=load(key);name=f'D-{kind}-{view}';root=OUT/key/'shared_reranker'/name;root.mkdir(parents=True,exist_ok=True)
 model_name='BAAI/bge-reranker-v2-m3' if kind=='bge' else 'jinaai/jina-reranker-v2-base-multilingual'
 pools={e:read(base(key,e)/'cache/C-pools.json') for e in ENGINES}
 combined=[sorted(set(x)|set(y)) for x,y in zip(pools['ollama'],pools['vllm'])]
 queries=[q['model_query'] for q in qs];texts=[doc_text(x,view) for x in docs]
 qt=tokenizer(queries,add_special_tokens=False,truncation=False)['input_ids'];dt=tokenizer(texts,add_special_tokens=False,truncation=False)['input_ids']
 assert tokenizer(queries[0],texts[0],truncation=False,return_token_type_ids=False)['input_ids']==pair_ids(tokenizer,qt[0],dt[0])
 specials=tokenizer.num_special_tokens_to_add(pair=True);slots=[];bounds={}
 for i,ids in enumerate(combined):
  for j in ids:
   windows=spans(len(dt[j]),1024-len(qt[i])-specials);bounds[i,j]=windows
   for wi,(start,end) in enumerate(windows):slots.append((i,j,wi,start,end,len(qt[i])+end-start+specials))
 slots.sort(key=lambda x:(x[5],x[0],x[1],x[2]));identity=token([slots,d.metadata,model_name,MODEL_REVISIONS[model_name],view,'fp16-window-max-v1'])
 statepath=root/'progress.json';shape=(len(qs),len(docs));scorepath=root/'scores.npy';winnerpath=root/'winning_windows.npy'
 if statepath.exists():
  state=read(statepath);assert state['identity']==identity
  scores=np.load(scorepath,mmap_mode='r+');winners=np.load(winnerpath,mmap_mode='r+')
 else:
  scores=np.lib.format.open_memmap(scorepath,mode='w+',shape=shape,dtype=np.float32);scores[:]=-1e20
  winners=np.lib.format.open_memmap(winnerpath,mode='w+',shape=shape,dtype=np.int16);winners[:]=-1
  scores.flush();winners.flush();state={'identity':identity,'completed':0,'seconds':0.,'batches':0,'max_pair_tokens':0}
 begin=state['completed'];last=checkpoint=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 while begin<len(slots):
  end=min(begin+96,len(slots))
  while end>begin+1 and slots[end-1][5]*(end-begin)>8192:end-=1
  subset=slots[begin:end]
  encoded=[pair_ids(tokenizer,qt[i],dt[j][start:stop]) for i,j,wi,start,stop,l in subset]
  batch=padded_batch(tokenizer,encoded);width=int(batch['input_ids'].shape[1]);assert width<=1024
  with torch.inference_mode():values=model(**{k:v.to('cuda') for k,v in batch.items()}).logits.reshape(-1).float().cpu().numpy()
  assert len(values)==len(subset) and np.isfinite(values).all()
  for (i,j,wi,start,stop,l),value in zip(subset,values):
   if value>scores[i,j] or (value==scores[i,j] and wi<winners[i,j]):scores[i,j]=value;winners[i,j]=wi
  state['batches']+=1;state['max_pair_tokens']=max(state['max_pair_tokens'],width)
  if end//1000!=begin//1000 or end==len(slots):
   scores.flush();winners.flush();elapsed=time.perf_counter()-checkpoint
   state.update(completed=end,seconds=state['seconds']+elapsed,total_windows=len(slots));save(statepath,state);checkpoint=time.perf_counter()
   if time.perf_counter()-last>25 or end==len(slots):
    print(json.dumps({'dataset':key,'reranker':name,'windows_done':end,'total':len(slots),'seconds':round(state['seconds'],1)}),flush=True);last=time.perf_counter()
  begin=end
 windows={}
 for (i,j),ranges in bounds.items():
  wi=int(winners[i,j]);assert 0<=wi<len(ranges) and scores[i,j]>-1e19
  start,end=ranges[wi]
  windows[i,j]={'count':len(ranges),'index':wi,'text':texts[j] if len(ranges)==1 else tokenizer.decode(dt[j][start:end],skip_special_tokens=True)}
 execution={'backend':'common_HF_PyTorch_CUDA','model':model_name,'revision':MODEL_REVISIONS[model_name],'dtype':'float16','view':view,
   'unique_query_document_pairs':len(bounds),'scored_windows':len(slots),'multi_window_pairs':sum(len(x)>1 for x in bounds.values()),'max_pair_tokens':state['max_pair_tokens'],
   'truncated_pairs':0,'pair_token_limit':1024,'overlap_tokens':128,'window_aggregation':'max raw logit; earliest window on tie',
   'seconds_shared_between_engines':state['seconds'],'vram_peak_bytes':int(torch.cuda.max_memory_allocated()),'engine_serves_reranker':False,
   'score_identity':identity,'token_window_text':'full view for one window, decoded exact passage tokens for split windows'}
 save(root/'execution.json',execution)
 for engine in ENGINES:
  a=np.load(base(key,engine)/'cache/A-scores.npy');b=np.load(base(key,engine)/'cache/B-scores.npy');c=pools[engine]
  ranked=[sorted(ids,key=lambda j:(-float(scores[i,j]),j)) for i,ids in enumerate(c)]
  summarize(key,engine,name,qs,docs,ranked,scores,execution)
  export_log(key,engine,name,qs,docs,c,a,b,scores,windows)
 del scores,winners;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('reranker',choices=['bge','jina']);p.add_argument('--datasets',nargs='+',choices=KEYS,default=list(KEYS));a=p.parse_args()
 import torch
 from transformers import AutoTokenizer,AutoModelForSequenceClassification
 torch.manual_seed(20261007);torch.backends.cuda.matmul.allow_tf32=False
 name='BAAI/bge-reranker-v2-m3' if a.reranker=='bge' else 'jinaai/jina-reranker-v2-base-multilingual'
 tok=AutoTokenizer.from_pretrained(model_path(name))
 model=AutoModelForSequenceClassification.from_pretrained(model_path(name),trust_remote_code=a.reranker=='jina',torch_dtype=torch.float16,
   **({'use_flash_attn':False} if a.reranker=='jina' else {})).to('cuda').eval()
 for key in a.datasets:
  for view in ('question','question_answer'):execute(model,tok,key,a.reranker,view)
