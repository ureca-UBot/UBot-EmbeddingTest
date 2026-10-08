"""Frozen, scoped v4 inference and evaluation helpers. No v1/v2 input loading."""
import gzip,hashlib,json,os,sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
DATASET_ROOT=ROOT.parent
sys.path.insert(0,str(DATASET_ROOT))
from dataset import Dataset,score
OUT=ROOT/'outputs/run-v4'
KEYS=('existing','kt','skt','lgu')
ENGINES=('ollama','vllm')
KS=(1,3,5,10,20)
MODEL_REVISIONS={
 'BAAI/bge-m3':'5617a9f61b028005a4858fdac845db406aefb181',
 'BAAI/bge-reranker-v2-m3':'953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e',
 'jinaai/jina-reranker-v2-base-multilingual':'9cfeff2df7d40d1b78e75e5e9cebec92a99813c9'}
FIELDS=['query','expected_faq_id','expected_faq_ids','faq_id','retrieval_rank','retrieval_cosine','retrieval_a_rank','retrieval_a_cosine','retrieval_b_rank','retrieval_b_cosine','rerank_rank','rerank_score','rank_delta','window_count','winning_window_index','winning_window_text','winning_window_score']
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def token(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p):
 opener=gzip.open if str(p).endswith('.gz') else open
 with opener(p,'rt',encoding='utf-8-sig') as f:return [json.loads(x) for x in f if x.strip()]
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(t,p)
def save_lines(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 opener=gzip.open if str(p).endswith('.gz') else open
 with opener(t,'wt',encoding='utf-8',**({'compresslevel':3} if opener is gzip.open else {'buffering':4194304})) as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
 os.replace(t,p)
def model_path(name):return '/models/hf/hub/models--'+name.replace('/','--')+'/snapshots/'+MODEL_REVISIONS[name]
def load(key):
 d=Dataset(key,DATASET_ROOT)
 if (OUT/'frozen_manifest.json').exists():
  frozen=read(OUT/'frozen_manifest.json')['datasets'][key]
  assert d.metadata==frozen,'Dataset changed since preflight'
 qs=lines(d.path/'user_questions.jsonl')
 assert sha(d.path/'user_questions.jsonl')==d.metadata['questions_sha256']
 for s in ('development','calibration','holdout'):d.labels(s,allow_unreviewed=True)
 return d,qs,d.corpus
def base(key,engine):return OUT/key/engine
def doc_text(d,view):return d['question'] if view=='question' else d['question']+'\n\n'+d['answer']
def ranks(scores):return np.argsort(-scores,axis=1,kind='stable')
def inverse(scores):
 r=ranks(scores);v=np.empty_like(r);np.put_along_axis(v,r,np.broadcast_to(np.arange(r.shape[1])+1,r.shape),axis=1);return v
def pool(a,b,k=20):
 ar,br=ranks(a),ranks(b);s=np.maximum(a,b)
 return [sorted(set(map(int,x[:k]))|set(map(int,y[:k])),key=lambda j:(-float(s[i,j]),j)) for i,(x,y) in enumerate(zip(ar,br))]
def lengths(tok,texts):
 result=[]
 for start in range(0,len(texts),64):result+=tok(texts[start:start+64],truncation=False,return_length=True)['length']
 return np.asarray(result,dtype=np.int32)
def append(p,row):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
def summarize(key,engine,name,qs,docs,ranked,scores,execution,union=None):
 out=base(key,engine)/name;ids=[d['faq_id'] for d in docs];rows=[]
 for i,(q,rr) in enumerate(zip(qs,ranked)):
  rr=list(map(int,rr));chosen=[ids[j] for j in rr[:20]]
  positions={ids[j]:p+1 for p,j in enumerate(rr)}
  fact_ranks=[min((positions[x] for x in f['acceptable_faq_ids'] if x in positions),default=None) for f in q['required_facts']]
  ms={str(k):score(q,[ids[j] for j in (union[k][i] if union else rr[:k])],max(1,len(union[k][i])) if union else k) for k in KS}
  rows.append(dict(question_id=q['question_id'],dataset_id=key,split=q['split'],leakage_group_id=q['leakage_group_id'],
    query=q['query'],model_query=q['model_query'],purpose=q['primary_purpose'],purpose_tags=q['purpose_tags'],
    evaluation_track=q['evaluation_track'],corpus_status=q['corpus_status'],expected_action=q['expected_action'],
    cue_location=q.get('cue_location'),evidence_layout=q.get('evidence_layout'),
    first_gold_rank=min((r for r in fact_ranks if r is not None),default=None),fact_best_ranks=fact_ranks,
    expected_faq_ids=q['answer_faq_ids'],required_facts=q['required_facts'],top_ids=chosen,
    top_scores=[float(scores[i,j]) for j in rr[:20]],top1_margin=float(scores[i,rr[0]]-scores[i,rr[1]]),
    candidate_count=len(rr),metrics=ms))
 groups=defaultdict(list)
 for r in rows:
  groups['all_supported_subtasks'].append(r)
  groups['split:'+r['split']].append(r)
  groups['track:'+r['evaluation_track']].append(r)
  groups['status:'+r['corpus_status']].append(r)
  groups['cue:'+str(r['cue_location'])].append(r)
  groups['evidence:'+str(r['evidence_layout'])].append(r)
  if r['corpus_status']=='FAQ_EXISTS' and r['evaluation_track'] not in ('context','tool_lookup','special_conditions'):
   groups['core'].append(r)
  for p in r['purpose_tags']:
   groups['purpose:'+p].append(r);groups[r['split']+'/purpose:'+p].append(r)
 result={}
 for label,group in groups.items():
  valid=[r for r in group if r['metrics']['1']['hit'] is not None]
  means={f'{m}@{k}':float(np.mean([r['metrics'][str(k)][m] for r in valid])) if valid else None for k in KS for m in ('hit','fact_recall','all_facts_hit','mrr')}
  bygroup=defaultdict(list)
  for r in valid:bygroup[r['leakage_group_id']].append(r)
  gm={f'{m}@{k}':float(np.mean([np.mean([r['metrics'][str(k)][m] for r in v]) for v in bygroup.values()])) if bygroup else None for k in KS for m in ('hit','fact_recall','all_facts_hit','mrr')}
  result[label]={'rows':len(group),'scored_rows':len(valid),'groups':len(bygroup),'row_mean':means,'leakage_group_macro':gm}
 save_lines(out/'per_query.jsonl',rows)
 save(out/'summary.json',{'dataset_id':key,'engine':engine,'structure':name,'inference_rows':len(qs),
   'groups':result,'execution':execution,'candidate_budget':'per-view K, raw full union' if union else 'ranked prefix K',
   'threshold_fitted':False,'label_status':'independent_review_pending; synthetic diagnostic',
   'holdout_use':'predeclared matrix diagnostic; no tuning or selection based on this run'})
 print(json.dumps({'dataset':key,'engine':engine,'complete':name,'Hit1':result['all_supported_subtasks']['row_mean']['hit@1'],'FactRecall10':result['all_supported_subtasks']['row_mean']['fact_recall@10']}),flush=True)
 return rows

def export_log(key,engine,name,qs,docs,candidates,a,b,rerank=None,windows=None,representation_scores=None):
 out=base(key,engine)/name;ai,bi=inverse(a),inverse(b);merged=np.maximum(a,b);count=0;transitions=[]
 def records():
  nonlocal count
  for i,(q,ids) in enumerate(zip(qs,candidates)):
   ids=list(map(int,ids));ordered=sorted(ids,key=lambda j:(-float(rerank[i,j]),j)) if rerank is not None else ids
   ri={j:r+1 for r,j in enumerate(ordered)};gold=set(q['answer_faq_ids'])
   before=min((p+1 for p,j in enumerate(ids) if docs[j]['faq_id'] in gold),default=None)
   after=min((p+1 for p,j in enumerate(ordered) if docs[j]['faq_id'] in gold),default=None)
   outcome='UNSCORED' if not gold else 'STAGE1_MISS' if before is None else 'IMPROVED' if after<before else 'WORSENED' if after>before else 'UNCHANGED'
   transitions.append(dict(question_id=q['question_id'],split=q['split'],before_rank=before,after_rank=after,outcome=outcome))
   for p,j in enumerate(ids):
    s=float(rerank[i,j]) if rerank is not None else None
    win=windows.get((i,j)) if windows else None
    cosine=float(a[i,j] if name=='A' else b[i,j] if name=='B' else merged[i,j]) if representation_scores is None else None
    row=dict(dataset_id=key,engine=engine,structure=name,question_id=q['question_id'],split=q['split'],
      query=q['query'],model_query=q['model_query'],expected_faq_id=q['answer_faq_ids'],expected_faq_ids=q['answer_faq_ids'],
      required_facts=q['required_facts'],faq_id=docs[j]['faq_id'],retrieval_rank=p+1,retrieval_cosine=cosine,
      retrieval_a_rank=int(ai[i,j]),retrieval_a_cosine=float(a[i,j]),retrieval_b_rank=int(bi[i,j]),retrieval_b_cosine=float(b[i,j]),
      rerank_rank=ri[j] if rerank is not None else None,rerank_score=s,rank_delta=p+1-ri[j] if rerank is not None else None,
      window_count=win['count'] if win else None,winning_window_index=win['index'] if win else None,
      winning_window_text=win['text'] if win else None,winning_window_score=s,
      retrieval_representation_score=float(representation_scores[i,j]) if representation_scores is not None else None)
    assert set(FIELDS)<=set(row);count+=1;yield row
 path=out/'candidate_log.jsonl.gz';save_lines(path,records());save_lines(out/'rank_changes.jsonl',transitions)
 save(out/'candidate_log_schema.json',{'row_count':count,'sha256':sha(path),'fields':FIELDS,
   'rank_base':1,'window_index_base':0,'null':'not applicable; never a fabricated zero',
   'retrieval_rank':'rank within stage-1 list; A/B ranks are full-corpus ranks',
   'rerank_sort':'raw logit descending, ties by corpus index; max over full document windows',
   'expected_faq_id':'legacy key stores all accepted IDs, same array as expected_faq_ids'})
