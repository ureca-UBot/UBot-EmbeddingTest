import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
OUT=ROOT/'outputs/run-v1'
MAIN_N=30000
REFERENCE_N=3000
N=MAIN_N+REFERENCE_N
MAX_LENGTH=8192
RERANK_MAX_LENGTH=1024
NAMES=['A','B','C-M20','D-bge-question-M20','D-bge-question_answer-M20',
       'D-jina-question-M20','D-jina-question_answer-M20','E-sparse',
       'E-dense-sparse-RRF-M20','F-full','F-dense-pool-M20']

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(r) for r in p.read_text(encoding='utf-8-sig').splitlines() if r]
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,p)
def save_lines(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as stream:
        for r in rows:stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(tmp,p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def inputs():
    manifest=read(DATA/'manifest.json')
    for name,hash_value in manifest['hashes'].items():
        assert sha(DATA/name)==hash_value,'Frozen input changed: '+name
    cases,docs=lines(DATA/'cases.jsonl'),lines(DATA/'corpus.jsonl')
    assert len(cases)==N and len(docs)==manifest['corpus_faqs']==3246
    assert sum(c['evaluation_role']=='main' for c in cases)==MAIN_N
    return cases,docs
def model_path(name):return read(ROOT/'configs/resolved_model_revisions.json')[name]['snapshot_path']
def doc_text(d,view):
    prefix=f"{d['brand']} / {d['category']}\nQuestion: {d['question']}"
    return prefix if view=='question' else prefix+'\nAnswer: '+d['answer']
def rankings(scores):
    return np.argsort(-scores,axis=1,kind='stable').astype(np.int16)
def pools(a,b,m=20):
    ar,br=rankings(a),rankings(b)
    ss=np.maximum(a,b)
    result=[]
    for i,(aa,bb) in enumerate(zip(ar,br)):
        ids=set(map(int,aa[:m]))|set(map(int,bb[:m]))
        result.append(sorted(ids,key=lambda j:(-float(ss[i,j]),j))[:m])
    return result
def metrics(ranked,gold,k):
    if not gold:return {'hit':None,'coverage':None,'all_sources':None,'mrr':None}
    positions={v:i+1 for i,v in enumerate(ranked[:k])}
    matched=[positions[f] for f in gold if f in positions]
    return {'hit':int(bool(matched)),'coverage':len(matched)/len(gold),
            'all_sources':int(len(matched)==len(gold)), 'mrr':1/min(matched) if matched else 0.}
def summarize(engine,name,cases,docs,ranked,scores,extra):
    folder=OUT/'engines'/engine/name
    if (folder/'summary.json').exists():raise RuntimeError('Completed stage cannot be overwritten: '+name)
    ids=[d['faq_id'] for d in docs]
    rows=[]
    for i,(c,rr) in enumerate(zip(cases,ranked)):
        rr=list(map(int,rr[:20]))
        top=[ids[j] for j in rr[:20]]
        values=[float(scores[i,j]) for j in rr[:20]]
        rows.append({k:c[k] for k in ['case_id','parent_case_id','type','cohort','brand','category','primary_variation','validation_tags','evaluation_role','family_id','variant_slot','language_pair_reference','source_language','source_ids']} |
                    {'query':c['query'],'query_hash':hashlib.sha256(c['model_query'].encode()).hexdigest(),
                     'top_ids':top,'top_scores':values,'metrics':{str(k):metrics(top,c['source_ids'],k) for k in [1,3,5,10,20]},
                     'rank_hash':hashlib.sha256(json.dumps(top).encode()).hexdigest()})
    metric_keys=[f'{metric}@{k}' for k in [1,3,5,10,20] for metric in ['hit','coverage','all_sources','mrr']]
    metric_values=np.asarray([[r['metrics'][str(k)][metric] for k in [1,3,5,10,20] for metric in ['hit','coverage','all_sources','mrr']] for r in rows],dtype=np.float64)
    row_indices={id(r):i for i,r in enumerate(rows)}
    groups={}
    for kind in ['cohort','brand','source_language','type','primary_variation','validation_tag']:
        grouped=defaultdict(list)
        for r in rows:
            for key in r['validation_tags'] if kind=='validation_tag' else [r[kind]]:
                grouped[key].append(r)
        values={}
        for key,group in sorted(grouped.items()):
            # Reference queries are a diagnostic group, never mixed into main
            # category/type/tag/language aggregates or architecture selection.
            valid=[r for r in group if r['metrics']['3']['hit'] is not None and
                   (kind=='cohort' or r['evaluation_role']=='main')]
            families,parents=defaultdict(list),defaultdict(list)
            for r in valid:
                families[r['family_id']].append(r)
                parents[r['parent_case_id']].append(r)
            vv=metric_values[[row_indices[id(r)] for r in valid]]
            def average(field=None):
                if not valid:return {k:None for k in metric_keys}
                if field:
                    _,inverse=np.unique([r[field] for r in valid],return_inverse=True)
                    counts=np.bincount(inverse)
                    means=np.asarray([np.bincount(inverse,weights=vv[:,j])/counts for j in range(len(metric_keys))]).mean(axis=1)
                else:means=vv.mean(axis=0)
                return {k:float(v) for k,v in zip(metric_keys,means)}
            values[key]={'rows':len(valid),'families':len(families),'parents':len(parents),
                         'row_mean':average(),'family_mean':average('family_id'),
                         'parent_mean':average('parent_case_id')}
        groups[kind]=values
    summary={'engine':engine,'structure':name,'case_count':len(rows),'main_case_count':MAIN_N,
             'reference_case_count':REFERENCE_N,'same_full_30000_main_input':True,
             'metric_contract':'canonical FAQ source support, not independently reviewed atomic-fact coverage',
             **groups,'execution':extra}
    save_lines(folder/'per_query.jsonl',rows)
    save(folder/'summary.json',summary)
    print(json.dumps({'completed':name,'engine':engine,'main_rows':MAIN_N,'reference_rows':REFERENCE_N,
                      'general':groups['cohort']['general']['parent_mean'],'execution':extra},ensure_ascii=False),flush=True)
    return rows,summary
def calibrate_policy(rows,cases):
    included=[i for i,c in enumerate(cases) if c['evaluation_role']=='main']
    scores=np.asarray([rows[i]['top_scores'][0] for i in included])
    margins=np.asarray([rows[i]['top_scores'][0]-rows[i]['top_scores'][1] for i in included])
    assert not any(c['no_faq_truth'] for c in cases)
    return {'status':'NOT_FITTED_NO_VALIDATED_NEGATIVES','threshold':None,'margin':None,
            'no_faq_rows':0,'answerable_rows':MAIN_N,'same_set_diagnostics':True,'holdout_evaluation':False,
            'no_faq_precision':None,'no_faq_recall':None,'no_faq_f1':None,
            'score_quantiles':{str(q):float(np.quantile(scores,q)) for q in [0,.01,.05,.5,.95,.99,1]},
            'gap_quantiles':{str(q):float(np.quantile(margins,q)) for q in [0,.01,.05,.5,.95,.99,1]},
            'reason':'All source-backed cases are answerable; a NO_FAQ threshold cannot be validated from positives only.'}
