import hashlib,json,time,os
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data';OUT=ROOT/'outputs/run-v1'
N=30000;MAX_LENGTH=1024
NAMES=['A','B','C-M20','D-bge-question-M20','D-bge-question_answer-M20','D-jina-question-M20',
       'D-jina-question_answer-M20','E-sparse','E-dense-sparse-RRF-M20','F-full','F-dense-pool-M20']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(r) for r in p.read_text(encoding='utf-8-sig').splitlines() if r]
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def save_lines(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(tmp,p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def inputs():
    m=read(DATA/'manifest.json')
    for name,h in m['hashes'].items():assert sha(DATA/name)==h,'Frozen input changed: '+name
    cs=lines(DATA/'cases.jsonl');ds=lines(DATA/'corpus.jsonl');assert len(cs)==N and len(ds)==1024
    return cs,ds
def model_path(name):return read(ROOT/'configs/resolved_model_revisions.json')[name]['snapshot_path']
def doc_text(d,view):return d['question'] if view=='question' else f"질문: {d['question']}\n답변: {d['answer']}"
def rankings(scores):return np.argsort(-scores,axis=1,kind='stable')
def pools(a,b,m=20):
    aq=rankings(a);bq=rankings(b);ss=np.maximum(a,b)
    result=[]
    for i,(aa,bb) in enumerate(zip(aq,bq)):
        u=set(map(int,aa[:m]))|set(map(int,bb[:m]))
        result.append(sorted(u,key=lambda j:(-float(ss[i,j]),j))[:m])
    return result
def metrics(ranked,gold,k):
    if not gold:return {'hit':None,'coverage':None,'all_sources':None,'mrr':None}
    rr={v:i+1 for i,v in enumerate(ranked[:k])};matched=[rr[i] for i in gold if i in rr]
    return {'hit':int(bool(matched)),'coverage':len(matched)/len(gold),
            'all_sources':int(len(matched)==len(gold)),'mrr':1/min(matched) if matched else 0.}
def summarize(engine,name,cases,docs,ranked,scores,extra):
    folder=OUT/'engines'/engine/name
    if (folder/'summary.json').exists():raise RuntimeError('Completed stage cannot be overwritten: '+name)
    rows=[];ids=[d['faq_id'] for d in docs]
    for i,(c,rr) in enumerate(zip(cases,ranked)):
        rr=list(map(int,rr));top=[ids[j] for j in rr[:20]]
        values=[float(scores[i,j]) for j in rr[:20]]
        rows.append({'case_id':c['case_id'],'source_row':c['source_row'],'type':c['type'],'cohort':c['cohort'],
                     'parent_case_id':c['parent_case_id'],'primary_variation':c['primary_variation'],
                     'validation_tags':c['validation_tags'],'variation_repeat_group_id':c['variation_repeat_group_id'],
                     'family_id':c['family_id'],'query':c['query'],'query_hash':hashlib.sha256(c['model_query'].encode()).hexdigest(),
                     'top_ids':top,'top_scores':values,
                     'metrics':{str(k):metrics(top,c['source_ids'],k) for k in [1,3,5,10,20]},
                     'rank_hash':hashlib.sha256(json.dumps(top).encode()).hexdigest()})
    groups={}
    for group_kind in ['cohort','type','primary_variation','validation_tag']:
        values={}
        grouped=defaultdict(list)
        for r in rows:
            for key in r['validation_tags'] if group_kind=='validation_tag' else [r[group_kind]]:grouped[key].append(r)
        for key,rr in sorted(grouped.items()):
            valid=[r for r in rr if r['metrics']['3']['hit'] is not None]
            families=defaultdict(list)
            for r in valid:families[r['family_id']].append(r)
            parents=defaultdict(list)
            for r in valid:parents[r['parent_case_id']].append(r)
            flat={f'{metric}@{k}':float(np.mean([r['metrics'][str(k)][metric] for r in valid])) if valid else None
                  for k in [1,3,5,10,20] for metric in ['hit','coverage','all_sources','mrr']}
            macro={f'{metric}@{k}':float(np.mean([np.mean([r['metrics'][str(k)][metric] for r in fam]) for fam in families.values()])) if families else None
                  for k in [1,3,5,10,20] for metric in ['hit','coverage','all_sources','mrr']}
            parent_macro={f'{metric}@{k}':float(np.mean([np.mean([r['metrics'][str(k)][metric] for r in fam]) for fam in parents.values()])) if parents else None
                  for k in [1,3,5,10,20] for metric in ['hit','coverage','all_sources','mrr']}
            values[key]={'rows':len(rr),'ranking_rows':len(valid),'families':len(families),'parents':len(parents),
                         'row_mean':flat,'family_mean':macro,'parent_mean':parent_macro}
        groups[group_kind]=values
    summary={'engine':engine,'structure':name,'case_count':len(rows),'same_full_30000_input':True,
             'metric_contract':'strict workbook source ID coverage, not independently reviewed atomic-fact coverage',
             **groups,'execution':extra}
    save_lines(folder/'per_query.jsonl',rows);save(folder/'summary.json',summary)
    print(json.dumps({'completed':name,'engine':engine,'rows':len(rows),'general':groups['cohort']['general']['family_mean'],
                      'execution':extra},ensure_ascii=False),flush=True)
    return rows,summary

def calibrate_policy(rows,cases):
    included=[i for i,c in enumerate(cases) if c['cohort'] in ['general','condition','no_faq']]
    scores=np.array([rows[i]['top_scores'][0] for i in included]);gaps=np.array([rows[i]['top_scores'][0]-rows[i]['top_scores'][1] for i in included])
    truth=np.array([cases[i]['no_faq_truth'] for i in included])
    ts=np.unique(np.r_[scores.min()-1,np.quantile(scores,np.linspace(0,1,61)),scores.max()+1])
    ms=np.unique(np.r_[0,np.quantile(gaps,np.linspace(0,1,31)),gaps.max()+1]);best=None
    for t in ts:
        for m in ms:
            reject=(scores<t)|(gaps<m);tp=int(np.sum(truth&reject));fp=int(np.sum(~truth&reject));fn=int(np.sum(truth&~reject));tn=int(np.sum(~truth&~reject))
            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0;accept=tn/(tn+fp)
            result={'threshold':float(t),'margin':float(m),'no_faq_f1':float(f1),'answerable_acceptance':float(accept),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
                    'no_faq_precision':tp/(tp+fp) if tp+fp else 0.,'no_faq_recall':tp/(tp+fn) if tp+fn else 0.,
                    'no_faq_false_acceptance':fn/(tp+fn),'answerable_false_rejection':fp/(fp+tn),
                    'rows':len(included),'no_faq_rows':int(truth.sum()),'same_set_fit_and_report':True,'holdout_evaluation':False,
                    'reject_rule':'top1 < threshold OR top1-top2 < margin','threshold_grid_quantiles':61,'margin_grid_quantiles':31}
            key=(f1,accept,-float(t),-float(m))
            if best is None or key>best[0]:best=(key,result)
    result=best[1];reject=(scores<result['threshold'])|(gaps<result['margin'])
    good=np.array([bool(rows[i]['metrics']['3']['all_sources']) for i in included])
    result['answerable_accepted_and_all_sources3']=float(np.mean((~reject & good)[~truth]))
    result['score_quantiles']={str(q):float(np.quantile(scores,q)) for q in [0,.05,.5,.95,1]}
    result['gap_quantiles']={str(q):float(np.quantile(gaps,q)) for q in [0,.05,.5,.95,1]}
    return result
