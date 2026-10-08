import hashlib,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data';OUT=ROOT/'outputs/run-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(r) for r in p.read_text(encoding='utf-8-sig').splitlines() if r]
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def save_lines(p,rows):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inputs():
    m=read(DATA/'manifest.json')
    for name,h in m['hashes'].items():assert sha(DATA/name)==h,'Frozen input changed: '+name
    cs=lines(DATA/'cases.jsonl');ds=lines(DATA/'corpus.jsonl');assert len(cs)==3000 and len(ds)==1024
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
                     'family_id':c['family_id'],'query':c['query'],'query_hash':hashlib.sha256(c['model_query'].encode()).hexdigest(),
                     'top_ids':top,'top_scores':values,
                     'metrics':{str(k):metrics(top,c['source_ids'],k) for k in [1,3,5,10,20]},
                     'rank_hash':hashlib.sha256(json.dumps(top).encode()).hexdigest()})
    groups={}
    for group_kind in ['cohort','type']:
        values={}
        for key in sorted({r[group_kind] for r in rows}):
            rr=[r for r in rows if r[group_kind]==key];valid=[r for r in rr if r['metrics']['3']['hit'] is not None]
            families=defaultdict(list)
            for r in valid:families[r['family_id']].append(r)
            flat={f'{metric}@{k}':float(np.mean([r['metrics'][str(k)][metric] for r in valid])) if valid else None
                  for k in [1,3,5,10,20] for metric in ['hit','coverage','all_sources','mrr']}
            macro={f'{metric}@{k}':float(np.mean([np.mean([r['metrics'][str(k)][metric] for r in fam]) for fam in families.values()])) if families else None
                  for k in [1,3,5,10,20] for metric in ['hit','coverage','all_sources','mrr']}
            values[key]={'rows':len(rr),'ranking_rows':len(valid),'families':len(families),'row_mean':flat,'family_mean':macro}
        groups[group_kind]=values
    summary={'engine':engine,'structure':name,'case_count':len(rows),'same_full_3000_input':True,
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
                    'rows':len(included),'no_faq_rows':int(truth.sum()),'same_set_fit_and_report':True,'holdout_evaluation':False}
            key=(f1,accept,-float(t),-float(m))
            if best is None or key>best[0]:best=(key,result)
    return best[1]
