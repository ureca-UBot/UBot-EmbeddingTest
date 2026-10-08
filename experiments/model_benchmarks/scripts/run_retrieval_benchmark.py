"""Sequential, resumable A-F exploration; holdout requires a frozen selection.

No source expected answers, oracle queries or GT are used as retrieval inputs.
Source labels remain provisional and are never silently marked approved.
"""
from __future__ import annotations
import argparse, gc, json, time, hashlib, statistics, sys, platform
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import torch
from jsonschema import Draft202012Validator
from retrieval_common import fact_metrics, merge_dense_views, semantic_contract_errors, stable_hash, no_faq_metrics

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'faq/retrieval_run_v2'
OUT=ROOT/'outputs/retrieval/run-20261006-v1'
MODEL_PATHS=ROOT/'configs/resolved_model_revisions.json'

def jsread(path):return json.loads(path.read_text(encoding='utf-8'))
def lines(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def save(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def save_lines(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in data),encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def sync():
    if torch.cuda.is_available():torch.cuda.synchronize()
def cleanup():gc.collect();torch.cuda.empty_cache()
def query(c):
    if c['query_mode']=='frozen_contextual_rewrite':
        # Fixed conservative history rule, no GT-dependent reference resolution.
        history='\n'.join(f"{h['role']}: {h['content']}" for h in c['history'])
        return '대화 이력:\n'+history+'\n현재 질문: '+c['query']
    return c['query']

def dataset(split):
    manifest=jsread(DATA/'split_manifest.json')
    for name,value in manifest['hashes'].items():
        if sha(DATA/name)!=value:raise ValueError('Frozen dataset changed: '+name)
    cases=[c for c in lines(DATA/'cases.jsonl') if c['split']==split]
    corpus=lines(DATA/'corpus.jsonl')
    schema=jsread(ROOT/'schemas/retrieval_case.schema.json')
    validator=Draft202012Validator(schema)
    errors=[]
    for c in cases:errors.extend(f"{c['case_id']}: {e.message}" for e in validator.iter_errors(c))
    errors+=semantic_contract_errors(lines(DATA/'cases.jsonl'),{'current-faq-1024-v1':{d['faq_id']:d for d in corpus}})
    if errors:raise ValueError('\n'.join(errors[:30]))
    return cases,corpus

def setup():
    torch.manual_seed(20261006);np.random.seed(20261006)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'runtime.json',{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'dtype':'float32','effective_max_tokens':512,'source_labels':'provisional','models':jsread(MODEL_PATHS),'dataset_manifest_hash':sha(DATA/'split_manifest.json')})

def model_path(name):return jsread(MODEL_PATHS)[name]['snapshot_path']
def text(d,view):return d['question'] if view=='question' else f"질문: {d['question']}\n답변: {d['answer']}"
def rank(scores,ids):return sorted(range(len(ids)),key=lambda i:(-float(scores[i]),ids[i]))

def report(name,cases,rankings,score_rows,ids,split='calibration',extra=None):
    extra=dict(extra or {})
    extra['peak_torch_vram_bytes']=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    rows=[]
    for c,ranking,scores in zip(cases,rankings,score_rows):
        ordered=[ids[i] for i in ranking]
        m={str(k):fact_metrics(ordered,c['fact_groups'],k) for k in [1,3,5,10,20]}
        rows.append({'case_id':c['case_id'],'family_id':c['family_id'],'cohort':c['reporting_cohort'],'source_category':c['source']['category'],'answerability':c['answerability'],'query':query(c),'query_hash':stable_hash(query(c)),'top_ids':ordered[:40],'top_scores':[float(scores[i]) for i in ranking[:40]],'metrics':m,'rank_hash':stable_hash(ordered[:20])})
    summaries={}
    for cohort in sorted(set(r['cohort'] for r in rows)):
        rr=[r for r in rows if r['cohort']==cohort and r['metrics']['3']['fact_recall'] is not None]
        grouped=defaultdict(list)
        for r in rr:grouped[r['family_id']].append(r)
        summaries[cohort]={'rows':sum(r['cohort']==cohort for r in rows),'ranking_rows':len(rr),'families':len(grouped),'family_mean':{f'{metric}@{k}':float(np.mean([np.mean([r['metrics'][str(k)][metric] for r in group]) for group in grouped.values()])) if grouped else None for k in [1,3,5,10,20] for metric in ['semantic_hit','fact_recall','all_facts_hit','semantic_mrr','fact_mrr']}}
    path=OUT/split/name
    save_lines(path/'per_query.jsonl',rows)
    summary={'experiment':name,'split':split,'case_count':len(cases),'label_status':'provisional_source_labels','cohorts':summaries,'extra':extra or {}}
    save(path/'summary.json',summary)
    print(json.dumps({'stage':name,'split':split,'core_general':summaries.get('core_general'), 'extra':extra or {}},ensure_ascii=False),flush=True)
    return rows

def dense(view,split='calibration'):
    from sentence_transformers import SentenceTransformer
    cases,corpus=dataset(split);ids=[d['faq_id'] for d in corpus]
    model=SentenceTransformer(model_path('BAAI/bge-m3'),device='cuda',model_kwargs={'dtype':torch.float32})
    model.max_seq_length=512
    texts=[text(d,view) for d in corpus]
    corpus_file=OUT/f'cache/dense-{view}-corpus.npy';corpus_file.parent.mkdir(exist_ok=True)
    times={};sync();start=time.perf_counter()
    if corpus_file.exists():dv=np.load(corpus_file)
    else:dv=model.encode(texts,batch_size=16,normalize_embeddings=True,show_progress_bar=True);np.save(corpus_file,dv)
    sync();times['corpus_encode_or_cache_seconds']=time.perf_counter()-start
    queries=list(dict.fromkeys(query(c) for c in cases));sync();start=time.perf_counter()
    qfile=OUT/f'cache/dense-{split}-queries.npy'
    if qfile.exists():qv=np.load(qfile)
    else:qv=model.encode(queries,batch_size=16,normalize_embeddings=True,show_progress_bar=True);np.save(qfile,qv);save(OUT/f'cache/dense-{split}-query_texts.json',queries)
    sync();times['query_encode_or_cache_seconds']=time.perf_counter()-start
    index={q:i for i,q in enumerate(queries)}
    scores=np.asarray(qv,dtype=np.float32)@np.asarray(dv,dtype=np.float32).T
    expanded=np.stack([scores[index[query(c)]] for c in cases])
    np.save(OUT/f'cache/{split}-{view}-scores.npy',expanded)
    rankings=[rank(s,ids) for s in expanded]
    report('A' if view=='question' else 'B',cases,rankings,expanded,ids,split,times)
    tok=model.tokenizer
    token_counts=[len(tok(t,add_special_tokens=True,truncation=False)['input_ids']) for t in texts+queries]
    save(OUT/split/('A' if view=='question' else 'B')/'token_lengths.json',{'max':max(token_counts),'truncated_count':sum(n>512 for n in token_counts),'count':len(token_counts)})
    del model;cleanup()

def dual(split='calibration'):
    cases,corpus=dataset(split);ids=[d['faq_id'] for d in corpus]
    a=np.load(OUT/f'cache/{split}-question-scores.npy');b=np.load(OUT/f'cache/{split}-question_answer-scores.npy')
    maximum=np.maximum(a,b)
    for budget in [3,5,10,20]:
        ranked=[];unions=[];pool=[]
        for aa,bb in zip(a,b):
            result=merge_dense_views(dict(zip(ids,aa)),dict(zip(ids,bb)),budget,budget)
            ranked.append([ids.index(fid) for fid in result['top_m_ids']]);unions.append(result['actual_u']);pool.append(result)
        report('C-M'+str(budget),cases,ranked,maximum,ids,split,{'per_view_k':budget,'matched_budget':budget,'union_count_mean':float(np.mean(unions)),'union_count_max':max(unions)})
        save_lines(OUT/f'{split}/C-M{budget}/candidate_pools.jsonl',[{'case_id':c['case_id'],**r} for c,r in zip(cases,pool)])
        # Union and volume controls are not mislabeled Recall@budget.
        diagnostic=[]
        for i,(c,p) in enumerate(zip(cases,pool)):
            ra=[ids[j] for j in rank(a[i],ids)];rb=[ids[j] for j in rank(b[i],ids)]
            diagnostic.append({'case_id':c['case_id'],'union_fact_recall':fact_metrics(p['raw_union_ids'],c['fact_groups'],max(1,p['actual_u']))['fact_recall'],'actual_u':p['actual_u'],'A_top_2k':fact_metrics(ra,c['fact_groups'],2*budget)['fact_recall'],'B_top_2k':fact_metrics(rb,c['fact_groups'],2*budget)['fact_recall'],'A_top_actual_u':fact_metrics(ra,c['fact_groups'],p['actual_u'])['fact_recall'],'B_top_actual_u':fact_metrics(rb,c['fact_groups'],p['actual_u'])['fact_recall']})
        save_lines(OUT/f'{split}/C-M{budget}/volume_controls.jsonl',diagnostic)

def rerank(which,view,split='calibration'):
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    cases,corpus=dataset(split);ids=[d['faq_id'] for d in corpus];docs={d['faq_id']:d for d in corpus}
    name='BAAI/bge-reranker-v2-m3' if which=='bge' else 'jinaai/jina-reranker-v2-base-multilingual'
    model=AutoModelForSequenceClassification.from_pretrained(model_path(name),trust_remote_code=which=='jina',torch_dtype=torch.float32,**({'use_flash_attn':False} if which=='jina' else {})).to('cuda').eval()
    tokenizer=AutoTokenizer.from_pretrained(model_path(name),trust_remote_code=which=='jina')
    pools={m:{p['case_id']:p['top_m_ids'] for p in lines(OUT/f'{split}/C-M{m}/candidate_pools.jsonl')} for m in [10,20]}
    a=np.load(OUT/f'cache/{split}-question-scores.npy')
    a20={c['case_id']:[ids[j] for j in rank(s,ids)[:20]] for c,s in zip(cases,a)}
    # Encode each unique query/document pair once; use identical scores in M10/M20.
    pair_keys=[];pairs=[];seen=set()
    for c in cases:
        union=list(dict.fromkeys(pools[20][c['case_id']]+pools[10][c['case_id']]+(a20[c['case_id']] if which=='bge' else [])))
        for fid in union:
            key=(query(c),fid)
            if key not in seen:seen.add(key);pair_keys.append(key);pairs.append([key[0],text(docs[fid],view)])
    values=[];start=time.perf_counter()
    for begin in range(0,len(pairs),24):
        batch=pairs[begin:begin+24]
        with torch.inference_mode():
            inputs=tokenizer(batch,padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
            v=model(**inputs).logits.reshape(-1).float().cpu().numpy();values.extend(v.tolist())
        if begin%2400==0:print(f'D {which} {view}: {begin}/{len(pairs)} pairs',flush=True)
    sync();elapsed=time.perf_counter()-start
    scoremap=dict(zip(pair_keys,values))
    if any(not np.isfinite(v) for v in values):raise ValueError('Nonfinite reranker scores')
    for m in [10,20]:
        scores=[];rankings=[]
        for c in cases:
            pool=pools[m][c['case_id']];ss=np.full(len(ids),-1e20,dtype=np.float32)
            for fid in pool:ss[ids.index(fid)]=scoremap[(query(c),fid)]
            rankings.append([ids.index(fid) for fid in sorted(pool,key=lambda fid:(-scoremap[(query(c),fid)],fid))]);scores.append(ss)
        name=f'D-{which}-{view}-M{m}'
        rows=report(name,cases,rankings,scores,ids,split,{'pairs_scored':len(pairs),'scoring_seconds':elapsed,'raw_score_adapter':'identity_no_second_sigmoid'})
        baseline={r['case_id']:r for r in lines(OUT/f'{split}/C-M{m}/per_query.jsonl')}
        changes=[]
        for c,r in zip(cases,rows):
            before=baseline[c['case_id']];bm=before['metrics']['3'];am=r['metrics']['3']
            delta=None if bm['semantic_mrr'] is None else am['semantic_mrr']-bm['semantic_mrr']
            changes.append({'case_id':c['case_id'],'rr_delta_at_3':delta,'all_facts_before':bm['all_facts_hit'],'all_facts_after':am['all_facts_hit'],'stage1_missing_facts':[g['fact_id'] for g in c['fact_groups'] if not(set(g['primary_ids']+g['acceptable_ids'])&set(pools[m][c['case_id']]))],'status':'NOT_APPLICABLE' if delta is None else ('IMPROVED' if delta>0 else 'WORSENED' if delta<0 else 'UNCHANGED')})
        save_lines(OUT/f'{split}/{name}/rank_changes.jsonl',changes)
    if which=='bge':
        for m in [10,20]:
            scores=[];rankings=[]
            for c in cases:
                pool=a20[c['case_id']][:m];ss=np.full(len(ids),-1e20,dtype=np.float32)
                for fid in pool:ss[ids.index(fid)]=scoremap[(query(c),fid)]
                rankings.append(sorted([ids.index(fid) for fid in pool],key=lambda i:(-ss[i],ids[i])));scores.append(ss)
            report(f'D-Acontrol-bge-{view}-M{m}',cases,rankings,scores,ids,split)
    del model;cleanup()

def legacy(key):
    from sentence_transformers import SentenceTransformer
    d=jsread(DATA/(key+'.json'));corpus=d['corpus'];ids=[r['faq_id'] for r in corpus]
    cases=[]
    for r in d['queries']:
        v=r['values']
        if v.get('처리 의도')!='FAQ_RAG':continue
        primary=re_split(v.get('Primary GT'))
        acceptable=re_split(v.get('Acceptable GT') or v.get('허용 GT'))
        if not primary:raise ValueError('missing legacy GT')
        cases.append({'case_id':key+'-'+str(v['ID']),'family_id':key+'-'+str(v['ID']),'reporting_cohort':'legacy_regression','source':{'category':r['sheet']},'answerability':'full','query':v['사용자 질문'],'query_mode':'standalone','fact_groups':[{'fact_id':'f1','primary_ids':primary,'acceptable_ids':acceptable}]})
    model=SentenceTransformer(model_path('BAAI/bge-m3'),device='cuda',model_kwargs={'dtype':torch.float32});model.max_seq_length=512
    dv=model.encode([r['question'] for r in corpus],batch_size=16,normalize_embeddings=True);qv=model.encode([query(c) for c in cases],batch_size=16,normalize_embeddings=True);scores=qv@dv.T
    report(key,cases,[rank(s,ids) for s in scores],scores,ids,'regression')
    del model;cleanup()

def re_split(v):
    import re
    return [x.strip() for x in re.split(r'[,;]',str(v or '')) if x.strip()]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['A','B','C','D','legacy'])
    parser.add_argument('--split',choices=['calibration','holdout'],default='calibration')
    parser.add_argument('--reranker',choices=['bge','jina'],default='bge')
    parser.add_argument('--view',choices=['question','question_answer'],default='question')
    parser.add_argument('--legacy-key',choices=['legacy_60','legacy_235'],default='legacy_235')
    args=parser.parse_args();setup()
    if args.split=='holdout' and not (OUT/'selection.json').exists():raise ValueError('Holdout requires frozen selection')
    if args.stage=='A':dense('question',args.split)
    if args.stage=='B':dense('question_answer',args.split)
    if args.stage=='C':dual(args.split)
    if args.stage=='D':rerank(args.reranker,args.view,args.split)
    if args.stage=='legacy':legacy(args.legacy_key)

if __name__=='__main__':main()
