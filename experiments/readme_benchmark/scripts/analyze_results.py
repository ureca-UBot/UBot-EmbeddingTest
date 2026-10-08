"""Analyze failures and cross-language drift without changing frozen labels."""
import json
from collections import Counter,defaultdict
import numpy as np
from run_retrieval_benchmark import OUT,DATA,lines,jsread,save,save_lines

def main():
    cases={c['case_id']:c for c in lines(DATA/'cases.jsonl')}
    chosen=jsread(OUT/'selection.json')['selected'];comparisons={}
    runs=[('calibration',stage) for stage in dict.fromkeys(['A','B','C-M20',chosen])]+[('holdout',chosen)]
    for split,stage in runs:
        source=OUT/'holdout/per_query.jsonl' if split=='holdout' else OUT/'calibration'/stage/'per_query.jsonl'
        rows=lines(source);byid={r['case_id']:r for r in rows}
        pairgroups=defaultdict(list)
        for r in rows:
            c=cases[r['case_id']]
            if c.get('semantic_pair_id'):pairgroups[c['semantic_pair_id']].append(c)
        pairrows=[]
        for pid,group in pairgroups.items():
            anchor=next(c for c in group if c['language_variant']=='ko_anchor');ar=byid[anchor['case_id']]
            gold={i for g in anchor['fact_groups'] for i in g['primary_ids']+g['acceptable_ids']}
            def first_rank(rr):return next((i for i,f in enumerate(rr['top_ids'],1) if f in gold),len(rr['top_ids'])+1)
            for c in group:
                if c['language_variant']=='ko_anchor':continue
                rr=byid[c['case_id']]
                pairrows.append({'pair_id':pid,'variant':c['language_variant'],'anchor_case_id':anchor['case_id'],'variant_case_id':c['case_id'],
                                'anchor_first_fact_rank':first_rank(ar),'variant_first_fact_rank':first_rank(rr),
                                'rank_delta':first_rank(rr)-first_rank(ar),'rank_censor':len(rr['top_ids'])+1,
                                'top10_overlap':len(set(ar['top_ids'][:10])&set(rr['top_ids'][:10]))/10,
                                'anchor_hit3':ar['metrics']['3']['semantic_hit'],'variant_hit3':rr['metrics']['3']['semantic_hit'],
                                'variant_fact_recall3':rr['metrics']['3']['fact_recall'],'variant_all_facts_hit3':rr['metrics']['3']['all_facts_hit']})
        lang={}
        for variant in sorted({r['variant'] for r in pairrows}):
            rr=[r for r in pairrows if r['variant']==variant]
            lang[variant]={'pairs':len(rr),'mean_rank_delta_censored':float(np.mean([r['rank_delta'] for r in rr])),
                           'mean_top10_overlap':float(np.mean([r['top10_overlap'] for r in rr])),
                           'variant_hit3':float(np.mean([r['variant_hit3'] for r in rr])),
                           'anchor_hit3':float(np.mean([r['anchor_hit3'] for r in rr])),
                           'ko_hit_en_miss3':sum(r['anchor_hit3'] and not r['variant_hit3'] for r in rr),
                           'ko_miss_en_hit3':sum(not r['anchor_hit3'] and r['variant_hit3'] for r in rr)}
        key=stage if split=='calibration' else 'holdout-'+stage
        save_lines(OUT/f'analysis/language/{key}-pairs.jsonl',pairrows);comparisons[key]=lang
    save(OUT/'analysis/language_summary.json',comparisons)
    reranker_audit=[]
    for rr in ['bge','jina']:
        for view in ['question','question_answer']:
            for budget in [10,20]:
                name=f'D-{rr}-{view}-M{budget}'
                before={r['case_id']:r for r in lines(OUT/f'calibration/C-M{budget}/per_query.jsonl')}
                after=lines(OUT/f'calibration/{name}/per_query.jsonl')
                for r in after:
                    c=cases[r['case_id']];br=before[r['case_id']]
                    if not c['fact_groups']:continue
                    gold={i for g in c['fact_groups'] for i in g['primary_ids']+g['acceptable_ids']}
                    rb=next((i for i,f in enumerate(br['top_ids'],1) if f in gold),None)
                    ra=next((i for i,f in enumerate(r['top_ids'],1) if f in gold),None)
                    miss=[g['fact_id'] for g in c['fact_groups'] if not(set(g['primary_ids']+g['acceptable_ids'])&set(br['top_ids']))]
                    status='STAGE1_MISS' if miss else 'IMPROVED' if ra<rb else 'WORSENED' if ra>rb else 'UNCHANGED'
                    reranker_audit.append({'experiment':name,'case_id':c['case_id'],'baseline_rank':rb,'reranked_rank':ra,
                                          'rank_delta':ra-rb if ra is not None and rb is not None else None,
                                          'status':status,'stage1_missing_facts':miss,
                                          'all_facts_before3':br['metrics']['3']['all_facts_hit'],
                                          'all_facts_after3':r['metrics']['3']['all_facts_hit']})
    save_lines(OUT/'analysis/reranker_query_deltas.jsonl',reranker_audit)
    save(OUT/'analysis/reranker_delta_summary.json',
         {name:dict(Counter(r['status'] for r in reranker_audit if r['experiment']==name)) for name in sorted({r['experiment'] for r in reranker_audit})})
    rows=lines(OUT/'calibration'/chosen/'per_query.jsonl');failures=[];types=defaultdict(list)
    for r in rows:
        c=cases[r['case_id']]
        if c['track']!='core':continue
        types[c['primary_type']].append(r)
        missing=[g['fact_id'] for g in c['fact_groups'] if not (set(g['primary_ids']+g['acceptable_ids'])&set(r['top_ids'][:20]))]
        negs=set(c['hard_negative_ids'])
        if not r['metrics']['3']['all_facts_hit']:
            failures.append({'case_id':c['case_id'],'query':c['query'],'type':c['primary_type'],'split':'calibration',
                             'failure_class':'RETRIEVAL_MISS' if missing else 'RANKING_MISS',
                             'missing_candidate_facts':missing,'hard_negative_top1':r['top_ids'][0] in negs,
                             'actual_top3':r['top_ids'][:3],'expected_fact_groups':c['fact_groups'],'label_review':'pending'})
    summary={t:{'rows':len(rr),'hit1_row_mean':float(np.mean([r['metrics']['1']['semantic_hit'] for r in rr])),
                'hit3_row_mean':float(np.mean([r['metrics']['3']['semantic_hit'] for r in rr])),
                'fact_recall3_row_mean':float(np.mean([r['metrics']['3']['fact_recall'] for r in rr])),
                'all_facts3_row_mean':float(np.mean([r['metrics']['3']['all_facts_hit'] for r in rr]))} for t,rr in types.items()}
    save(OUT/'analysis/type_summary.json',summary);save_lines(OUT/'analysis/failure_examples.jsonl',failures)
    save(OUT/'analysis/failure_summary.json',{'selected':chosen,'failures_at_3':len(failures),'classes':dict(Counter(r['failure_class'] for r in failures)),
                                           'labels_changed_from_failures':False,'bm25_trigger_review':'pending independent lexical classification'})
    print(json.dumps({'language':comparisons[chosen],'failures_at_3':len(failures)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
