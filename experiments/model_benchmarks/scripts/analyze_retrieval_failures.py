"""Read-only calibration analysis; hypotheses are not domain-reviewed labels."""
from collections import Counter, defaultdict
import numpy as np
from run_retrieval_benchmark import OUT, DATA, lines, save, save_lines, jsread

def mean_by_family(rows, key, k):
    values=defaultdict(list)
    for r in rows:
        value=r['metrics'][str(k)][key]
        if value is not None:values[r['family_id']].append(value)
    return float(np.mean([np.mean(v) for v in values.values()])) if values else None

def main():
    cases={c['case_id']:c for c in lines(DATA/'cases.jsonl') if c['split']=='calibration'}
    baseline={r['case_id']:r for r in lines(OUT/'calibration/A/per_query.jsonl')}
    summaries=[];failures=[];pool_cache={}
    for path in sorted((OUT/'calibration').glob('*/per_query.jsonl')):
        stage=path.parent.name;rows=lines(path)
        slices={}
        for label,predicate in [('singleton',lambda c:len(c['fact_groups'])==1),
                                ('multiple_facts',lambda c:len(c['fact_groups'])>1)]:
            core=[r for r in rows if r['cohort']=='core_general' and predicate(cases[r['case_id']])]
            slices[label]={'rows':len(core),'families':len(set(r['family_id'] for r in core)),
                           'hit_at_1':mean_by_family(core,'semantic_hit',1),
                           'hit_at_3':mean_by_family(core,'semantic_hit',3),
                           'all_facts_at_3':mean_by_family(core,'all_facts_hit',3),
                           'fact_recall_at_10':mean_by_family(core,'fact_recall',10)}
        counts=Counter()
        for r in rows:
            c=cases[r['case_id']]
            if r['metrics']['3']['all_facts_hit'] is None:continue
            if r['cohort'] not in ['core_general','targeted_condition','contextual']:continue
            if r['metrics']['3']['all_facts_hit']:continue
            missing=[g for g in c['fact_groups'] if not(set(g['primary_ids']+g['acceptable_ids'])&set(r['top_ids'][:20]))]
            stage1_missing=missing
            pool_name=None
            if stage.startswith('D-Acontrol'):pool_name='A'
            elif stage.startswith('D-'):pool_name='C-M'+stage.rsplit('M',1)[1]
            if pool_name:
                if pool_name not in pool_cache:
                    pool_cache[pool_name]={p['case_id']:p for p in lines(OUT/f'calibration/{pool_name}/per_query.jsonl')}
                pool_rows=pool_cache[pool_name]
                m=int(stage.rsplit('M',1)[1]);pool_ids=pool_rows[r['case_id']]['top_ids'][:m]
                stage1_missing=[g for g in c['fact_groups'] if not(set(g['primary_ids']+g['acceptable_ids'])&set(pool_ids))]
            label='RETRIEVAL_MISS' if stage1_missing else 'RANKING_ERROR'
            old=baseline[r['case_id']]
            if stage.startswith('D-') and old['metrics']['3']['all_facts_hit']:label='RERANKER_REGRESSION'
            lexical_hypothesis=('EXACT_ENTITY' in c['tags'] and bool(stage1_missing))
            def first_rank(row):
                acceptable={i for g in c['fact_groups'] for i in g['primary_ids']+g['acceptable_ids']}
                return next((i for i,fid in enumerate(row['top_ids'],1) if fid in acceptable),None)
            old_rank=first_rank(old);new_rank=first_rank(r)
            failures.append({'stage':stage,'case_id':c['case_id'],'query':r['query'],
                             'cohort':r['cohort'],'tags':c['tags'],'classification':label,
                             'lexical_miss_hypothesis_requires_review':lexical_hypothesis,
                             'missing_candidate_facts':[g['fact_id'] for g in stage1_missing],
                             'baseline_rank':old_rank,'new_rank':new_rank,
                             'rank_delta':None if old_rank is None or new_rank is None else old_rank-new_rank,
                             'top_ids':r['top_ids'][:20],'fact_groups':c['fact_groups'],
                             'annotation_status':'pending_independent_domain_review'})
            counts[label]+=1
        summaries.append({'stage':stage,'slices':slices,'failure_rows_by_hypothesis':dict(counts)})
    save_lines(OUT/'diagnostics/calibration-failures.jsonl',failures)
    save(OUT/'diagnostics/calibration-slices.json',summaries)
    save(OUT/'diagnostics/hybrid-decision.json',{
        'status':'DEFERRED_DOMAIN_REVIEW_REQUIRED','BM25_executed':False,
        'reason':'Native dense+sparse RRF was measured. Remaining lexical-miss flags use pending source tags and do not establish reviewed exact-entity failures; do not add another retrieval architecture from those flags.',
        'scope':'calibration_only','review_file':'calibration-failures.jsonl'})
    print('Calibration failure and fact-count slices saved',flush=True)

if __name__=='__main__':main()
