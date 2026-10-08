"""Apply frozen calibration policies once; preserve every held-out row."""
import numpy as np
from collections import defaultdict
from run_retrieval_benchmark import OUT,DATA,lines,save,jsread,sha
from retrieval_common import fact_metrics,no_faq_metrics

def main():
    output=OUT/'holdout/policy_results.json'
    if output.exists():raise SystemExit('Holdout policy evaluation already exists; do not retune.')
    frozen=jsread(OUT/'selection.json');cases={c['case_id']:c for c in lines(DATA/'cases.jsonl') if c['split']=='holdout'}
    results=[]
    policies=[('A','P0',0.75),('A','calibrated',frozen['policies']['A']['threshold'])]
    name=frozen['holdout_challenger'];policies.append((name,'calibrated',frozen['policies'][name]['threshold']))
    for name,policy,t in policies:
        rows=lines(OUT/f'holdout/{name}/per_query.jsonl');groups=defaultdict(list);binary_labels=[];decisions=[];per_case=[]
        for r in rows:
            c=cases[r['case_id']];accepted=[fid for fid,score in zip(r['top_ids'][:3],r['top_scores'][:3]) if score>=t]
            metric=fact_metrics(accepted,c['fact_groups'],3)
            if r['cohort']=='core_general' and metric['all_facts_hit'] is not None:groups[r['family_id']].append(metric['all_facts_hit'])
            if c['answerability'] in ['full','none'] and r['cohort'] in ['core_general','targeted_condition','no_faq']:binary_labels.append(c['answerability']);decisions.append(bool(accepted))
            per_case.append({'case_id':r['case_id'],'accepted_ids':accepted,'metrics':metric})
        result={'experiment':name,'policy':policy,'threshold':t,'family_mean_core_AllFactsHit_at_3':float(np.mean([np.mean(v) for v in groups.values()])),'no_faq':no_faq_metrics(binary_labels,decisions),'case_count':len(rows),'label_status':'provisional_source_labels'}
        results.append(result);save(OUT/f'holdout/{name}/{policy}-per-query.json',per_case)
    save(output,{'selection_hash':sha(OUT/'selection.json'),'dataset_manifest_hash':sha(DATA/'split_manifest.json'),'results':results,'retuning_after_holdout':False,'domain_review_complete':False})
    print(output.read_text(encoding='utf-8'),flush=True)

if __name__=='__main__':main()
