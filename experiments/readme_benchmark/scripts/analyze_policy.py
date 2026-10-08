"""Report the frozen rejection policy's effect; never tune or rerun."""
from collections import defaultdict
import numpy as np
from run_retrieval_benchmark import OUT,DATA,lines,save
cases={c['case_id']:c for c in lines(DATA/'cases.jsonl')}
rows=lines(OUT/'holdout/per_query.jsonl');result={}
for cohort in ['core_general','targeted_condition']:
    rr=[r for r in rows if r['cohort']==cohort];groups=defaultdict(list)
    for r in rr:groups[r['family_id']].append(r)
    result[cohort]={'rows':len(rr),'raw_hit3':float(np.mean([np.mean([r['metrics']['3']['semantic_hit'] for r in g]) for g in groups.values()])),
                    'accepted_hit3':float(np.mean([np.mean([r['metrics']['3']['semantic_hit'] and not r['predicted_no_faq'] for r in g]) for g in groups.values()])),
                    'raw_all_facts3':float(np.mean([np.mean([r['metrics']['3']['all_facts_hit'] for r in g]) for g in groups.values()])),
                    'accepted_all_facts3':float(np.mean([np.mean([r['metrics']['3']['all_facts_hit'] and not r['predicted_no_faq'] for r in g]) for g in groups.values()])),
                    'answerable_rejected_rows':sum(r['predicted_no_faq'] for r in rr)}
result['unsupported_accepted_cases']=[{'case_id':r['case_id'],'query':cases[r['case_id']]['query'],'top3':r['top_ids'][:3]} for r in rows if r['no_faq_truth'] and not r['predicted_no_faq']]
result['threshold_changed_after_holdout']=False
save(OUT/'analysis/policy_effects.json',result);print(result)
