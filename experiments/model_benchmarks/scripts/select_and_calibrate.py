"""Select and freeze policies from calibration; never read holdout results."""
import json
from collections import defaultdict
import numpy as np
from run_retrieval_benchmark import OUT,DATA,lines,save,jsread,sha
from retrieval_common import fact_metrics,no_faq_metrics

def family_mean(rows,values):
    grouped=defaultdict(list)
    for r,v in zip(rows,values):grouped[r['family_id']].append(v)
    return float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None

def score_margin_diagnostics(name):
    rows=lines(OUT/f'calibration/{name}/per_query.jsonl')
    groups=defaultdict(list)
    for r in rows:
        if len(r['top_scores'])<2:continue
        label=('NO_FAQ' if r['answerability']=='none' else
               'TOP3_ALL_FACTS' if r['metrics']['3']['all_facts_hit'] else 'TOP3_MISSING_FACTS')
        groups[r['cohort']+'/'+label].append((r['top_scores'][0],r['top_scores'][0]-r['top_scores'][1]))
    result={}
    for label,values in groups.items():
        arr=np.asarray(values)
        result[label]={'rows':len(values),'top1_score_quantiles':np.quantile(arr[:,0],[0,.1,.5,.9,1]).tolist(),
                       'top1_minus_top2_quantiles':np.quantile(arr[:,1],[0,.1,.5,.9,1]).tolist()}
    save(OUT/f'calibration/{name}/score_margin_diagnostics.json',
         {'split':'calibration_only','quantiles':[0,.1,.5,.9,1],'groups':result,
          'margin_policy':'diagnostic_only','GT_is_not_runtime_input':True})

def calibrate(name,cases):
    rows=lines(OUT/f'calibration/{name}/per_query.jsonl');cm={c['case_id']:c for c in cases}
    binary=[r for r in rows if r['answerability'] in ['full','none'] and r['cohort'] in ['core_general','targeted_condition','no_faq']]
    thresholds=sorted({s for r in binary for s in r['top_scores'][:3]})
    thresholds=[np.nextafter(min(thresholds),-np.inf)]+thresholds+[np.nextafter(max(thresholds),np.inf)]
    best=None;curve=[]
    for t in thresholds:
        labels=[r['answerability'] for r in binary];accepted=[bool(r['top_scores'] and r['top_scores'][0]>=t) for r in binary]
        detection=no_faq_metrics(labels,accepted)
        if detection['unsafe_accept_rate'] is None:continue
        positives=[r for r in binary if r['answerability']=='full' and r['cohort']=='core_general']
        metric=family_mean(positives,[fact_metrics([fid for fid,s in zip(r['top_ids'][:3],r['top_scores'][:3]) if s>=t],cm[r['case_id']]['fact_groups'],3)['all_facts_hit'] for r in positives])
        curve.append({'threshold':float(t),'all_facts_at_3':metric,**detection})
        if detection['unsafe_accept_rate']<=0.01 and (best is None or metric>best['all_facts_at_3']):best=curve[-1]
    save(OUT/f'calibration/{name}/threshold_curve.json',curve)
    return best

def main():
    if (OUT/'selection.json').exists():raise SystemExit('Selection already frozen; do not retune.')
    candidates=[]
    for path in (OUT/'calibration').glob('*/summary.json'):
        summary=jsread(path);core=summary['cohorts'].get('core_general')
        if core and core['families']:candidates.append((summary['experiment'],core['family_mean']))
    baseline=dict(candidates)['A']
    ranked=sorted(candidates,key=lambda pair:(-pair[1]['all_facts_hit@3'],-pair[1]['semantic_hit@1'],pair[0]))
    challenger=next(name for name,metrics in ranked if name!='A')
    chosen=ranked[0][0]
    rows={name:{r['case_id']:r for r in lines(OUT/f'calibration/{name}/per_query.jsonl')} for name in ['A',challenger]}
    case_lookup={c['case_id']:c for c in lines(DATA/'cases.jsonl')}
    deltas=defaultdict(lambda:defaultdict(list))
    for cid,r in rows['A'].items():
        if r['cohort']=='core_general' and r['metrics']['3']['all_facts_hit'] is not None:
            deltas[case_lookup[cid]['split_group_id']][r['family_id']].append(rows[challenger][cid]['metrics']['3']['all_facts_hit']-r['metrics']['3']['all_facts_hit'])
    family_values=[[np.mean(v) for v in grouped.values()] for grouped in deltas.values()]
    sums=np.array([sum(v) for v in family_values]);weights=np.array([len(v) for v in family_values]);rng=np.random.default_rng(20261006)
    samples=rng.integers(0,len(sums),size=(2000,len(sums)))
    bootstrap=sums[samples].sum(axis=1)/weights[samples].sum(axis=1)
    lower,upper=np.quantile(bootstrap,[0.025,0.975]).tolist()
    cases=lines(DATA/'cases.jsonl');cal=[c for c in cases if c['split']=='calibration']
    for name in ['A',challenger]:score_margin_diagnostics(name)
    policies={name:calibrate(name,cal) for name in ['A',challenger]}
    qualifying=[name for name,metric in ranked if metric['semantic_hit@1']>=baseline['semantic_hit@1']-0.01]
    selected=qualifying[0] if qualifying else 'A'
    save(OUT/'selection.json',{'status':'frozen_provisional_source_label_experiment','dataset_manifest_hash':sha(DATA/'split_manifest.json'),'candidate_results':[{'name':name,**metric} for name,metric in ranked],'best_quality_score':chosen,'selected_with_top1_gate':selected,'holdout_challenger':challenger,'paired_bootstrap_all_facts_at_3':{'baseline':'A','challenger':challenger,'families':int(weights.sum()),'resampling_unit':'connected_source_component','components':len(weights),'mean_delta':float(sums.sum()/weights.sum()),'ci95':[lower,upper],'superiority_supported':lower>0},'policies':policies,'baseline_P0':0.75,'margin_policy':'diagnostic_only_no_GT_runtime_gate','runtime_serving_common_lane':'dense_question_component','independent_domain_review_complete':False})
    print((OUT/'selection.json').read_text(encoding='utf-8'),flush=True)

if __name__=='__main__':main()
