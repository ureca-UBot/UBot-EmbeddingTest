"""Select architecture on the new calibration split, then calibrate rejection."""
from __future__ import annotations
import json, hashlib
from collections import defaultdict
import numpy as np
from run_retrieval_benchmark import OUT,DATA,lines,jsread,save,sha
from retrieval_common import no_faq_metrics

def main():
    if (OUT/'selection.json').exists():raise SystemExit('Selection already frozen; do not inspect holdout and tune again')
    allowed=['A','B','C-M20','D-bge-question-M20','D-bge-question_answer-M20','D-jina-question-M20','D-jina-question_answer-M20',
             'E-dense-sparse-RRF-M20','F-dense-pool-M20']
    candidates=[]
    for name in allowed:
        path=OUT/'calibration'/name/'summary.json'
        if not path.exists():continue
        s=jsread(path);g=s['cohorts']['core_general']['family_mean'];t=s['cohorts']['targeted_condition']['family_mean']
        candidates.append({'name':name,'core_all_facts_3':g['all_facts_hit@3'],'core_hit_1':g['semantic_hit@1'],
                           'target_all_facts_3':t['all_facts_hit@3'],'core_fact_recall_20':g['fact_recall@20']})
    if len(candidates)<7:raise ValueError('Complete A/B/C and all four D cells before selecting')
    candidates.sort(key=lambda x:(-x['core_all_facts_3'],-x['target_all_facts_3'],-x['core_hit_1'],x['name']))
    selected=candidates[0]
    cases={c['case_id']:c for c in lines(DATA/'cases.jsonl') if c['split']=='calibration' and c['track'] in ['core','no_faq']}
    rows=lines(OUT/'calibration'/selected['name']/'per_query.jsonl')
    score=np.array([r['top_scores'][0] for r in rows],dtype=float)
    margin=np.array([r['top_scores'][0]-r['top_scores'][1] for r in rows],dtype=float)
    truth=np.array([cases[r['case_id']]['track']=='no_faq' for r in rows])
    thresholds=np.unique(np.r_[np.min(score)-1,np.quantile(score,np.linspace(0,1,61)),np.max(score)+1])
    margins=np.unique(np.r_[0,np.quantile(margin,np.linspace(0,1,31)),np.max(margin)+1])
    tested=[]
    for threshold in thresholds:
        for mm in margins:
            reject=(score<threshold)|(margin<mm)
            tp=int(np.sum(reject&truth));fp=int(np.sum(reject&~truth));fn=int(np.sum(~reject&truth));tn=int(np.sum(~reject&~truth))
            precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
            f1=2*precision*recall/(precision+recall) if precision+recall else 0
            accept=tn/(tn+fp) if tn+fp else 0
            tested.append({'threshold':float(threshold),'margin':float(mm),'no_faq_precision':precision,
                           'no_faq_recall':recall,'no_faq_f1':f1,'answerable_acceptance':accept,
                           'tp':tp,'fp':fp,'fn':fn,'tn':tn})
    # State the tradeoff instead of inventing a product-approved cutoff.
    tested.sort(key=lambda x:(-x['no_faq_f1'],-x['answerable_acceptance'],x['threshold'],x['margin']))
    policy=tested[0]
    save(OUT/'calibration/threshold_grid.json',tested)
    value={'status':'FROZEN_PROVISIONAL','dataset_version':'readme-authored-v1',
           'dataset_manifest_sha256':sha(DATA/'split_manifest.json'),'selected':selected['name'],'candidate_comparison':candidates,
           'selection_rule':'calibration core_general family AllFactsHit@3, then targeted family AllFactsHit@3, then core Hit@1; unique candidate budget 20',
           'full_corpus_F_is_diagnostic_only':True,
           'threshold_rule':'maximize calibration NO_FAQ F1; tie by answerable acceptance; raw identity score and top1-top2 margin',
           'policy':policy,'calibration_rows':len(rows),'calibration_no_faq_rows':int(truth.sum()),
           'holdout_consulted':False,'independent_annotation_review_complete':False,'product_acceptance':False,
           'legacy_results_consulted':False}
    save(OUT/'selection.json',value);print(json.dumps(value,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
