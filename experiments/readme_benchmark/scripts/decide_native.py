"""Conditionally execute native BGE heads after A-D calibration only."""
from run_retrieval_benchmark import OUT,jsread,save
names=['A','B','C-M20','D-bge-question-M20','D-bge-question_answer-M20','D-jina-question-M20','D-jina-question_answer-M20']
summaries=[jsread(OUT/'calibration'/n/'summary.json') for n in names]
best_hit=max(s['cohorts']['core_general']['family_mean']['semantic_hit@1'] for s in summaries)
best_recall=max(s['cohorts']['targeted_condition']['family_mean']['fact_recall@10'] for s in summaries)
result={'execute_sparse_and_multivector':best_hit<.95 or best_recall<.99,
        'reason':'A-D still below README initial Hit@1 95% or targeted FactRecall@10 99%; provisional labels',
        'best_core_hit_at_1':best_hit,'best_targeted_fact_recall_at_10':best_recall,
        'holdout_inspected':False,'bm25_enabled':False,'bm25_reason':'No independently reviewed lexical-miss audit; do not enable automatically'}
save(OUT/'native_decision.json',result);print(result)
