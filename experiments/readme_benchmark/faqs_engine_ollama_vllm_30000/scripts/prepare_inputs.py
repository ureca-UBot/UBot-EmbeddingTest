"""Adapt the new FAQ dataset only; no previous test rows or results are loaded."""
from collections import Counter
from pathlib import Path
from common import ROOT,DATA,N,MAIN_N,REFERENCE_N,lines,read,save,save_lines,sha

source=ROOT.parent/'faqs_based_30000'
original=lines(source/'data/variations.jsonl')
inputs=lines(DATA/'model_inputs.jsonl')
assert len(original)==len(inputs)==MAIN_N
assert all(c['case_id']==q['case_id'] and c['query']==q['query'] for c,q in zip(original,inputs))
main=[]
for i,c in enumerate(original):
    main.append({
        'case_id':c['case_id'],'parent_case_id':c['parent_case_id'],'source_row':i,
        'type':c['primary_variation'],'cohort':'multi_intent' if c['variant_slot']=='V10' else 'condition' if c['evaluation_family']=='condition' else 'general',
        'primary_variation':c['primary_variation'],'validation_tags':c['validation_tags'],
        'variant_slot':c['variant_slot'],'evaluation_role':'main','brand':c['brand'],'category':c['category'],
        'source_language':c['source_language'],'query':c['query'],'model_query':c['query'],
        'source_ids':c['source_faq_ids'],'no_faq_truth':False,'family_id':c['split_group_id'],
        'language_pair_reference':c['language_pair_reference'],'gt_granularity':c['gt_granularity'],
        'review_status':c['review_status']
    })
parents={c['parent_case_id']:c for c in main if c['variant_slot']=='V01'}
refs=[]
for i,b in enumerate(lines(DATA/'paired_baselines.jsonl')):
    parent=parents[b['parent_case_id']]
    refs.append({**parent,'case_id':b['case_id'],'source_row':MAIN_N+i,'type':'PAIRED_REFERENCE',
        'cohort':'paired_reference','primary_variation':'PAIRED_REFERENCE','validation_tags':[],
        'variant_slot':'BASE','evaluation_role':'reference','query':b['query'],'model_query':b['query'],
        'language_pair_reference':None,'source_ids':[b['faq_id']]})
assert len(refs)==REFERENCE_N and len({c['case_id'] for c in main+refs})==N
save_lines(DATA/'cases.jsonl',main+refs)
filenames=['cases.jsonl','corpus.jsonl','corpus_views.jsonl','reranker_chunks.jsonl','paired_baselines.jsonl','source_pairs.jsonl','model_inputs.jsonl','purpose_catalog.json']
manifest={'source_dataset':'faqs_based_30000','source_variants_sha256':sha(source/'data/variations.jsonl'),
    'source_raw_faq_sha256':read(source/'data/manifest.json')['source_sha256'],
    'main_rows':MAIN_N,'paired_reference_rows':REFERENCE_N,'inference_rows':N,'corpus_faqs':3246,
    'engines':['ollama','vllm'],'type_counts':dict(Counter(c['type'] for c in main)),
    'cohort_counts':dict(Counter(c['cohort'] for c in main)),
    'hashes':{name:sha(DATA/name) for name in filenames},
    'protocol':{'dense_input_limit':8192,'reranker_input_limit':1024,
        'reranker_qa_input':'all source-preserving answer windows per candidate FAQ',
        'reranker_qa_merge':'maximum raw logit across a FAQ candidate answer windows',
        'candidate_faq_budget':20,'dense_precision':{'ollama':'GGUF F16','vllm':'HF float32'},
        'reranker_precision':'common CUDA float16; FP32/FP16 probe recorded',
        'native_precision':'common FlagEmbedding CUDA float32',
        'no_faq_threshold_fit':False,'query_truncation':False,'holdout':False,
        'selection':'general parent AllSources@3; ties by multi-intent parent AllSources@3, condition parent AllSources@3, general parent Hit@1'},
    'old_test_data_imported':False,'other_engine_inference_reused':False,
    'independent_annotation_complete':False,'repeat_cases':0}
save(DATA/'manifest.json',manifest)
print(manifest['cohort_counts'],flush=True)
