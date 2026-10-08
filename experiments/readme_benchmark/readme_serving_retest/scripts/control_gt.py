"""Evidence-backed control-label corrections; no query, corpus or model-score change."""
import copy
import os
import numpy as np
from common import DATA, OUT, read, lines, save, save_lines, doc_text, rankings, pools, summarize, export_log, sha

def revised_cases(cases,docs,write_outputs=True):
    revised=copy.deepcopy(cases);byid={d['faq_id']:d for d in docs}
    edits=[]
    def fact(name,statement,ids,quotes):
        evidence=[]
        for fid,quote in zip(ids,quotes):
            d=byid[fid];assert quote in d['answer'],(fid,quote)
            evidence.append({'faq_id':fid,'quote':quote,'source_sheet':d['source_sheet'],
                             'source_cell':f"D{d['source_row']}"})
        return {'fact_id':name,'statement':statement,'acceptable_faq_ids':ids,'evidence':evidence,
                'independent_review_status':'source-checked label correction by assistant; independent review pending'}
    premium=fact('PREMIUM_BASE_MONTHLY_PRICE','5G 프리미엄 기본 월 요금은 95,000원이다.',
        ['FAQ-102','FAQ-153','FAQ-164','FAQ-175'],
        ['5G 프리미엄은 월 95,000원','5G 프리미엄 월 95,000원','프리미엄 95,000원','프리미엄(95,000원)'])
    light=fact('LIGHT_AFTER_BASIC_SPEED','5G 라이트의 기본 데이터 소진 후 속도는 1Mbps이다.',
        ['FAQ-104','FAQ-160','FAQ-358','FAQ-359'],
        ['소진 후 1Mbps','5G 라이트는 30GB 소진 후 1Mbps','5G 라이트는 1Mbps','5G 라이트와 청소년 5G는 1Mbps'])
    lte=fact('LTE_AFTER_BASIC_SPEED','LTE 무제한의 기본 데이터 소진 후 속도는 5Mbps이다.',
        ['FAQ-106','FAQ-160','FAQ-359'],
        ['이후에는 5Mbps','LTE 무제한은 100GB 소진 후 5Mbps','5G 스탠다드와 LTE 무제한은 5Mbps'])
    for case in revised:
        case['gt_annotation_version']='INITIAL_SOURCE_GT_V1'
        replacement=([premium] if case['case_id'] in ['PUR-ENTITY-001','PUR-CONTEXT-001'] else
                     [light,lte] if case['case_id']=='PUR-COMPARE-001' else None)
        if replacement:
            edits.append({'case_id':case['case_id'],'before_required_facts':case['required_facts'],
                          'after_required_facts':replacement,'query_changed':False,
                          'reason':'Accept all explicit source-supported required facts; do not demand irrelevant 30/100GB boundaries in a post-basic-data speed comparison.'})
            case['required_facts']=copy.deepcopy(replacement)
            case['gt_annotation_version']='CONTROL_FACT_REVIEW_V2'
    report={'version':'CONTROL_FACT_REVIEW_V2','corrections':edits,'independent_review_complete':False,
            'review_scope':'three purpose-control queries only; does not validate every synthetic query or all possible alternative FAQs',
            'model_queries_changed':0,'corpus_changed':False,'model_scores_changed':False,
            'timing':'label correction after observing diagnostics, before final aggregation; no holdout evaluated'}
    assert all(old['model_query']==new['model_query'] for old,new in zip(cases,revised))
    if write_outputs:
        save(DATA/'control_gt_corrections.json',report)
        save_lines(DATA/'evaluation_cases.jsonl',revised)
    else:
        assert read(DATA/'control_gt_corrections.json')==report
        assert lines(DATA/'evaluation_cases.jsonl')==revised
    return revised

def reaggregate(cases,docs,structures):
    for engine in ['ollama','vllm']:
        base=OUT/'engines'/engine
        a=np.load(base/'cache/A-scores.npy');b=np.load(base/'cache/B-scores.npy')
        candidate_sets={k:pools(a,b,k) for k in [1,3,5,10,20]}
        candidates=candidate_sets[20]
        for name in structures:
            folder=base/name
            previous=read(folder/'summary.json')
            # Keep the initial annotations/results; GPU inference is never repeated for a label correction.
            for file in ['summary.json','per_query.jsonl','candidate_log.jsonl','candidate_log_schema.json','rank_changes.jsonl']:
                original=folder/file
                archive=folder/(original.stem+'.initial_gt'+original.suffix)
                if original.exists() and not archive.exists(): os.replace(original,archive)
            if name=='A': scores=a;rr=rankings(a)
            elif name=='B': scores=b;rr=rankings(b)
            elif name.startswith('C-'): scores=np.maximum(a,b);rr=candidates
            else:
                scores=np.load(base/'cache'/f'{name}-scores.npy',mmap_mode='r')
                rr=[sorted(ids,key=lambda j:(-float(scores[i,j]),j)) for i,ids in enumerate(candidates)]
            execution={**previous['execution'],'control_gt_annotation_version':'CONTROL_FACT_REVIEW_V2',
                       'model_inference_repeated_for_gt_fix':False,'control_gt_corrections_sha256':sha(DATA/'control_gt_corrections.json')}
            summarize(engine,name,cases,docs,rr,scores,execution,candidate_sets if name.startswith('C-') else None)
            if name.startswith('D-'):
                export_log(engine,name,'question_answer' if name.endswith('question_answer') else 'question',cases,docs,candidates,scores,execution)
