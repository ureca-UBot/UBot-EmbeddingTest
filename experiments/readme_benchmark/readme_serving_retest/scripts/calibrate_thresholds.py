"""Exact score+margin gate fitting on frozen retrieval results, with grouped OOF diagnostics."""
import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import numpy as np
from common import DATA, OUT, ROOT, KS, inputs, read, lines, save, sha, expected_ids
from control_gt import revised_cases

TARGET = OUT / 'threshold_calibration_v1'
OBJECTIVE = 'macro_f1'
SEED = 20261007

def now(): return datetime.now(timezone.utc).isoformat()

def write_lines(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8',buffering=8*1024*1024) as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(temporary,path)

def accept(score, margin, score_threshold=None, margin_threshold=None):
    score=np.asarray(score,dtype=np.float64);margin=np.asarray(margin,dtype=np.float64)
    # float64 is essential: rounding nextafter thresholds back to float32 changes >= at ties.
    result=np.ones(len(score),dtype=bool)
    if score_threshold is not None:result &= score>=score_threshold
    if margin_threshold is not None:result &= margin>=margin_threshold
    return result

def classification(faq_exists,accepted):
    faq_exists=np.asarray(faq_exists,dtype=bool);accepted=np.asarray(accepted,dtype=bool)
    # Positive class in the NO_FAQ metrics is rejection of a truly absent answer.
    tp=int(np.sum(~faq_exists & ~accepted));fn=int(np.sum(~faq_exists & accepted))
    fp=int(np.sum(faq_exists & ~accepted));tn=int(np.sum(faq_exists & accepted))
    ratio=lambda a,b:a/b if b else 0.
    no_f1=Fraction(2*tp,2*tp+fp+fn) if 2*tp+fp+fn else Fraction(0)
    faq_f1=Fraction(2*tn,2*tn+fp+fn) if 2*tn+fp+fn else Fraction(0)
    return {'faq_rows':tn+fp,'no_faq_rows':tp+fn,'no_faq_tp':tp,'no_faq_fn':fn,
            'faq_false_rejections':fp,'faq_accepted':tn,'macro_f1':float((no_f1+faq_f1)/2),
            'no_faq_precision':ratio(tp,tp+fp),'no_faq_recall':ratio(tp,tp+fn),
            'no_faq_f1':float(no_f1),'no_faq_false_accept_rate':ratio(fn,tp+fn),
            'faq_precision_presence':ratio(tn,tn+fn),'faq_recall_presence':ratio(tn,tn+fp),
            'faq_f1_presence':float(faq_f1),'overall_coverage':ratio(tn+fn,tp+fn+fp+tn)}

def fit(score,margin,faq_exists,mode='joint'):
    score=np.asarray(score,dtype=np.float64);margin=np.asarray(margin,dtype=np.float64)
    label=np.asarray(faq_exists,dtype=bool)
    assert len(score)==len(margin)==len(label) and np.isfinite(score).all() and np.isfinite(margin).all()
    assert np.any(label) and np.any(~label) and np.all(margin>=0)
    # For any fixed subset of rejected negatives, a tighter cutoff cannot improve macro-F1:
    # it can only reject additional positive FAQ cases. Thus every optimum is attained
    # at disabled cutoffs or immediately above a negative score/margin boundary.
    # With two negatives the exact continuous-domain search requires at most nine gates.
    scores=[None]+[float(np.nextafter(x,np.inf)) for x in np.unique(score[~label])]
    margins=[None]+[float(np.nextafter(x,np.inf)) for x in np.unique(margin[~label])]
    if mode=='score_only':margins=[None]
    elif mode=='margin_only':scores=[None]
    elif mode!='joint':raise ValueError(mode)
    trials=[];winner=None;best=None
    for threshold in scores:
        for min_margin in margins:
            permitted=accept(score,margin,threshold,min_margin);result=classification(label,permitted)
            tp,tn,fp,fn=[result[k] for k in ['no_faq_tp','faq_accepted','faq_false_rejections','no_faq_fn']]
            no_f1=Fraction(2*tp,2*tp+fp+fn) if 2*tp+fp+fn else Fraction(0)
            faq_f1=Fraction(2*tn,2*tn+fp+fn) if 2*tn+fp+fn else Fraction(0)
            value=(no_f1+faq_f1)/2
            active=int(threshold is not None)+int(min_margin is not None)
            key=(value,tn,tp,-active,min_margin is None,
                 -threshold if threshold is not None else np.inf,-min_margin if min_margin is not None else np.inf)
            trial={'score_threshold':threshold,'margin_threshold':min_margin,'metrics':result}
            trials.append(trial)
            if best is None or key>best:best=key;winner=trial
    return {**winner,'objective':OBJECTIVE,'mode':mode,'optimality':'exact within score>=t AND margin>=m gate family',
            'candidate_gates':len(trials),'ties':'macro-F1, FAQ acceptance, NO_FAQ detection, fewer active gates, prefer score-only',
            'threshold_comparison':'>=; keep JSON float64 precision','trials':trials}

def split(cases):
    main=[i for i,c in enumerate(cases) if c['evaluation_role']=='main']
    no_faq=[i for i,c in enumerate(cases) if c['retrieval_status']=='NO_FAQ']
    assert len(main)==5000 and len(no_faq)==2
    # Group not only parent IDs, but every transitive overlapping acceptable FAQ family.
    parent={}
    def find(item):
        parent.setdefault(item,item)
        if parent[item]!=item:parent[item]=find(parent[item])
        return parent[item]
    def union(left,right):
        left,right=find(left),find(right)
        if left!=right:parent[max(left,right)]=min(left,right)
    for i in main:
        members=[cases[i]['parent_case_id']]+expected_ids(cases[i])
        for item in members:union(members[0],item)
    groups=defaultdict(list)
    for i in main:groups[find(cases[i]['parent_case_id'])].append(i)
    keys=sorted(groups);random.Random(SEED).shuffle(keys)
    keys.sort(key=lambda key:-len(groups[key]))
    fold=[[],[]];group_fold={}
    for key in keys:
        target=0 if len(fold[0])<=len(fold[1]) else 1
        fold[target].extend(groups[key]);group_fold[key]=target
    for target,i in enumerate(sorted(no_faq,key=lambda i:cases[i]['case_id'])):fold[target].append(i)
    fold=[sorted(part) for part in fold]
    gold=[{fid for i in part if cases[i]['evaluation_role']=='main' for fid in expected_ids(cases[i])} for part in fold]
    assert not gold[0]&gold[1] and not set(fold[0])&set(fold[1])
    assignments={i:f for f,part in enumerate(fold) for i in part}
    rows=[{'case_id':c['case_id'],'parent_case_id':c['parent_case_id'],'fold':assignments.get(i),
           'role':'fit_and_oof' if i in assignments else 'purpose_control_only',
           'retrieval_status':c['retrieval_status']} for i,c in enumerate(cases)]
    report={'seed':SEED,'family_groups':len(groups),'parent_faqs':len({cases[i]['parent_case_id'] for i in main}),
            'fold_sizes':list(map(len,fold)),'fold_no_faq_ids':[[cases[i]['case_id'] for i in f if i in no_faq] for f in fold],
            'overlapping_acceptable_FAQ_families':0,'main_rows':len(main),'fit_rows':len(main)+len(no_faq),
            'purpose_controls_excluded_from_fit':len(cases)-len(main)-len(no_faq),
            'retrospective':True,'untouched_holdout':False,
            'reason':'Existing retrieval architectures and scores have already been inspected; this split is only out-of-fit threshold validation.'}
    return fold,rows,report

def directories():
    result=[]
    for base in [OUT/'engines',OUT/'native_ef/engines']:
        for source in sorted(base.glob('*/*/per_query.jsonl')):
            engine=source.parent.parent.name;structure=source.parent.name
            result.append((engine,structure,source))
    assert len(result)==32
    return result

def gated_main(rows,cases,permitted):
    selected=[i for i,c in enumerate(cases) if c['evaluation_role']=='main']
    result={'rows':len(selected),'accepted':int(np.sum(permitted[selected])),
            'faq_coverage':float(np.mean(permitted[selected]))}
    for k in KS:
        for metric in ['hit','fact_recall','all_facts_hit','mrr']:
            values=np.asarray([rows[i]['metrics'][str(k)][metric] for i in selected])
            mask=permitted[selected]
            result[f'raw_{metric}@{k}']=float(values.mean())
            result[f'gated_{metric}@{k}']=float(np.mean(values*mask))
            result[f'conditional_accepted_{metric}@{k}']=float(values[mask].mean()) if np.any(mask) else None
    return result

def fit_all():
    raw,docs=inputs();cases=revised_cases(raw,docs,write_outputs=False)
    assert read(OUT/'completion_audit.json')['completed'] and read(OUT/'native_ef/completion_audit.json')['completed']
    TARGET.mkdir(parents=True,exist_ok=True)
    save(TARGET/'run_status.json',{'status':'running','timestamp':now(),'objective':OBJECTIVE})
    folds,assignments,split_report=split(cases)
    write_lines(TARGET/'split_assignments.jsonl',assignments);save(TARGET/'split_manifest.json',split_report)
    selected=np.asarray(sorted(folds[0]+folds[1]),dtype=int)
    faq=np.asarray([bool(c['required_facts']) for c in cases])
    hashes={'cases.jsonl':sha(DATA/'cases.jsonl'),'corpus.jsonl':sha(DATA/'corpus.jsonl'),
            'evaluation_cases.jsonl':sha(DATA/'evaluation_cases.jsonl')}
    save(TARGET/'experiment_manifest.json',{'objective':OBJECTIVE,'authorized_by_user':True,
        'scope':'empirical optimal fitting on the existing, provisional two NO_FAQ controls',
        'acceptance_rule':'top1_score >= score_threshold AND top1_minus_top2 >= margin_threshold',
        'score_thresholds':'raw per-structure scores; no shared cosine/logit/sparse/MaxSim/RRF threshold',
        'uncertainty':'two NO_FAQ questions; independent review and fresh holdout unavailable',
        'scores_reused_from_completed_matched_input_runs':True,'source_hashes':hashes,
        'deployment_label':'PROVISIONAL_EMPIRICAL_OPTIMUM_NOT_OPERATIONAL_VALIDATION',
        'objective_chosen_before_fit':True,'created_utc':now()})
    results=[];config=[];source_hashes=[];log_total=0
    for engine,name,source in directories():
        rows=lines(source);assert len(rows)==len(cases) and [r['case_id'] for r in rows]==[c['case_id'] for c in cases]
        assert all(r['required_facts']==c['required_facts'] for r,c in zip(rows,cases))
        score=np.asarray([r['top_scores'][0] for r in rows],dtype=np.float64)
        margin=np.asarray([r['top_scores'][0]-r['top_scores'][1] for r in rows],dtype=np.float64)
        assert np.all(margin>=0)
        oof=np.zeros(len(cases),dtype=bool);oof_details=[];oof_threshold={}
        for f in [0,1]:
            training=np.asarray(folds[1-f]);validation=np.asarray(folds[f])
            frozen=fit(score[training],margin[training],faq[training])
            valid_accept=accept(score[validation],margin[validation],frozen['score_threshold'],frozen['margin_threshold'])
            oof[validation]=valid_accept
            for i in validation:oof_threshold[int(i)]=frozen
            oof_details.append({'validation_fold':f,'fit':frozen,
                'validation_metrics':classification(faq[validation],valid_accept),
                'validation_no_faq_ids':[cases[i]['case_id'] for i in validation if not faq[i]]})
        frozen=fit(score[selected],margin[selected],faq[selected])
        fitted=accept(score,margin,frozen['score_threshold'],frozen['margin_threshold'])
        raw_accept=np.ones(len(selected),dtype=bool)
        score_type=('RRF rank fusion' if 'RRF' in name else 'reranker raw logit' if name.startswith('D-') else
                    'learned sparse weight dot product' if 'sparse-only' in name else
                    'query-token mean MaxSim' if name.startswith('F-') else
                    'max A/B dense cosine (diagnostic C order)' if name.startswith('C-') else 'dense cosine')
        result={'engine':engine,'structure':name,'score_type':score_type,
            'candidate_k_semantics':read(source.parent/'summary.json')['candidate_recall_k_semantics'],
            'fit':frozen,'score_only_fit':fit(score[selected],margin[selected],faq[selected],'score_only'),
            'margin_only_fit':fit(score[selected],margin[selected],faq[selected],'margin_only'),
            'unthresholded_metrics':classification(faq[selected],raw_accept),
            'oof_metrics':classification(faq[selected],oof[selected]),
            'oof_main':gated_main(rows,cases,oof),'refit_main':gated_main(rows,cases,fitted),
            'oof_folds':oof_details,'score_source':str(source.relative_to(ROOT)),
            'score_source_sha256':sha(source),'no_faq_control_count':2}
        results.append(result);config.append({'engine':engine,'structure':name,
            'score_threshold':frozen['score_threshold'],'margin_threshold':frozen['margin_threshold'],
            'comparison':'>=','numeric_precision':'float64; do not round to display precision',
            'fit_rows':len(selected),'no_faq_rows':2,'objective':OBJECTIVE,
            'status':'PROVISIONAL_EMPIRICAL_OPTIMUM_NOT_OPERATIONAL_VALIDATION'})
        source_hashes.append({'path':str(source.relative_to(ROOT)),'sha256':sha(source)})
        folder=TARGET/'engines'/engine/name
        save(folder/'fit.json',result)
        # C retains the complete A20+B20 candidate union; other structures preserve their existing Top20 output.
        cpool=read(OUT/'engines'/engine/'cache/C-pools.json') if name.startswith('C-') else None
        def predictions():
            for i,(case,row) in enumerate(zip(cases,rows)):
                before=row['metrics'];after={k:None if v is None else {key:value*int(fitted[i]) for key,value in v.items()} for k,v in before.items()}
                context=[docs[j]['faq_id'] for j in cpool[i]] if cpool is not None else row['top_ids']
                low_score=frozen['score_threshold'] is not None and score[i]<frozen['score_threshold']
                low_margin=frozen['margin_threshold'] is not None and margin[i]<frozen['margin_threshold']
                yield {'case_id':case['case_id'],'query':case['query'],'model_query':case['model_query'],
                    'expected_faq_id':expected_ids(case),'required_facts':case['required_facts'],
                    'retrieval_status':case['retrieval_status'],'purpose':case['primary_validation_purpose'],
                    'evaluation_role':case['evaluation_role'],'execution_group':case['execution_group'],
                    'engine':engine,'structure':name,'top1_faq_id':row['top_ids'][0],
                    'top1_score':float(score[i]),'top2_score':row['top_scores'][1],'margin':float(margin[i]),
                    'score_threshold':frozen['score_threshold'],'margin_threshold':frozen['margin_threshold'],
                    'accepted':bool(fitted[i]),'decision':'ACCEPT_CONTEXT' if fitted[i] else 'REJECT_CONTEXT',
                    'rejection_reasons':(['LOW_SCORE'] if low_score else [])+(['LOW_MARGIN'] if low_margin else []),
                    'context_faq_ids':context if fitted[i] else [],'raw_candidate_count':row['candidate_count'],
                    'metrics_before_gate':before,'metrics_after_gate':after,
                    'oof_accepted':bool(oof[i]) if i in oof_threshold else None,
                    'oof_score_threshold':oof_threshold[i]['score_threshold'] if i in oof_threshold else None,
                    'oof_margin_threshold':oof_threshold[i]['margin_threshold'] if i in oof_threshold else None,
                    'oof_fold':assignments[i]['fold'],'threshold_fit_member':i in oof_threshold,
                    'source_per_query':str(source.relative_to(ROOT)),'scores_recomputed':False,
                    'candidate_rank_and_16_field_logs':'preserved in original completed run; join by case_id',
                    'metric_scope':result['candidate_k_semantics'],'score_type':score_type}
        write_lines(folder/'per_query.jsonl',predictions());log_total+=len(cases)
        print(json.dumps({'engine':engine,'structure':name,'score_threshold':frozen['score_threshold'],
            'margin_threshold':frozen['margin_threshold'],'refit_macro_f1':frozen['metrics']['macro_f1'],
            'oof':result['oof_metrics'],'main_oof_hit1':result['oof_main']['gated_hit@1']}),flush=True)
    save(TARGET/'thresholds.json',{'objective':OBJECTIVE,'status':'provisional','structures':config})
    save(TARGET/'comparison.json',{'structures':results,'split':split_report,'source_hashes':hashes,
                                  'structures_count':len(results),'per_query_rows':log_total,'objective':OBJECTIVE})
    audit(cases,source_hashes,hashes)
    report(results,split_report)
    save(TARGET/'run_status.json',{'status':'complete','structures':len(results),'per_query_rows':log_total,'timestamp':now()})

def audit(cases,sources,hashes):
    for file,digest in hashes.items():assert sha(DATA/file)==digest
    for source in sources:assert sha(ROOT/source['path'])==source['sha256']
    total=0;checks=[]
    for path in sorted((TARGET/'engines').glob('*/*/per_query.jsonl')):
        rows=lines(path);result=read(path.parent/'fit.json');fitted=result['fit']
        assert len(rows)==len(cases)
        for row,case in zip(rows,cases):
            assert row['case_id']==case['case_id'] and row['query']==case['query'] and row['required_facts']==case['required_facts']
            assert row['accepted']==bool(accept([row['top1_score']],[row['margin']],fitted['score_threshold'],fitted['margin_threshold'])[0])
            assert (row['decision']=='ACCEPT_CONTEXT')==row['accepted']
            if not row['accepted']:assert row['context_faq_ids']==[] and row['rejection_reasons']
            for k,v in row['metrics_before_gate'].items():
                assert row['metrics_after_gate'][k]==(None if v is None else {key:value*int(row['accepted']) for key,value in v.items()})
        full_fit=[r for r in rows if r['threshold_fit_member']]
        assert classification([bool(r['required_facts']) for r in full_fit],[r['accepted'] for r in full_fit])==fitted['metrics']
        assert classification([bool(r['required_facts']) for r in full_fit],[r['oof_accepted'] for r in full_fit])==result['oof_metrics']
        total+=len(rows);checks.append({'engine':result['engine'],'structure':result['structure'],'rows':len(rows),'sha256':sha(path)})
    assert len(checks)==32 and total==32*len(cases)
    save(TARGET/'completion_audit.json',{'completed':True,'structures':len(checks),'per_query_rows':total,
        'gates_and_metrics_verified':True,'original_inputs_GT_and_source_score_logs_unchanged':True,
        'score_and_margin_fitted':True,'genuine_holdout_available':False,'no_faq_labels':2,
        'independent_GT_review_complete':False,'outputs':checks,'timestamp':now()})

def report(results,split_report):
    baseline=results[0]['unthresholded_metrics']['macro_f1']
    best=max(results,key=lambda r:r['oof_metrics']['macro_f1'])
    best_metrics=best['oof_metrics']
    text=['# Score / Margin 임계값 피팅과 재평가','',
          '사용자가 선택한 목표: FAQ 수락과 NO_FAQ 탐지의 macro-F1 최대. 동률이면 FAQ 오거절이 적은 값, 더 단순한 gate를 선택한다.',
          '수락 조건: Top1 score >= score_threshold AND Top1-Top2 >= margin_threshold. null은 해당 조건을 비활성화한다. JSON float64 값을 그대로 사용하며 표의 표시 자릿수로 반올림해 배포하지 않는다.',
          '원래 코퍼스 1,024개, 주 질문 5,000개, 통제 21개와 A~F 점수·순위를 유지한다. 검색 모델은 재실행하지 않고 Docker CPU에서 threshold 판정과 지표를 재실행했다. 질문 단위 수락 gate이며 개별 문서 점수 필터나 확률 calibration을 추가한 실험이 아니다.',
          f"FAQ 부모 {split_report['parent_faqs']}개, 겹치는 대체 정답을 합친 {split_report['family_groups']}개 family를 2-fold로 나눴다. 한 family의 변형은 같은 fold다. 각 fold NO_FAQ 1건은 그 fold를 평가할 때 피팅에서 제외한다. 전체 5,002건으로 재피팅한 최종값의 학습 성적과 OOF 성적을 구분한다.",
          'NO_FAQ는 기존 수수료 질문 2건뿐이며 라벨 독립 검수는 미완료다. 기존 검색 결과를 이미 확인한 뒤 만든 회고적 분할이므로 새로운 Holdout이나 운영용 최적값 검증이라고 부르지 않는다. 검증 목적 통제 양성 18건 및 모호한 문항 1건은 피팅에서 제외했다.',
          '정답이 존재하는 질문을 임계값 때문에 거절하면 전체 분모의 Hit/FactRecall은 0으로 계산한다. 수락한 질문만의 조건부 지표를 별도로 저장하며 전체 정확도 향상으로 포장하지 않는다. NO_FAQ/F1은 FAQ 존재 여부 gate의 지표이고, 수락한 Top1이 실제 정답인지는 검색 지표로 따로 본다.', '',
          f"전체 수락 기준선은 macro-F1 {baseline*100:.2f}%다. 가장 높은 OOF macro-F1은 {best_metrics['macro_f1']*100:.2f}%({best['structure']})이며 NO_FAQ {best_metrics['no_faq_tp']}/2건을 거절하고 FAQ {best_metrics['faq_false_rejections']}/5,000건을 오거절했다. NO_FAQ를 모두 검출한 구조는 없었다.",
          'OOF는 각 fold를 제외하고 피팅한 서로 다른 임계값으로 계산한다. 표의 최종 임계값은 OOF 계산 후 전체 5,002건에 재피팅한 값이므로 OOF에 사용한 값과 다를 수 있다. 최종 off가 OOF의 모든 판정도 수락한다는 뜻은 아니다. off는 누락이 아니라 현재 목적함수에서 해당 조건을 끄는 것이 최적이었다는 뜻이다.', '',
          '| 출처 | 구조 | 최종 score 임계값 | 최종 margin 임계값 | 재피팅 macro-F1 | OOF macro-F1 | OOF NO_FAQ 거절 | OOF FAQ 오거절 | OOF FAQ 유지율 | OOF Hit@1 |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v:'off' if v is None else f'{v:.9g}'
    for result in results:
        f=result['fit'];o=result['oof_metrics'];main=result['oof_main']
        text.append(f"| {result['engine']} | {result['structure']} | {fmt(f['score_threshold'])} | {fmt(f['margin_threshold'])} | {f['metrics']['macro_f1']*100:.2f}% | {o['macro_f1']*100:.2f}% | {o['no_faq_tp']}/{o['no_faq_rows']} | {o['faq_false_rejections']}/{o['faq_rows']} | {o['faq_recall_presence']*100:.2f}% | {main['gated_hit@1']*100:.2f}% |")
    text += ['', '## fold별 평가 임계값', '',
             '각 행은 표시한 fold의 질문을 피팅에 넣지 않고 평가한 결과다. NO_FAQ 하나씩의 분할에 따른 불안정성을 확인할 수 있다.', '',
             '| 출처 | 구조 | 평가 fold | score 임계값 | margin 임계값 | 평가 NO_FAQ | 거절 | FAQ 오거절 |',
             '|---|---|---:|---:|---:|---|---:|---:|']
    for result in results:
        for fold in result['oof_folds']:
            f=fold['fit'];m=fold['validation_metrics']
            text.append(f"| {result['engine']} | {result['structure']} | {fold['validation_fold']} | {fmt(f['score_threshold'])} | {fmt(f['margin_threshold'])} | {', '.join(fold['validation_no_faq_ids'])} | {m['no_faq_tp']}/{m['no_faq_rows']} | {m['faq_false_rejections']}/{m['faq_rows']} |")
    text += ['', '최적값 탐색 범위는 score+margin AND gate다. NO_FAQ 각 경계를 조금 넘는 값과 off를 전수 탐색한다. 같은 음성 거절 집합에서 FAQ만 추가로 거절하는 더 높은 임계값은 macro-F1을 높이지 못하므로 이 gate 범위의 학습 최적값을 모두 포함한다. 전체 시험값·score-only/margin-only 대조·각 fold 결과는 fit.json에 저장한다.',
             '', '## 판정 로그와 재현', '',
             '32개 구조×5,021개의 판정은 engines/<출처>/<구조>/per_query.jsonl에 저장한다. query, expected_faq_id, Top1/Top2, margin, 두 임계값, 수락 여부, 거절 이유, 실제 context, gate 전후 Hit/FactRecall, OOF 수락 및 fold를 기록한다. 기존 16개 필드 후보 로그와 rank_delta는 유지하고 case_id로 연결한다.',
             '[임계값 설정](thresholds.json) · [비교 JSON](comparison.json) · [분할](split_manifest.json) · [전수 검증](completion_audit.json)',
             '', '실행: `docker compose -f compose.yaml run --rm --no-deps validate scripts/test_threshold_calibration.py` → `docker compose -f compose.yaml run --rm --no-deps validate scripts/calibrate_thresholds.py fit`. 기존 입력과 결과는 읽기만 하고 별도 threshold_calibration_v1에 출력한다.',
             '', '방법 참고: [scikit-learn 임계값 튜닝](https://scikit-learn.org/stable/modules/classification_threshold.html). 본 실험은 2차원 gate를 직접 엄밀하게 탐색하며 TunedThresholdClassifierCV를 호출한 결과가 아니다.']
    (TARGET/'comparison.md').write_text('\n'.join(text)+'\n',encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['fit','report']);args=parser.parse_args()
    if args.stage=='report':
        completed=read(TARGET/'comparison.json');report(completed['structures'],completed['split'])
        raise SystemExit(0)
    try:fit_all()
    except Exception as error:
        save(TARGET/'run_status.json',{'status':'failed','error':str(error),'timestamp':now()});raise
