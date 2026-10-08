"""Compare completed new-corpus engine runs without fabricating thresholds."""
import gc
import hashlib
import json
from collections import Counter,defaultdict
import numpy as np
from common import ROOT,DATA,OUT,N,MAIN_N,REFERENCE_N,NAMES,inputs,read,lines,save,save_lines,sha
from candidate_logs import export_candidate_log,FIELDS

ENGINES=['ollama','vllm']
def pct(v):return '미집계' if v is None else f'{100*v:.2f}%'
def rank(row,gold):return min([row['top_ids'].index(sid)+1 for sid in gold if sid in row['top_ids']] or [999])
def audit(rows,cases):
    assert len(rows)==N and [r['case_id'] for r in rows]==[c['case_id'] for c in cases]
    assert all(r['query_hash']==hashlib.sha256(c['model_query'].encode()).hexdigest() for r,c in zip(rows,cases))
    assert sum(r['evaluation_role']=='main' for r in rows)==MAIN_N

def main():
    cases,docs=inputs();manifest=read(DATA/'manifest.json')
    validation=read(OUT/'input_validation.json')
    index={c['case_id']:i for i,c in enumerate(cases)}
    summaries={};table=[];purposes=[];brands=[];strata=[];language=[];regression_counts={};failure_counts={}
    for engine in ENGINES:
        base=OUT/'engines'/engine
        assert read(base/'stop_verification.json')['verified_stopped']
        selection=read(base/'selection.json');assert selection['all_5000_main_rows_executed'] and selection['stages']==len(NAMES)
        execution=read(base/'embedding_execution.json')
        assert execution['main_question_rows_sent']==MAIN_N and execution['reference_question_rows_sent']==REFERENCE_N
        assert not execution['hf_dense_fallback'] and execution['truncated_inputs']==0
        request_records=lines(base/'embedding_requests.jsonl')
        actual=[r for r in request_records if r.get('success') and r['role']=='query_main_5000']
        assert {i for r in actual for i in range(r['first_row'],r['first_row']+r['texts'])}==set(range(MAIN_N))
        summaries[engine]={}
        baseline=lines(base/'C-M20/per_query.jsonl');audit(baseline,cases)
        for name in NAMES:
            summary=read(base/name/'summary.json')
            rows=lines(base/name/'per_query.jsonl');audit(rows,cases)
            summaries[engine][name]=summary
            cohort=summary['cohort']
            table.append({'engine':engine,'structure':name,'main_rows':MAIN_N,'reference_rows':REFERENCE_N,
                'general':cohort['general']['parent_mean'],'condition':cohort['condition']['parent_mean'],
                'multi_intent':cohort['multi_intent']['parent_mean'],
                'threshold':None,'margin':None,'no_faq_f1':None,'threshold_status':'NOT_FITTED_NO_VALIDATED_NEGATIVES'})
            for purpose in read(DATA/'purpose_catalog.json'):
                tag=purpose['tag']
                data=summary['validation_tag'].get(tag,{'rows':0,'families':0,'parents':0,'row_mean':None,'family_mean':None,'parent_mean':None})
                purposes.append({'engine':engine,'structure':name,'tag':tag,
                    'validation_purpose':purpose['validation_purpose'],'scope_and_limit':purpose['scope_and_limit'],**data})
            for brand,data in summary['brand'].items():brands.append({'engine':engine,'structure':name,'brand':brand,**data})
            grouped=defaultdict(list)
            for row in rows[:MAIN_N]:grouped[(row['brand'],row['category'],row['source_language'])].append(row)
            for (brand,category,source_language),group in sorted(grouped.items()):
                strata.append({'engine':engine,'structure':name,'brand':brand,'category':category,
                    'source_language':source_language,'rows':len(group),
                    'parents':len({r['parent_case_id'] for r in group}),
                    **{f'{metric}@{k}':float(np.mean([r['metrics'][str(k)][metric] for r in group]))
                       for k in [1,3,5,10,20] for metric in ['hit','coverage','all_sources','mrr']}})
            paired=[];failures=[];regressions=[]
            for c,row,before in zip(cases,rows,baseline):
                if c['evaluation_role']!='main':continue
                gold=c['source_ids']
                if c['language_pair_reference']:
                    source=rows[index[c['language_pair_reference']]]
                    assert source['source_ids']==row['source_ids']
                    sr,vr=rank(source,gold),rank(row,gold)
                    paired.append({'original_case_id':source['case_id'],'mixed_case_id':row['case_id'],
                        'variant_slot':c['variant_slot'],'source_language':c['source_language'],
                        'original_all_sources3':source['metrics']['3']['all_sources'],
                        'mixed_all_sources3':row['metrics']['3']['all_sources'],
                        'original_rank_top20':None if sr==999 else sr,'mixed_rank_top20':None if vr==999 else vr,
                        'rank_delta':sr-vr if sr!=999 and vr!=999 else None,
                        'top1_same':source['top_ids'][0]==row['top_ids'][0],
                        'top3_overlap':len(set(source['top_ids'][:3])&set(row['top_ids'][:3]))/3,
                        'original_query':source['query'],'mixed_query':row['query']})
                if not row['metrics']['3']['all_sources']:
                    failures.append({'case_id':c['case_id'],'parent_case_id':c['parent_case_id'],
                        'brand':c['brand'],'category':c['category'],'validation_tags':c['validation_tags'],
                        'failure_type':'RETRIEVAL_MISS' if not row['metrics']['20']['all_sources'] else 'RANKING_ERROR',
                        'query':c['query'],'source_ids':gold,'top3':row['top_ids'][:3],
                        'source_id_gt_requires_independent_review':True})
                if name.startswith('D-'):
                    br,ar=rank(before,gold),rank(row,gold)
                    status='STAGE1_MISS' if not before['metrics']['20']['all_sources'] else 'IMPROVED' if ar<br else 'WORSENED' if ar>br else 'UNCHANGED'
                    regressions.append({'case_id':c['case_id'],'status':status,
                        'baseline_first_source_rank_top20':None if br==999 else br,
                        'reranked_first_source_rank_top20':None if ar==999 else ar,
                        'all_sources3_before':before['metrics']['3']['all_sources'],
                        'all_sources3_after':row['metrics']['3']['all_sources'],
                        'became_complete3':row['metrics']['3']['all_sources']>before['metrics']['3']['all_sources'],
                        'became_incomplete3':row['metrics']['3']['all_sources']<before['metrics']['3']['all_sources']})
            save_lines(base/name/'language_pair_deltas.jsonl',paired)
            save_lines(base/name/'failures.jsonl',failures)
            failure_counts[engine+'/'+name]=dict(Counter(r['failure_type'] for r in failures))
            language.append({'engine':engine,'structure':name,'pairs':len(paired),
                'original_basis':'actual additional original source query, separately counted from 5000 variants',
                'original_all_sources3':float(np.mean([r['original_all_sources3'] for r in paired])),
                'mixed_all_sources3':float(np.mean([r['mixed_all_sources3'] for r in paired])),
                'top1_agreement':float(np.mean([r['top1_same'] for r in paired])),
                'top3_overlap':float(np.mean([r['top3_overlap'] for r in paired])),
                'improvements3':sum(r['mixed_all_sources3']>r['original_all_sources3'] for r in paired),
                'regressions3':sum(r['mixed_all_sources3']<r['original_all_sources3'] for r in paired)})
            if regressions:
                export_candidate_log(engine,name,'question_answer' if '-question_answer-' in name else 'question',cases,docs)
                save_lines(base/name/'reranker_deltas.jsonl',regressions)
                regression_counts[engine+'/'+name]={'first_source_rank_status':dict(Counter(r['status'] for r in regressions)),
                    'all_sources3_improvements':sum(r['became_complete3'] for r in regressions),
                    'all_sources3_regressions':sum(r['became_incomplete3'] for r in regressions)}
            del rows,paired,failures,regressions;gc.collect()
            print(json.dumps({'audited':engine+'/'+name,'main_rows':MAIN_N,'references':REFERENCE_N}),flush=True)
        del baseline;gc.collect()
    save(OUT/'comparison.json',table)
    save(OUT/'per_purpose_comparison.json',purposes)
    save(OUT/'per_brand_comparison.json',brands)
    save(OUT/'per_stratum_comparison.json',strata)
    save(OUT/'language_comparison.json',language)
    save(OUT/'failure_counts.json',failure_counts)
    save(OUT/'reranker_regression_counts.json',regression_counts)
    paired_deltas={}
    changed=[]
    for name in NAMES:
        left=lines(OUT/'engines/ollama'/name/'per_query.jsonl')[:MAIN_N]
        right=lines(OUT/'engines/vllm'/name/'per_query.jsonl')[:MAIN_N]
        improved=worsened=0
        for a,b in zip(left,right):
            x,y=a['metrics']['3']['all_sources'],b['metrics']['3']['all_sources']
            improved+=int(y>x);worsened+=int(y<x)
            if x!=y:changed.append({'structure':name,'case_id':a['case_id'],'brand':a['brand'],
                'ollama_all_sources3':x,'vllm_all_sources3':y,'ollama_top3':a['top_ids'][:3],'vllm_top3':b['top_ids'][:3]})
        paired_deltas[name]={'main_rows':MAIN_N,'top1_agreement':sum(a['top_ids'][0]==b['top_ids'][0] for a,b in zip(left,right))/MAIN_N,
            'top10_overlap':sum(len(set(a['top_ids'][:10])&set(b['top_ids'][:10]))/10 for a,b in zip(left,right))/MAIN_N,
            'vllm_improvements_all_sources3':improved,'vllm_regressions_all_sources3':worsened}
        del left,right;gc.collect()
    save(OUT/'paired_engine_deltas.json',paired_deltas)
    save_lines(OUT/'changed_source_coverage.jsonl',changed)
    q1=np.load(OUT/'engines/ollama/cache/query_main_5000.npy',mmap_mode='r')
    q2=np.load(OUT/'engines/vllm/cache/query_main_5000.npy',mmap_mode='r')
    cos=np.empty(MAIN_N,dtype=np.float32)
    for i in range(0,MAIN_N,256):cos[i:i+256]=np.sum(q1[i:i+256]*q2[i:i+256],axis=1)
    save(OUT/'dense_vector_parity.json',{'rows':MAIN_N,'min_cosine':float(cos.min()),'mean_cosine':float(cos.mean()),
        'p05_cosine':float(np.quantile(cos,.05)),'note':'Includes GGUF F16 versus HF FP32 format/precision differences; not an isolated engine effect.'})
    report=['# 새 FAQ 5,000행 · Ollama/vLLM 구조 비교','',
        '새 faqs.jsonl 기반 500쌍×10개, 본 평가 5,000행을 Ollama와 vLLM 각각 실행했다. 원문 기준 질의 500행은 한영 비교를 위해 별도로 실행했다. 이전 테스트 질문·정답·벡터·검색 결과·임계값은 사용하지 않았다.','',
        f"입력 변형 SHA256: `{manifest['source_variants_sha256']}`. FAQ 코퍼스 {len(docs):,}개, 질문 최대 {validation['queries_max_tokens']}토큰, 전체 질문+답변 문서 최대 {validation['full_qa_docs_max_tokens']}토큰. Dense 입력 길이 8,192토큰, reranker 입력 길이 1,024토큰으로 고정했으며 질문과 답변을 자르지 않았다.",'',
        '원천/변형 의미와 정답의 독립 검수는 미완료다. 지표는 canonical FAQ ID의 SourceHit/SourceCoverage/AllSourcesHit다. 원자 FactRecall·생성 정답률·별도 Holdout 결과가 아니다. 같은 부모의 변형을 독립 관측으로 간주하지 않고 부모 평균과 행 평균을 구분한다.','',
        '## 같은 구조에서 엔진 비교','',
        '| 구조 | Ollama 일반 AllSources@3 | vLLM 일반 AllSources@3 | Ollama 복합 AllSources@3 | vLLM 복합 AllSources@3 |',
        '|---|---:|---:|---:|---:|']
    for name in NAMES:
        o=summaries['ollama'][name]['cohort'];v=summaries['vllm'][name]['cohort']
        report.append(f"| {name} | {pct(o['general']['parent_mean']['all_sources@3'])} | {pct(v['general']['parent_mean']['all_sources@3'])} | {pct(o['multi_intent']['parent_mean']['all_sources@3'])} | {pct(v['multi_intent']['parent_mean']['all_sources@3'])} |")
    report+=['','일반 3238행, 조건 1262행, 복합 500행을 별도로 집계했다. V10은 두 FAQ가 모두 필요한 질문이다. 추가 원문 500행은 본 평가 평균에 합산하지 않는다.','']
    for engine in ENGINES:
        ex=read(OUT/'engines'/engine/'embedding_execution.json')
        selection=read(OUT/'engines'/engine/'selection.json')
        lp=next(r for r in language if r['engine']==engine and r['structure']==selection['selected'])
        report+=[f'## {engine}','',
            f"본 평가 질의 HTTP 시간 {ex['main_query_http_seconds']:.2f}초, 질의 처리량 {ex['main_queries_per_second']:.2f}개/초. 원문 기준 500질의 시간 {ex['reference_query_http_seconds']:.2f}초, 두 문서 view HTTP 시간 {ex['document_http_seconds']:.2f}초. 실제 HTTP 요청 실패 {ex['request_failures']}건. 순차 가변 batch 값으로 동시 부하 성능 결과가 아니다.",
            f"긍정 문항 내 진단 기준으로 선정된 구조는 **{selection['selected']}**다. 선정 규칙은 일반 부모 AllSources@3, 동률 시 복합·조건 AllSources@3와 일반 Hit@1 순이다. Holdout 검증 및 운영 적용 선정은 미완료다.",
            f"선정 구조의 한영 대응어 {lp['pairs']:,}쌍: 실제 원문 AllSources@3 {pct(lp['original_all_sources3'])}, 변형 {pct(lp['mixed_all_sources3'])}. 개선 {lp['improvements3']:,}쌍, 악화 {lp['regressions3']:,}쌍. V04는 대응어 한 개, V05는 여러 대응어/문장 변형이므로 모든 변화가 언어 하나의 효과라고 가정하지 않는다.",'']
    report+=['## 임계값과 실행 범위','',
        '질의 HTTP 시간에는 각 엔진의 첫 실제 embedding 요청도 포함된다. Ollama의 readiness는 /api/tags이며 모델은 첫 embedding에서 지연 로드된다. vLLM /health는 모델 초기화 후 준비된다. 모델 포맷·정밀도, 첫 로드 시점과 길이별 가변 batch를 함께 포함하는 이번 조건의 처리량이며, 동일 정밀도의 warm-state 동시 부하 비교로 해석하지 않는다.','',
        '앞선 30,000행 실행에서 긴 문서 physical batch 한도를 진단했다. 이번 축소 실행은 입력 길이에 따라 num_batch 2,048/4,096, num_ctx=8,192, truncate=false를 적용한다. 이번 엔진별 질문과 전체 문서 벡터·reranker 점수·native 점수는 새로 계산하며 이전 실행 값을 재사용하지 않는다. [Ollama v0.35.0 원본 코드](https://github.com/ollama/ollama/blob/v0.35.0/server/routes.go)의 관련 동작과 앞선 실측을 기준으로 설정했다.','',
        '**NO_FAQ 문항이 0개이므로 threshold/margin은 미적합(null)이다. NO_FAQ precision/recall/F1도 계산하지 않았다.** 점수·margin 분위수는 각 엔진 `policies.json`에 기록했다. 다른 구조의 점수와 임계값을 이식하지 않는다.','',
        'A 질문 dense, B 질문+답변 dense, C 각 view Top-20 union에서 max cosine으로 FAQ 20개 선택. D는 각 엔진의 C 후보에 BGE/Jina×질문/질문+답변 네 조합을 적용했다. 질문+답변은 해당 FAQ의 모든 분할 답변 창을 점수화하고 최고 raw logit으로 합쳤다. 후보 FAQ budget은 20개이며 창을 별도 FAQ로 세지 않는다.','',
        'E sparse와 F full은 공통 FlagEmbedding native 대조군을 각 세트에서 다시 계산했다. E hybrid와 F dense-pool은 해당 엔진의 dense 결과를 사용한다. D는 공통 CUDA FP16, E/F는 공통 CUDA FP32이며 엔진 자체의 sparse/multi-vector/reranker API를 측정한 결과가 아니다. FP32/FP16 표본 차이는 D의 precision-probe에 기록했다. Ollama GGUF F16과 vLLM HF FP32의 포맷·정밀도 차이도 비교 결과에 포함한다.','',
        'Ollama 기동→readiness→HTTP 전체 측정→종료 및 폐쇄 확인→Ollama 후보 보조 모델 실행→vLLM의 같은 순서로 진행했다. TEI는 미실행이다. 반복 안정성·실제 장애 로그·대화 rewrite·문서 공격·개인 API·생성 모델·원자 사실 GT·NO_FAQ 음성 문항·Holdout은 이번 실행 범위에 포함하지 않았다.','',
        '## 산출물','',
        '`comparison.json`은 구조별 결과, `per_purpose_comparison.json`·`per_brand_comparison.json`은 목적/브랜드별 지표, `language_comparison.json`은 실제 원문 대비 한영 변형, `reranker_regression_counts.json`은 C 대비 개선/악화다. 엔진별 각 구조 `per_query.jsonl`·`failures.jsonl`·`language_pair_deltas.jsonl`·D `reranker_deltas.jsonl`에 개별 사례가 있다. `paired_engine_deltas.json`·`changed_source_coverage.jsonl`·`dense_vector_parity.json`에 엔진 차이를 기록했다.','']
    report+=['각 D 구조 `candidate_log.jsonl`은 질의×후보 FAQ별 query, expected_faq_id(필수 FAQ 배열), faq_id, retrieval_rank/cosine, retrieval_a_rank/cosine, retrieval_b_rank/cosine, rerank_rank/score, rank_delta, window_count, winning_window_index/text/score를 기록한다. rank_delta = retrieval_rank - rerank_rank이며 양수는 개선, 음수는 하락, 0은 유지다. 본 평가 100,000행과 원문 기준 10,000행이다. retrieval rank는 C 후보 내 1-based 순위, A/B rank는 전체 3,246개 FAQ의 각 view 순위, winning window는 FAQ 안의 0-based 순서다. 동점 문서 창은 먼저 나온 창을 선택한다. 정의와 파일 해시는 candidate_log_schema.json에 있다.','']
    (OUT/'새FAQ_Ollama_vLLM_비교.md').write_text('\n'.join(report),encoding='utf-8')
    save(OUT/'completion_audit.json',{'engines':ENGINES,'structures_per_engine':len(NAMES),'main_rows_per_structure':MAIN_N,
        'reference_rows_per_structure':REFERENCE_N,'main_case_structure_results':MAIN_N*len(NAMES)*2,
        'reference_case_structure_results':REFERENCE_N*len(NAMES)*2,'corpus_faqs':len(docs),
        'reranker_candidate_log_files':8,'reranker_candidate_log_rows':N*20*8,
        'main_reranker_candidate_log_rows':MAIN_N*20*8,'reference_reranker_candidate_log_rows':REFERENCE_N*20*8,
        'reranker_candidate_log_fields':FIELDS,'actual_winning_windows_recorded':True,
        'all_main_queries_and_order_verified':True,'all_engines_stopped':True,'tei_executed':False,
        'hf_dense_fallback':False,'input_truncated':False,'no_faq_threshold_fitted':False,'holdout_executed':False,
        'independent_annotation_complete':False,'input_sha256':sha(DATA/'cases.jsonl')})
    save(OUT/'execution_code_identity.json',{'scripts':{p.name:sha(p) for p in (ROOT/'scripts').glob('*.py')},
        'compose_sha256':sha(ROOT/'compose.yaml'),'run_all_sha256':sha(ROOT/'run_all.ps1')})
    save(OUT/'artifact_inventory.json',{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['artifact_inventory.json','run_status.json']})
    print(json.dumps({'completed_engines':ENGINES,'structures_per_engine':len(NAMES),'main_rows':MAIN_N,'reference_rows':REFERENCE_N,'no_faq_threshold_fitted':False}),flush=True)
if __name__=='__main__':main()
