"""Compare completed, matched engine matrices without another model run."""
import csv,hashlib,json
from collections import Counter
from common import ROOT,OUT,inputs,read,lines,save,save_lines,sha

ENGINES=['ollama','vllm','tei']
NAMES=['A','B','C-M20','D-bge-question-M20','D-bge-question_answer-M20','D-jina-question-M20',
       'D-jina-question_answer-M20','E-sparse','E-dense-sparse-RRF-M20','F-full','F-dense-pool-M20']
def pct(v):return '미집계' if v is None else f'{100*v:.2f}%'
def main():
    cs,ds=inputs();manifest=read(ROOT/'data/manifest.json');validation=read(OUT/'input_validation.json')
    exact_question_rows=sum(c['query'] in {d['question'] for d in ds} for c in cs)
    result={};table=[];types=[];all_rows={}
    for engine in ENGINES:
        base=OUT/'engines'/engine;stop=read(base/'stop_verification.json');assert stop['verified_stopped']
        selection=read(base/'selection.json');assert selection['all_3000_rows_executed'] and selection['stages']==11
        execution=read(base/'embedding_execution.json');assert execution['question_rows_sent']==3000 and not execution['hf_dense_fallback']
        requests=lines(base/'embedding_requests.jsonl');assert sum(r['texts'] for r in requests if r['role']=='query_all_3000')==3000
        policies=read(base/'policies.json');summaries={}
        for name in NAMES:
            s=read(base/name/'summary.json');rows=lines(base/name/'per_query.jsonl')
            assert s['case_count']==3000 and [r['case_id'] for r in rows]==[c['case_id'] for c in cs]
            assert Counter(r['type'] for r in rows)==Counter(c['type'] for c in cs)
            assert [r['query'] for r in rows]==[c['query'] for c in cs]
            summaries[name]=s;all_rows[(engine,name)]=rows
            g=s['cohort']['general']['family_mean'];t=s['cohort']['condition']['family_mean']
            table.append({'engine':engine,'structure':name,'rows_executed':3000,'general_hit1':g['hit@1'],
                          'general_hit3':g['hit@3'],'general_all_sources3':g['all_sources@3'],
                          'condition_all_sources3':t['all_sources@3'],'calibration_no_faq_f1':policies[name]['no_faq_f1'],
                          'threshold':policies[name]['threshold'],'margin':policies[name]['margin'],
                          'dense_backend':s['execution'].get('dense_backend'),
                          'auxiliary_backend':s['execution'].get('reranker_backend') or s['execution'].get('native_backend')})
            for typ,v in s['type'].items():types.append({'engine':engine,'structure':name,'type':typ,'rows':v['rows'],
                                                       'ranking_rows':v['ranking_rows'],**v['row_mean']})
        result[engine]={'selection':selection,'structures':summaries,'embedding_execution':execution,'stopped':True}
    for filename,rows in [('comparison.csv',table),('per_type_comparison.csv',types)]:
        with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(OUT/'comparison.json',result)
    deltas={};changed=[]
    for name in NAMES:
        baseline=all_rows[('ollama',name)]
        for engine in ['vllm','tei']:
            rr=all_rows[(engine,name)];agreement=sum(a['top_ids'][0]==b['top_ids'][0] for a,b in zip(baseline,rr))/3000
            overlap=sum(len(set(a['top_ids'][:10])&set(b['top_ids'][:10]))/10 for a,b in zip(baseline,rr))/3000
            improved=worsened=0
            for a,b in zip(baseline,rr):
                before=a['metrics']['3']['all_sources'];after=b['metrics']['3']['all_sources']
                if before is None:continue
                improved+=int(after>before);worsened+=int(after<before)
                if after!=before:changed.append({'structure':name,'engine':engine,'case_id':a['case_id'],'type':a['type'],
                                                'ollama_all_sources3':before,'engine_all_sources3':after,
                                                'ollama_top3':a['top_ids'][:3],'engine_top3':b['top_ids'][:3]})
            deltas[name+'/'+engine]={'paired_rows':3000,'top1_agreement_with_ollama':agreement,
                                    'top10_overlap_with_ollama':overlap,'source_coverage_improvements_at3':improved,
                                    'source_coverage_regressions_at3':worsened}
    save(OUT/'paired_deltas.json',deltas);save_lines(OUT/'changed_source_coverage.jsonl',changed)
    text=['# 엔진별 원본 3,000문항 Calibration 구조 비교','',
          'Ollama·vLLM·TEI 각각의 BGE-M3 dense API로 동일한 3,000행을 실제 실행했다. 15유형×200행, 원문 질문·행 순서·반복 200행을 유지했다. 각 엔진에서 A~F 11개 구조의 질의별 결과를 보존했다. 기존 603문항 실험의 질문·결과·임계값은 이번 실행 입력으로 가져오지 않았다.','',
          f"사용자가 지정한 원본 workbook SHA256은 `{manifest['source_sha256']}`다. 원문 고유 질문은 {manifest['unique_raw_questions']}개이며 같은 문자열과 동일 정책 변형이 포함되어 3,000개의 독립 표본이라고 해석하지 않는다. 모델 query 최댓값은 {validation['queries_max_tokens']} tokens다. {exact_question_rows}행의 사용자 질문은 FAQ 원문 질문과 정확히 같다. 반복 200행은 실행 ID의 원본 반복 입력별로 묶으며 서로 다른 입력을 같은 안정성 그룹으로 합치지 않는다.",'',
          '원문의 제공 Context는 고정 생성 평가용 정답 근거를 포함하므로 검색 입력에서 제외했다. 원문 대화 이력은 고정 규칙으로 연결하고 사용자 질문 자체는 수정하지 않았다. 정답·추적 ID·채점 기준·API 응답·페르소나는 임베딩과 reranker 입력에 넣지 않았다. API 호출·생성 응답·충돌 해결·페르소나 준수는 이 검색 구조 평가의 성공률로 주장하지 않는다.','',
          '## 공통 실행 조건','',
          'FAQ 원문 1,024문서를 대상으로 검색했다. A는 질문, B는 질문+답변, C는 두 view의 Top-20 union에서 FAQ ID 중복을 제거하고 max cosine으로 20개를 선택했다. D는 각 엔진 C의 동일한 20개 후보를 BGE/Jina×질문/질문+답변 네 조합으로 재정렬했다.','',
          'D reranker는 공통 HF/PyTorch CUDA FP32이며 E/F sparse·late interaction은 공통 FlagEmbedding CUDA FP32다. E hybrid와 F pool의 dense 후보는 해당 엔진의 실제 API 결과다. E sparse와 F full은 엔진별로 다시 실행한 공통 native 대조군이므로 엔진의 sparse/multi-vector 서빙 성능으로 해석하지 않는다. 모든 부품을 엔진 자체 API로 서빙한 실험은 아니다.','',
          'Ollama는 GGUF F16, TEI와 vLLM은 FP32로 실행했다. 고정 모델 가중치·이미지 cache만 재사용했으며 다른 엔진의 query·후보·reranker 점수·native 특징 결과를 재사용하지 않았다. Query 3,000행은 중복을 제거하지 않고 엔진에 보냈다. D는 매 조합마다 60,000개의 query/document pair를 실제 계산하여 반복 문항도 다시 점수화했다.','',
          '## 지표 계약','',
          'SourceHit@K는 추적 FAQ ID 중 하나 이상이 상위 K에 있는지, SourceCoverage@K는 추적 ID의 회수 비율, AllSourcesHit@K는 모든 추적 ID가 있는지를 뜻한다. 원문 추적 ID는 독립 검수한 원자적 사실·동등 근거 집합과 다르므로 기존 603문항의 semantic/AllFacts 수치와 직접 비교하지 않는다.','',
          '아래 일반 지표는 SF·NC·PS 600행, 조건 지표는 CE·MC 400행에서 원문의 독립 집계 단위별 평균을 낸 값이다. PI·MT·UI·CF·AR·AD·RT와 Context 부족 10행은 별도 집계한다. 모든 15유형 200행씩 실제 검색했으며 별도 집계는 행 삭제를 의미하지 않는다. SR·HR·EC 중 추적 ID가 없는 NO_ANSWER_TOPIC 590행만 NO_FAQ calibration에 포함했다. PARTIAL·API 요구·고정 Context의 ABSTAIN을 무조건 NO_FAQ로 바꾸지 않았다.','',
          '## 같은 구조에서 엔진 비교','',
          '| 구조 | Ollama 일반 AllSources@3 | vLLM 일반 AllSources@3 | TEI 일반 AllSources@3 |','|---|---:|---:|---:|']
    for name in NAMES:
        text.append('| '+name+' | '+' | '.join(pct(result[e]['structures'][name]['cohort']['general']['family_mean']['all_sources@3']) for e in ENGINES)+' |')
    for engine in ENGINES:
        text+=['',f'## {engine} Calibration 한 세트','',
               '| 구조 | 실행 행 | 일반 SourceHit@1 | 일반 SourceHit@3 | 일반 AllSources@3 | 조건 AllSources@3 | Calibration NO_FAQ F1 |',
               '|---|---:|---:|---:|---:|---:|---:|']
        for row in [r for r in table if r['engine']==engine]:
            text.append(f"| {row['structure']} | 3,000 | {pct(row['general_hit1'])} | {pct(row['general_hit3'])} | {pct(row['general_all_sources3'])} | {pct(row['condition_all_sources3'])} | {row['calibration_no_faq_f1']:.4f} |")
        sel=result[engine]['selection'];policy=sel['policy'];text+=['',
          f"이번 calibration 선정 구조는 **{sel['selected']}**, 해당 score threshold는 **{policy['threshold']:.8f}**, margin은 **{policy['margin']:.8f}**다. 각 구조마다 이번 1,590행(일반·조건 1,000 + NO_FAQ 590)에서 NO_FAQ F1 최대, 동률이면 정답 질문 채택률 최대 기준으로 새 임계값을 계산했다. 점수 척도가 구조별로 다르므로 임계값을 다른 모델에 그대로 이식하지 않는다. 별도 holdout이 없어 표의 F1은 학습에 쓴 Calibration 자체의 수치다.",
          f"실제 HTTP query 입력은 {result[engine]['embedding_execution']['question_rows_sent']}행이며 HF dense 대체 실행은 없었다. 엔진 기동→확인→전체 실험→종료→HTTP 폐쇄·활성 컨테이너 0 확인을 완료했다."]
    text+=['','## 실행 확인과 파일','',
          '반복 안정성은 각 엔진 `repeat_stability.json`, 임계값은 `policies.json`, 모든 3,000행의 결과는 각 구조 `per_query.jsonl`에 있다. `comparison.csv`는 엔진×구조 전체 지표, `per_type_comparison.csv`는 15유형별 지표다. `paired_deltas.json`은 동일 문항 기준 엔진 간 rank 변화, `changed_source_coverage.jsonl`은 Top-3 근거 회수가 달라진 문항이다.','',
          '고정 Context를 없애고 원본 질문으로 새 검색을 수행했으므로 원본 생성 평가의 정답 상태만으로 검색·답변 행동을 모두 채점하지 않았다. 추적 ID의 대체 근거 누락 가능성, 동일 질문/정책 중복, 생성·API 미실행, 전체 Calibration 사용이라는 제한을 포함해 해석한다.','']
    report=OUT/'엔진별_Calibration_3000_비교.md';report.write_text('\n'.join(text),encoding='utf-8')
    save(OUT/'completion_audit.json',{'engines':3,'structures_per_engine':11,'case_rows_per_structure':3000,
         'case_structure_results':99000,'original_questions_and_order_verified':True,'type_counts_each_engine':manifest['type_counts'],
         'repeat_rows_each_structure':200,'all_engines_stopped':True,'hf_dense_fallback':False,
         'common_auxiliary_models_explicit':True,'holdout_executed':False,'source_manifest_sha256':sha(ROOT/'data/manifest.json')})
    save(OUT/'execution_code_identity.json',{'scripts':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted((ROOT/'scripts').glob('*.py'))},
                                          'compose_sha256':sha(ROOT/'compose.yaml'),'run_all_sha256':sha(ROOT/'run_all.ps1')})
    inventory={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_inventory.json'}
    save(OUT/'artifact_inventory.json',inventory);print(str(report),flush=True)

if __name__=='__main__':main()
