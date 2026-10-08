"""Write the new experiment report and exact artifact hash inventory."""
import hashlib,json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/run-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def pct(x):return '미측정' if x is None else f'{100*x:.2f}%'
def main():
    m=read(ROOT/'data/split_manifest.json');s=read(OUT/'selection.json');h=read(OUT/'holdout/policy_results.json')
    e=read(OUT/'serving/shortlist.json');d=read(OUT/'diagnostics/summary.json');hw=read(OUT/'hardware/cpu-gpu.json')
    langs=read(OUT/'analysis/language_summary.json')[s['selected']]
    holdout_langs=read(OUT/'analysis/language_summary.json')['holdout-'+s['selected']]
    native=read(OUT/'native_decision.json')
    effects=read(OUT/'analysis/policy_effects.json')
    resources=read(OUT/'serving/resource_summary.json')
    lengths=read(OUT/'calibration/B/token_lengths.json')
    native_status='조건부 실행했다' if native['execute_sparse_and_multivector'] else '초기 비교 기준에 따라 실행하지 않았다'
    names=['A','B','C-M20','D-bge-question-M20','D-bge-question_answer-M20','D-jina-question-M20','D-jina-question_answer-M20','E-sparse','E-dense-sparse-RRF-M20','F-full','F-dense-pool-M20']
    table=['| 구조 | 일반 Hit@1 | 일반 Hit@3 | 일반 AllFactsHit@3 | 조건 AllFactsHit@3 |','|---|---:|---:|---:|---:|']
    for name in names:
        p=OUT/'calibration'/name/'summary.json'
        if not p.exists():continue
        ss=read(p);g=ss['cohorts']['core_general']['family_mean'];t=ss['cohorts']['targeted_condition']['family_mean']
        table.append(f"| {name} | {pct(g['semantic_hit@1'])} | {pct(g['semantic_hit@3'])} | {pct(g['all_facts_hit@3'])} | {pct(t['all_facts_hit@3'])} |")
    hg=h['cohorts']['core_general']['family_mean'];hn=h['no_faq']
    engine_decision=(f"공통 dense component의 첫 후보는 **{e['first_candidate']}**다."
                     if e['first_candidate'] else
                     '세 엔진 모두 사전에 고정한 strict parity 기준을 통과하지 못해 **확정한 엔진 후보는 없다**. 속도 결과만으로 기준을 낮추거나 엔진을 선정하지 않았다.')
    text=[ '# README 기반 신규 테스트 실행 결과', '',
           f"새 폴더의 평가셋으로 선택한 구조는 **{s['selected']}**다. 동결한 구조로 holdout을 한 번 평가했으며, 일반 검색 Hit@1은 **{pct(hg['semantic_hit@1'])}**, Hit@3은 **{pct(hg['semantic_hit@3'])}**다. 독립 검수가 끝나지 않은 잠정 실험이므로 제품 적용 확정 결과로 사용하지 않는다.", '',
           '기존 테스트 질문·분류 비율·정답 라벨·선정 결과·임계값은 새 평가 입력으로 가져오지 않았다. 검색 코퍼스는 사용자 파일의 `FAQ 원문` 1,024문서만 사용했다. 기존 프로그램 코드를 복사해 수정했고 모델 이미지·고정 가중치 캐시를 재사용했다. 기존 프로젝트 파일과 결과는 원래 위치에 남아 있다.', '',
           f"새 문항 {m['non_repeat_rows']}행, 반복 실행 {m['case_rows']-m['non_repeat_rows']}행, 반복 제외 고유 query 문자열 {m['unique_non_repeat_query_strings']}개다. 사실 작성 seed는 {m['factual_anchor_count']}개이고 연결된 정책 분할 그룹은 {m['policy_split_components']}개다. 같은 정책의 표현 변형과 동일 입력 반복을 독립 표본으로 보지 않는다.", '',
           'calibration main 322행과 holdout main 249행은 일반·조건·NO_FAQ만 포함한다. 문맥 12행, 문서 공격 8행, 확인 질문 12행, 반복 40회는 일반 검색 평균에서 분리했다. 일반·조건 표는 정책 그룹별 평균이다. 한영 표와 유형별 상세는 각 행의 평균이므로 직접 같은 분모로 비교하지 않는다.', '',
           '## Calibration 구조 비교', '',*table,'',
           f'A/B/C는 동일 query를 사용했고 C는 질문·질문+답변 결과를 합쳐 FAQ ID로 중복을 제거했다. D의 BGE/Jina×질문/질문+답변 네 셀은 동일한 C 후보 집합을 사용했다. 후보 수 10/20과 C의 원시 union·동일 후보 수 통제 결과도 보존했다. Reranker의 문항별 개선·악화·후보 누락은 각 `rank_changes.jsonl`에 기록했다. E/F는 A~D를 먼저 확인한 뒤 {native_status}. F full corpus는 진단용으로만 두고 구조 선정에는 후보 20개의 F를 비교했다. BM25는 독립 검수된 lexical miss 분류가 없어 실행하지 않았다.', '',
           '## 동결 Holdout', '',
           f"선정 기준은 calibration의 일반 AllFactsHit@3, 조건 AllFactsHit@3, 일반 Hit@1 순이다. threshold와 margin은 새 calibration의 NO_FAQ F1을 기준으로 결정했다. 선택 파일의 해시와 평가셋 해시를 기록했고 holdout 결과를 보고 다시 조정하지 않았다. `holdout_passes={h['holdout_passes']}`.", '',
           f"일반 Hit@1 {pct(hg['semantic_hit@1'])}, Hit@3 {pct(hg['semantic_hit@3'])}, FactRecall@3 {pct(hg['fact_recall@3'])}, AllFactsHit@3 {pct(hg['all_facts_hit@3'])}, FactRecall@10 {pct(hg['fact_recall@10'])}.", '',
           f"위 검색 지표는 거절 정책 적용 전 순위 지표다. 거절된 정답 질문을 실패로 포함하면 일반 유효 Hit@3은 {pct(effects['core_general']['accepted_hit3'])}, 유효 AllFactsHit@3은 {pct(effects['core_general']['accepted_all_facts3'])}다. Calibration에서 Jina 선정은 일반 AllFactsHit@3을 우선한 결과이며, 질문+답변 dense보다 일반 Hit@1은 낮았다. 비용·Top-1 손실까지 고려한 제품 채택 판단과 구분한다.", '',
           f"NO_FAQ precision {pct(hn['precision'])}, recall {pct(hn['recall'])}, F1 {pct(hn['f1'])}. 정답 없는 질문의 잘못된 채택 {hn['fn']}건, 정답 있는 질문의 거절 {hn['fp']}건. 답이 필요한 개인 조회는 NO_FAQ로 분류하지 않았다. API 실행 자체와 생성 응답은 이 retrieval 실험에서 검증하지 않았다.", '',
           '## 한국어·영어 의미 동등성', '',
           '| 변형 | Calibration 비교 쌍 | 한국어 Hit@3 | 변형 Hit@3 | 한국어 성공→변형 실패 | Top-10 overlap |','|---|---:|---:|---:|---:|---:|']
    for v,x in langs.items():text.append(f"| {v} | {x['pairs']} | {pct(x['anchor_hit3'])} | {pct(x['variant_hit3'])} | {x['ko_hit_en_miss3']} | {pct(x['mean_top10_overlap'])} |")
    text+=['','Holdout의 이미 저장한 결과에 대해서도 같은 쌍을 분석했다. 이후 모델·정답·임계값은 바꾸지 않았다.','',
           '| 변형 | Holdout 비교 쌍 | 한국어 Hit@3 | 변형 Hit@3 | 한국어 성공→변형 실패 |','|---|---:|---:|---:|---:|']
    for v,x in holdout_langs.items():text.append(f"| {v} | {x['pairs']} | {pct(x['anchor_hit3'])} | {pct(x['variant_hit3'])} | {x['ko_hit_en_miss3']} |")
    text+=['','GT·조건·숫자를 공유하는 term 변경 쌍과 전체 영어 문장 쌍을 분리했다. eSIM↔이심, 유심↔SIM, 와이파이↔Wi-Fi, QR 코드↔QR code와 일반 서비스 용어를 포함한다. 영어 상품명은 테스트용 의미 대응이며 공식 영문 브랜드라고 주장하지 않는다. Rank delta는 저장한 후보 밖에서 censored되며 숫자 비교 연산과 단위의 의미 일치는 독립 검수 대상이다.', '',
           '## 분리 진단', '',
           f"문맥은 `history_concat_v1` 규칙을 고정해 12행을 실행했다. 정답 라벨로 만든 rewrite를 입력하지 않았다. 반복은 네 입력을 각각 10번 실제 재실행했으며, 결과는 다음과 같다.", '',
           '| 반복 그룹 | 실제 실행 | Rank hash 종류 | Context hash 종류 |','|---|---:|---:|---:|']
    for k,x in d['repeat'].items():text.append(f"| {k} | {x['executions']} | {x['rank_hash_count']} | {x['context_hash_count']} |")
    text+=['',f"새 문서 공격 fixture {d['document_attack']['cases']}건 중 Top-3에 공격 fixture가 포함된 것은 {d['document_attack']['retrieved_at_3']}건이다. 이는 검색·구성 Context의 노출 여부이며 LLM에 전달하거나 명령을 따르는지 생성 테스트는 하지 않았다. AMBIGUOUS_QUERY 12행은 CLARIFY 계약만 검증했다. 명확화 실행 정책과 생성 모델이 없으므로 행동 성공률을 산출하지 않았다. REAL_FAILURE는 실제 로그 부재, TEMPORAL_VERSION은 유효기간·버전 정보 부재로 각각 0건이다.", '',
           '## Docker Serving 엔진', '',
           '각 엔진은 켜기→health/embedding 확인→측정→끄기→컨테이너 종료·HTTP 폐쇄·활성 컨테이너 0 확인 순서로 실행했다. 새 폴더만 읽기 전용으로 연결했고 결과 폴더만 쓰기를 허용했다. Jina custom code는 offline·network none·read only 환경에서 실행했다.', '',
           '| 엔진 | Dense parity | 실패 요청 | 셀 수 | C1/B1 p95 ms | 최고 texts/s | 종료 확인 |','|---|---|---:|---:|---:|---:|---|']
    for x in e['engines']:
        stop=read(OUT/f"serving/{x['engine']}/stop_verification.json")
        text.append(f"| {x['engine']} | {x['parity']} | {x['failed_attempts']} | {x['cells']} | {x['single_request_p95_ms']:.2f} | {x['best_texts_per_second']:.2f} | {stop['verified_stopped']} |")
    text+=['', '| 엔진 | Ready까지 초 | RAM 표본 최고 GiB | 호스트 VRAM 표본 최고 MiB | 0.9999 미만 문서 수 | Top-1 일치 | Top-10 overlap |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,x in resources.items():
        text.append(f"| {name} | {x['cold_start_to_ready_seconds']:.2f} | {x['container_ram_sampled_peak_bytes']/1024**3:.2f} | {x['host_total_vram_sampled_peak_mib']:.0f} | {x['corpus_vectors_below_0_9999']} | {pct(x['top1_agreement'])} | {pct(x['top10_overlap'])} |")
    text+=['','RAM은 Docker 컨테이너 사용량의 주기적 표본 최고값이며 VRAM은 데스크톱 등을 포함한 호스트 전체 GPU 사용량이다. 프로세스별 VRAM 할당량이나 매 순간의 정확한 peak라고 해석하지 않는다. TEI는 새 CUDA kernel cache를 만드는 첫 시작까지 포함했으므로 ready 시간은 cache가 이미 준비된 환경의 시작 시간과 구분한다.', '',
           'TEI 시작 로그에는 GELU를 tanh 근사로 계산한다는 메시지가 있으며, HF의 exact GELU와 차이가 날 수 있다고 명시돼 있다. strict parity 판정은 코퍼스·query cosine 최솟값 0.9999를 모든 엔진에 동일하게 적용했다. 이 검사의 실패를 곧바로 검색 정확도 실패로 해석하지 않는다. 문서 view는 질문+답변이며 비교 질문 50개는 신규 유형·언어 strata로 선택했다. 각 엔진의 cosine, Top-1 일치율, Top-10 겹침, 실제 모델 형식은 `serving/*/parity.json`에 기록했다.']
    text+=['',f"{engine_decision} 각 엔진에서 concurrency 1/4/8/16/32 × 요청당 query 1/8/32의 15개 셀을 측정했다. 셀당 10초 이상·100회 이상 요청, 1회 실험이므로 장시간 안정성 결론은 내리지 않는다. 입력 배치 순서, 1,024차원, 정규화 벡터 cosine과 검색 rank를 비교했다. Ollama GGUF F16은 HF float32와 구현·정밀도가 달라 drift가 있으면 같은 lane의 후보에서 제외했다. 선택한 전체 Retrieval+Reranker의 엔진별 종단 성능을 측정한 결과는 아니다.", '',
           '## 별도 CPU/GPU', '',
           f"엔진 비교와 일치 검사를 마친 뒤 동일한 SentenceTransformer BGE-M3 float32 component를 CPU와 GPU에서 각각 실행했다. 엔진 후보 확정 여부와 별개로 수행한 component 측정이다. CPU는 GPU 노출 없이 실행하고 GPU는 CUDA를 확인했다. 벡터 cosine 최솟값 {hw['cosine_min']:.8f}. 동일 query hash와 batch 1/8/32, 각 30요청을 사용했다. 엔진별 CPU/GPU 종단 비교와 구분한다.", '',
           '| 장치 | Batch | p95 ms | texts/s |','|---|---:|---:|---:|']
    for x in hw['results']:text.append(f"| {x['device']} | {x['batch']} | {x['p95_ms']:.2f} | {x['texts_per_second']:.2f} |")
    text+=['','## 해석 제한과 재현','',
           f"Calibration 문서·질문의 token 길이 최댓값은 {lengths['max']}이며 512 tokens에서 잘린 입력은 {lengths['truncated_count']}개다. LONG_QUERY는 상대적으로 긴 표현 변형을 포함하지만 context 한계에 가까운 장문이나 chunking stress를 검증한 데이터는 아니다. 짧은 초기 부하 측정에서 전체 장문 workload의 성능을 추론하지 않는다.", '',
           '정답과 허용 근거, HARD_NEGATIVE, NO_FAQ 부재 검수는 독립 검수가 끝나지 않았다. 답변 문자열 predicate로 찾은 대체 FAQ가 질문의 모든 하위 사실을 증명하는지는 사람이 확인해야 한다. 일부 대체 문서 누락과 숫자 대조의 사실 범위 문제를 `annotation_limitations.json`에 기록했으며, 결과를 보고 이번 동결 라벨을 바꾸지 않았다. 기존 테스트 열람 이력 때문에 완전한 블라인드 작성도 보장하지 않는다. 이러한 제한을 포함한 초기 비교 실험이다.', '',
           '평가 입력·근거는 `../../data/cases.jsonl`, `../../data/evidence.jsonl`, `../../data/split_manifest.json`이다. 원시 질의별 검색 결과는 `calibration/`, `holdout/`, `diagnostics/`, 엔진 결과는 `serving/`, 장치 결과는 `hardware/`에 있다. 기존 테스트 폴더는 Docker volume과 import 경로에 포함하지 않는다. 새 입력은 덮어쓰지 않으며 재실행은 새 버전·새 출력 폴더를 만들어야 한다.', '']
    report=OUT/'실행결과.md';report.write_text('\n'.join(text),encoding='utf-8')
    inventory={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_inventory.json'}
    (OUT/'artifact_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(str(report))

if __name__=='__main__':main()
