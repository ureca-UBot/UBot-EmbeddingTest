# 새 FAQ 500쌍 × 10개 · Ollama/vLLM

**현재 상태: 미실행·설계 재정렬 중.** 아래의 JSON 기반 500쌍 구성과 11개 구조 자동 실행 설명은 이 폴더를 준비했을 당시의 이력이다. 이후 사용자가 원천을 기존 3,000행 Excel로 변경하고 README에서 벗어난 설계를 지적했다. 이 폴더의 runner는 재개하지 않는다. 30,000행 실행은 중지 상태로 보존한다.

현재 기준은 사용자가 제공한 최신 README 원문 (로컬 전용: `../data/requested_readme.md`)이다. 이후 직접 지시한 기존 자산 재사용 제한, 폴더 분리, Excel 원천 선택, 분야 균형, 규모 변경, Ollama/vLLM 우선 측정과 추가 로그 필드는 원문 위에 적용되는 사용자 변경 사항이다. 이 변경 사항을 내가 추가한 알고리즘 가정과 구분한다.

| 항목 | README 기준 | 현재 구현의 차이 / 수정 기준 |
|---|---|---|
| A 입력 | `embed(faq.question)` | 브랜드·카테고리·Question 접두어를 넣었다. 기본 A는 원문 질문만 사용한다. |
| B 입력 | `embed(question + answer)` | 기본 B는 질문과 전체 답변을 고정 구분자로 결합한다. 추가 metadata는 별도 실험이다. |
| C 후보 | 각 view Top-K의 union + dedupe | max cosine 재정렬 후 20개로 잘랐다. 기본 C는 합집합을 보존하고 각 view K와 실제 후보 수를 기록한다. budget 축소는 별도 실험이다. |
| D 입력 | query↔FAQ question / query↔FAQ question+answer | 답변 창별 점수의 최대값 집계를 추가했다. 기본 D는 전체 FAQ 입력이다. 입력 한도 초과를 먼저 확인하고 창 집계가 필요하면 별도 구조로 분리한다. |
| E/F·Hybrid | A~D 실패 분석 이후 필요할 때 | 11개 구조에 항상 포함했다. 초기 실행은 A~D이며 후속 실험의 필요성을 실패 사례로 판단한다. |
| 평가 유형 | 핵심·문맥·NO_FAQ·특수 조건·반복을 분리 | 긍정 변형 중심이고 NO_FAQ 0건이었다. Excel 원천 분류와 검증 목적을 보존하고 README 유형/태그를 별도로 매핑한다. |
| GT | 필요한 사실과 acceptable FAQ 대안 | canonical FAQ 지원 묶음만 사용했다. 이를 검수된 FactRecall/AllFactsHit이라고 부르지 않는다. 필요한 사실 GT를 먼저 정리한다. |
| 한영 쌍 | 의미·수치·조건·GT 고정, 언어만 변경 | 일부 변형은 언어 외 문장 형태도 바뀌었다. 의미 동등성 쌍은 같은 GT와 고정된 조건으로 관리한다. |
| 추가 원문 질의 | 한영 쌍 관리 예시, 모든 원문 추가 필수는 아님 | 모든 부모에 원문 비교 질의를 추가했다. 원문 추가는 필요한 한영 대응 쌍의 입력으로만 관리하고 본 평가 행 수와 구분한다. |
| 임계값 | 구조 선정 후 NO_FAQ와 score/margin으로 calibration | null 분포 진단만 계산했다. 이를 완료된 threshold calibration으로 표시하지 않는다. |
| Holdout·반복 | 구조 선정 후 별도 최종 평가 / 반복은 독립 문항 아님 | 아직 미실행이며 이후 단계로 남긴다. 반복을 변형 문항으로 증식해 평균에 넣지 않는다. |
| Serving | 구조 검증 이후 latency/concurrency/memory/failure 비교 | 사용자 요청으로 엔진별 구조 비교를 먼저 수행했으나 현재 값은 순차 HTTP 진단이다. README의 serving 부하 시험 완료로 표시하지 않는다. |
| 로그 | query별 baseline/reranked/rank delta 및 개선·악화·miss | 사용자가 추가 요청한 후보별 16개 필드는 유지한다. 기본 D는 window_count=1, winning_window_index=0이며 전체 문서를 winning_window_text에 기록한다. 별도 window 실험은 구분한다. |

원천 변경의 기준 파일은 `FAQ_RAG_15개항목_각200건_총3000건_피드백수정본(1).xlsx`다. 바로 전 500쌍 지시와 원본 3,000질문 직접 실행의 구분은 질문이 남아 있으므로 확정된 실행 입력이라고 주장하지 않는다. 검증 목적·FAQ corpus·GT·A/B/C/D 입력과 후보 계약을 README와 대조한 후 다음 실행을 준비한다.

사용자 요청에 따라 새 faqs.jsonl 기반 3,000쌍에서 **500쌍 전체 변형 10개, 총 5,000질문**을 선정한다. 원문 비교 질문 500개는 별도로 실행하며 본 평가 평균에 합산하지 않는다. 검색 코퍼스는 원래 3,246 FAQ를 그대로 유지한다.

500쌍 선정은 검색 점수나 정답률을 보지 않고 수행했다. 244개 브랜드·언어·카테고리 구간마다 최소 한 쌍을 포함한 뒤, 브랜드/언어 및 카테고리의 원래 개수 비율에 가장 가까운 정수 배분을 사용했다. 구간 내 선택은 고정 FAQ 해시 순서다. 구간 보존 때문에 비율이 정확히 같지는 않다. 선정 감사 (로컬 전용: `data/selection_audit.json`)에 오차를 기록한다. 모든 primary variation 유형과 기존 validation tag를 보존했다. 일반 3,238행, 조건 1,262행, 복합 500행이며 한영 대응어 변형은 442행이다.

축소 전 30,000행 실행은 ../faqs_engine_ollama_vllm_30000 에 중지 상태로 보존한다. 코드·고정 모델 가중치만 재사용하며 질문·문서 벡터, reranker logits, native 점수, 임계값은 새로 계산한다. 이전 Excel 테스트는 사용하지 않는다.

Ollama → vLLM 순서로 엔진당 A/B/C, BGE/Jina×질문/질문+답변 네 조합, sparse/hybrid와 full/pool MaxSim 대조군까지 11개 구조를 실행한다. 각 dense 엔진은 기동→readiness→실제 HTTP 질의→종료→HTTP 폐쇄/컨테이너 0 확인 후 보조 모델을 실행한다. TEI는 제외한다.

Dense는 8,192토큰, reranker는 1,024토큰 한도이며 truncation을 사용하지 않는다. Ollama는 physical batch를 입력 길이에 따라 2,048/4,096으로 설정한다. Reranker 질문+답변은 후보 FAQ의 모든 원문 보존 창을 처리하고 최고 raw logit으로 합친다. 동점 창은 먼저 나온 창을 선택한다. 후보 budget은 20개다.

Ollama GGUF F16과 vLLM HF FP32의 차이도 결과에 포함한다. D는 공통 HF/CUDA FP16, E/F는 공통 FlagEmbedding/CUDA FP32다. 엔진 자체의 reranker/sparse/multi-vector API 측정으로 해석하지 않는다.

후보 로그 candidate_log.jsonl은 query, expected_faq_id(필수 canonical FAQ 배열), faq_id, retrieval_rank/cosine, retrieval_a_rank/cosine, retrieval_b_rank/cosine, rerank_rank/score, rank_delta, window_count, winning_window_index/text/score를 기록한다. A/B 순위는 전체 3,246 FAQ의 1-based 순위, retrieval_rank는 C 후보 내 1-based 순위, winning_window_index는 해당 FAQ 내 0-based 순서다. rank_delta = retrieval_rank - rerank_rank이고 양수 개선, 음수 하락, 0 유지다. 조합별 본 평가 100,000 후보행과 원문 비교 10,000 후보행이다.

GT는 독립 검수된 원자 사실이 아닌 FAQ 지원 묶음이다. NO_FAQ가 없어 거절 threshold/margin은 null이며 F1을 적합하지 않는다. 구조 선정은 같은 긍정 데이터의 순위 진단으로, Holdout이나 운영 적용 확정 결과가 아니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1
```

진행은 outputs/run-v1/run_status.json, 최종 비교는 outputs/run-v1/새FAQ_Ollama_vLLM_비교.md, 완료 증빙은 completion_audit.json이다. 원천 문서 내용은 데이터로만 처리한다.
