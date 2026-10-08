# 새 FAQ · Ollama와 vLLM 전체 구조 비교

사용자가 새로 제공한 `faqs.jsonl` 기반 [30,000행](../faqs_based_30000/README.md)을 실행한다. 기존 Excel 기반 평가셋·이전 변형 질문·기존 벡터·점수·임계값을 사용하지 않는다. 기존 실행 폴더는 보존하고 이번 코드·입력·결과는 이 폴더에 분리한다.

본 평가 30,000행과 별도 원문 기준 질문 3,000행을 Ollama → vLLM 순서로 실행한다. 본 평가의 일반 19,358행·조건 7,642행·복합 3,000행을 나눠 집계한다. 원문 기준 3,000행은 한영 비교의 대응 입력이며 본 평가 평균에 합산하지 않는다. 3,246개 canonical FAQ를 검색한다. TEI는 이번 요청에서 실행하지 않는다.

`run_all.ps1`은 엔진 기동→readiness→실제 HTTP 전체 실행→엔진 종료→HTTP 폐쇄·활성 컨테이너 0 확인→해당 엔진 후보의 보조 모델 실행 순서다. Ollama 세트가 끝난 뒤 vLLM을 켠다. GPU 메모리를 확보하기 위해 dense 엔진과 보조 모델을 동시에 상주시키지 않는다.

## 입력과 비교 구조

모델 입력은 `query` 문자열이다. GT·FAQ ID·태그·원천 답변은 query에 이어 붙이지 않는다. 검증과 각 단계 진입 때 `data/manifest.json`의 해시를 확인한다. 질문은 최대 225토큰, 전체 질문+답변은 최대 3,570토큰이다. Dense 한도는 8,192토큰이며 truncation을 비활성화한다.

| 구조 | 실행 방식 |
|---|---|
| A | 브랜드·카테고리·질문 dense |
| B | 같은 metadata + 전체 질문·답변 dense |
| C-M20 | A/B Top-20 union에서 FAQ ID 중복 제거 후 max cosine으로 20개 |
| D-BGE × 질문/질문+답변 | C의 20개 FAQ를 BGE reranker로 재정렬 |
| D-Jina × 질문/질문+답변 | C의 20개 FAQ를 Jina reranker로 재정렬 |
| E-sparse | 공통 BGE-M3 native sparse 대조군 |
| E-dense-sparse-RRF-M20 | 해당 엔진 B dense와 native sparse RRF, FAQ 20개 |
| F-full | 공통 BGE-M3 native full-corpus MaxSim 대조군 |
| F-dense-pool-M20 | 해당 엔진 B dense Top-20 후보를 native MaxSim으로 재정렬 |

엔진당 11개 구조다. D의 질문+답변은 source-preserving 분할 문서 전부를 점수화하고 FAQ별 최고 raw logit으로 합친다. 문서 창은 최대 256토큰, 답변 겹침은 최대 32토큰이다. 창을 별도 FAQ로 세지 않으며 C의 FAQ budget은 계속 20개다. D 입력 한도는 1,024토큰이고 실제 pair 길이를 검사한다. 후보 FAQ쌍은 조합당 660,000개이며 질문+답변의 실제 창쌍 수는 후보에 따라 달라져 별도로 기록한다. 이는 본 평가 600,000개와 원문 기준 60,000개를 포함한 수다.

Ollama는 캐시된 BGE-M3 GGUF F16, vLLM은 고정 HF revision의 FP32 모델을 사용한다. 포맷·정밀도 차이도 결과에 포함된다. D는 공통 CUDA FP16이며 FP32/FP16 표본 raw-logit 차이를 저장한다. E/F는 공통 FlagEmbedding CUDA FP32다. D/E/F를 엔진 자체 API로 서빙한 결과라고 해석하지 않는다. 각 엔진 세트에서 다시 계산하며 다른 엔진의 보조 inference 결과를 재사용하지 않는다.

Ollama의 BERT embedding은 전체 입력이 physical batch에도 들어가야 한다. 2,048토큰 이하 입력은 `num_batch=2048`, 긴 문서는 `num_batch=4096`을 사용하며 `num_ctx=8192`, `truncate=false`를 유지한다. 최장 3,570토큰 문서를 자르지 않고 처리한 진단은 `outputs/run-v1/engines/ollama/physical_batch_diagnostic.json`에 기록한다. 최초 긴 문서 HTTP 400과 같은 실행의 체크포인트 재개도 보존한다. [Ollama v0.35.0 구현](https://github.com/ollama/ollama/blob/v0.35.0/server/routes.go)의 `NumBatch` 및 `Truncate` 처리와 실제 엔진 로그를 확인했다.

## 지표와 임계값

GT는 독립 검수된 원자 사실이 아니라 canonical FAQ 지원 묶음이다. SourceHit·SourceCoverage·AllSourcesHit·MRR를 행 평균·부모 평균·연결 그룹 평균으로 구분한다. 복합 질문은 두 FAQ를 모두 검색해야 complete로 집계한다. 같은 질문·답변의 여러 원천 ID는 대체 ID이며 여러 필수 사실로 세지 않는다. 한영 변형은 실제 실행한 원문 기준 질의와 비교한다. V05에는 추가 문장 변형이 포함될 수 있어 모든 변화를 언어 하나의 인과효과로 해석하지 않는다.

**NO_FAQ가 0행이므로 거절 threshold/margin을 적합하지 않는다.** `policies.json`은 threshold/margin null과 score/margin 분포만 저장한다. NO_FAQ precision/recall/F1을 꾸며 계산하지 않는다. 순위 지표로 뽑는 구조는 같은 긍정 데이터의 진단용 후보이며 Holdout/운영 적용 확정 결과가 아니다.

반복 실행, 실제 장애 로그, 대화 rewrite, 생성 모델, 개인 API, 문서 공격, 원자 사실 검수, NO_FAQ 음성 문항, Holdout은 이번 요청의 실행 범위에 포함하지 않는다. 원천의 브랜드·카테고리·언어 및 검증 태그는 개별 결과에 보존한다.

## 실행과 확인

```powershell
docker compose --profile validation run --rm validate scripts/test_contracts.py
docker compose --profile validation run --rm validate scripts/run_experiment.py --stage validate
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1
```

`scripts/prepare_inputs.py`는 새 원천 데이터만 어댑터 형식으로 동결한다. 이미 실행 중이거나 완료한 데이터를 다시 준비하지 않는다. 중간 API vector·reranker logits·native 점수와 진행 상태는 engine별 `cache`에 저장한다. 완료한 구조는 덮어쓰지 않는다.

진행 상태는 run_status.json (로컬 전용: `outputs/run-v1/run_status.json`), 단계 로그는 `outputs/run-v1/logs/`에서 확인한다. 최종 보고서는 `outputs/run-v1/새FAQ_Ollama_vLLM_비교.md`이며 완료는 `completion_audit.json`으로 검증한다. 엔진별 `readiness.json`, `serving_model_runtime.json`, `embedding_requests.jsonl`, `stop_verification.json`에 실제 기동·입력·종료 증빙을 남긴다. 요청 latency는 순차 가변 batch 진단이며 동시 부하 성능 시험은 아니다.

각 D 구조의 `candidate_log.jsonl`에는 query, expected_faq_id, faq_id, retrieval_rank, retrieval_cosine, rerank_rank, rerank_score, rank_delta, window_count, winning_window_index, winning_window_text, winning_window_score를 후보 FAQ별로 기록한다. `expected_faq_id`는 복합 질의의 필수 FAQ를 보존하는 배열이다. Retrieval 순위는 C 후보 20개 안의 1-based 순위이며 cosine은 두 dense view 중 높은 값이다. `rank_delta = retrieval_rank - rerank_rank`는 양수 개선, 음수 하락, 0 유지다. Winning window index는 해당 FAQ 내 0-based 순서이며 동점은 먼저 나온 창을 선택한다. 질문 view는 창 한 개다. 질문+답변 view의 실제 winning window는 점수 계산 시 저장한다. 본 평가 600,000 후보행과 별도 원문 기준 60,000 후보행이며 정의와 SHA256은 `candidate_log_schema.json`에 남긴다.

추가 `retrieval_a_rank`, `retrieval_a_cosine`, `retrieval_b_rank`, `retrieval_b_cosine`은 A 질문 view와 B 질문+답변 view 각각의 값이다. A/B 순위는 후보 pool 안의 순위가 아니라 전체 3,246개 FAQ 중의 1-based 순위다. Top-20 밖의 후보도 실제 전체 순위를 기록한다. 같은 cosine은 코퍼스 순서로 정렬한다.
