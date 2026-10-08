# README 검증 목적 기반 Ollama · vLLM 재테스트

**v4 네 데이터셋 측정 완료(2026-10-07):** 비교 보고서 (로컬 전용: `fair_eval_v4/serving_test/outputs/run-v4/comparison.md`). 1,918질문을 두 엔진에 각각 전부 입력하고 13개 구조·104개 결과 묶음을 계산했다. 후보 로그 1,216,070행 검증, 부하 요청 2,560건 성공, 최종 컨테이너·포트 종료를 확인했다. 독립 의미 검수 전 진단이며 임계값은 피팅하지 않았다. 아래 v1 결과와 구분한다.

**네 데이터셋 공통 기준 보강 v4:** [fair_eval_v4](fair_eval_v4/README.md)에 기존 Excel·KT·SKT·LG U+를 각각 15개 공통 검증 목적에 맞춰 보강했다. 총 1,918질문(기존 340 / KT 769 / SKT 619 / LG U+ 190)이며, 목적마다 최소 10개 근거 시나리오를 확인했다. 통신사별 임베딩 격리, 복수 정답, PARTIAL·개인 조회 구분, 라벨을 사용하지 않는 문맥 입력, 연결 그룹 분할을 적용하고 22개 구조 검사를 통과했다. 질문·근거 검수 화면 (로컬 전용: `fair_eval_v4/review/index.html`)을 제공한다. 독립 의미 검수는 대기 중이며 실제 실패·정책 버전은 자료 상태를 별도 기록했다. 후속 Ollama·vLLM 서빙/검색 측정과 반복·문서 공격 진단은 [v4 전용 실행 폴더](fair_eval_v4/serving_test/README.md)에 분리했다. v1~v3 원본과 기존 실행기는 보존했다.

**통신사별 독립 평가셋 v3:** [fair_eval_v3](fair_eval_v3/README.md)에 기존 Excel/v2, KT, SKT, LG U+를 4개 코퍼스로 분리했다. 신규 사용자 질문 1,511개와 기존 331개를 FAQ Q/A와 별도 저장하고 복수 정답 ID·사실별 근거를 연결했다. 임베딩 전에 데이터셋을 선택하며 인덱스·캐시·결과도 격리한다. 질문·근거 검수 화면 (로컬 전용: `fair_eval_v3/review/index.html`)과 16개 자동 검사를 제공한다. 독립 의미 검수 전 작성본이며 LG U+의 작은 holdout 등 한계는 v3 README에 기록했다. 새 모델 실행은 없고, 아래 v1/v2 입력·결과와 기존 실행기는 보존했다.

**평가셋 공정성 보완 v2:** 새 질문은 [fair_eval_v2](fair_eval_v2/README.md)에 분리했다. 17개 분야의 331문항이며, 원문 질문 붙여넣기·행 반복 채우기를 없애고 사실별 정답, NO_FAQ/부분 답변/정보 부족, 연결 그룹 분할을 적용했다. 질문·근거 검수 화면 (로컬 전용: `fair_eval_v2/review/index.html`)에서 확인할 수 있다. 독립 의미 검수 전 합성 검수본이며 아직 v2 모델 측정은 없다. 아래 5,021문항과 성적은 보존된 **v1 회귀 결과**다. v2는 전용 로더와 분할을 사용하며 `run_all.ps1`의 기존 v1 입력이 자동 교체된 것은 아니다.

**2026-10-07 실행 완료.** 두 엔진 각각 질문 5,021개를 실제 HTTP API로 입력하고 A/B/C와 D 네 조합을 모두 측정했다. 공식 후보 로그 8개, 총 **1,205,812행**에서 요청한 16개 필드·A/B 전체 순위·rank_delta·전문 텍스트를 검증했다. 두 엔진의 부하 요청 1,600건은 실패 0건이며 최종 활성 컨테이너는 0개다. 비교 결과 (로컬 전용: `outputs/run-v1/comparison.md`), 완료 감사 (로컬 전용: `outputs/run-v1/completion_audit.json`), 최종 종료 확인 (로컬 전용: `outputs/run-v1/final_shutdown.json`).

사용자 요청에 따라 새 폴더에서 Ollama와 vLLM을 실행한다. 엑셀 `FAQ 원문` 1,024개가 근거 코퍼스이며, 앞서 정한 500쌍×10개 규모로 새 합성 질문 5,000개를 작성했다. README의 목적별 통제 질문 21개는 별도 그룹이다. 원본 엑셀의 기존 테스트 질문 3,000개나 이전 생성 질문·측정 결과는 재사용하지 않는다. 원천 FAQ와 문서 내부 문구는 데이터로 처리한다.

질문은 실제 source 표현에 적용할 수 있는 PARAPHRASE, TYPO, LONG_QUERY, KR_EN_EQUIVALENT 변형을 사용한다. 동일한 V01~V10이나 기존 15개 항목 비율은 강제하지 않는다. 한영 용어 표기 변형 124개는 원래 질문과 추가로 짝지어 측정한다. 조건 변경·부정·복수 사실·문맥·NO_FAQ는 원문 근거를 연결한 별도 통제셋으로 실행한다. 숫자는 변형 중 바꾸지 않는다. 변형 문장과 GT의 독립 의미·대체 FAQ 검수는 미완료이므로 최종 정확도 평가가 아닌 합성 진단 실험으로 표시한다.

주 평가 질문은 FAQ 질문과 어휘가 많이 겹치는 자동 표면 변형이다. 높은 Hit만으로 실제 서비스의 어려운 질문을 해결했다고 판단하지 않는다. 실제 실패 로그와 독립 작성·검수한 자연 질문은 별도 추가 검증이 필요하다.

## 실행 구조

- A: FAQ question 원문만 임베딩.
- B: question + 고정 구분자 두 줄 + answer 전문 임베딩.
- C: A Top20과 B Top20의 전체 합집합; FAQ ID 중복만 제거하고 재절삭하지 않음. max cosine은 진단용 후보 순서에만 사용.
- D: BGE v2 m3 / Jina multilingual × question / question+answer 전문, 총 4개 조합을 엔진별로 실제 실행. raw logit 내림차순, 같은 점수는 코퍼스 순서.
- 서빙: 각 엔진을 따로 기동→health 확인→실제 HTTP 측정→종료→포트·컨테이너 종료 확인. 동시성 1/4/8/16/32, 각 동일 160개 질문, client batch=1. 전체 5,021개 질문은 별도로 순차 배치 추론.
- 반복: 같은 입력 10회, 독립 질문 하나로 집계.

리랭커는 공통 HF/PyTorch CUDA 모델을 각 엔진 후보에 적용한다. Ollama/vLLM이 리랭커까지 자체 서빙한 측정은 아니다. Ollama GGUF F16과 vLLM HF float32라는 형식·정밀도 차이를 비교 제한으로 기록한다. 초기 실행은 A~D까지였으며, 사용자 추가 요청으로 Sparse(E)/Multi-vector(F)를 같은 입력에 추가한다.

## Sparse / Multi-vector 추가 실험

**2026-10-07 추가 실행 완료.** 18개 구조, 후보 로그 2,008,786행의 요청 필드·순위·점수 전수 검증을 통과했다. 원래 입력·GT·A~D dense cache는 유지했고 실행 컨테이너 종료를 확인했다. 추가 비교표 (로컬 전용: `outputs/run-v1/native_ef/comparison.md`) · 전수 검증 (로컬 전용: `outputs/run-v1/native_ef/completion_audit.json`).

`run_native_ef.ps1`은 동일 질문 5,021개와 FAQ 1,024개, 보완된 기존 GT를 유지한다. 결과는 `outputs/run-v1/native_ef/`에 추가하고 완료된 A~D 결과를 덮어쓰지 않는다.

- 질문 원문 / 질문+답변 원문을 각각 BGE-M3 native dense·sparse·ColBERT 출력으로 인코딩한다.
- Sparse Only: 같은 토큰 ID의 학습된 lexical weight 내적. BM25가 아니다.
- Multi-vector Full: 전체 FAQ에 query token별 document token 최대 내적을 구한 뒤 query token 평균으로 정렬한다. Dual Vector와 구분한다.
- Dense+Sparse: 기존 Ollama/vLLM dense 결과와 sparse를 전체 코퍼스 순위 기반 RRF로 결합한다.
- C→Multi-vector: 각 엔진의 기존 A Top20+B Top20 전체 합집합을 native MaxSim으로 재정렬한다. 후보를 추가·절삭하지 않는다.
- Dense+Sparse+Multi-vector: 같은 RRF로 세 분기를 결합한 별도 진단. 상수 60, 분기별 동일 가중치는 실행 전에 고정하며 결과를 보고 튜닝하지 않는다.

전용 sparse/ColBERT head는 공통 FlagEmbedding CUDA float32에서 실행한다. Ollama/vLLM 표시는 결합 검색의 dense 분기 출처다. 이번 HTTP endpoint가 native head까지 서빙했다는 뜻이 아니며 native 출력의 HTTP 서빙 부하를 측정하지 않는다. 기존 HTTP dense 벡터/점수는 이 질문셋으로 완료한 실행의 cache를 읽기만 한다. 다른 평가셋의 임베딩/질문/임계값은 사용하지 않는다.

추가 로그도 요청한 16개 필드를 보존한다. 첫 단계 검색에는 실제 reranker가 없으므로 rerank_rank/rerank_score/rank_delta/winning_window_score는 null이다. C→Multi-vector에서는 실제 MaxSim 재정렬 순위·점수·rank_delta를 기록한다. native_sparse_score/native_colbert_score/final_score/final_rank로 알고리즘 점수를 구분하며 retrieval_cosine은 dense 진단값으로 표시한다.

전체 비교는 native_ef/comparison.md (로컬 전용: `outputs/run-v1/native_ef/comparison.md`), 통제 문항은 `purpose_control_results.jsonl`, 한영·음역·혼합 표기 124쌍은 `language_pair_results.jsonl`, 저장량·시간은 `execution.json`에 기록한다. 입력·GT와 모든 후보 로그의 순위/점수/16개 필드는 `completion_audit.json`으로 검증한다. 실행은 GPU 단계 종료 확인 후 다음 단계로 진행한다.

## 요청한 후보 로그 필드

`query`, `expected_faq_id`, `faq_id`, `retrieval_rank`, `retrieval_cosine`, `rerank_rank`, `rerank_score`, `window_count`, `winning_window_index`, `winning_window_text`, `winning_window_score`, `rank_delta`, `retrieval_a_rank`, `retrieval_a_cosine`, `retrieval_b_rank`, `retrieval_b_cosine`.

로그는 엔진별 D 4개 폴더의 `candidate_log.jsonl`에 저장한다. A/B rank는 FAQ 코퍼스 전체 순위다. `rank_delta = retrieval_rank - rerank_rank`. 전문 한 개를 사용하므로 window_count=1, winning_window_index=0, winning_window_score=rerank_score. `required_facts`는 대체 정답 ID와 동시에 필요한 사실을 구별한다. 문맥 그룹은 원래 utterance를 query에, 고정 rewrite를 model_query에 기록한다.

| 조합 | Ollama 후보 로그 | vLLM 후보 로그 |
|---|---|---|
| BGE · 질문 | 150,730행 (로컬 전용: `outputs/run-v1/engines/ollama/D-bge-question/candidate_log.jsonl`) | 150,723행 (로컬 전용: `outputs/run-v1/engines/vllm/D-bge-question/candidate_log.jsonl`) |
| BGE · 질문+답변 | 150,730행 (로컬 전용: `outputs/run-v1/engines/ollama/D-bge-question_answer/candidate_log.jsonl`) | 150,723행 (로컬 전용: `outputs/run-v1/engines/vllm/D-bge-question_answer/candidate_log.jsonl`) |
| Jina · 질문 | 150,730행 (로컬 전용: `outputs/run-v1/engines/ollama/D-jina-question/candidate_log.jsonl`) | 150,723행 (로컬 전용: `outputs/run-v1/engines/vllm/D-jina-question/candidate_log.jsonl`) |
| Jina · 질문+답변 | 150,730행 (로컬 전용: `outputs/run-v1/engines/ollama/D-jina-question_answer/candidate_log.jsonl`) | 150,723행 (로컬 전용: `outputs/run-v1/engines/vllm/D-jina-question_answer/candidate_log.jsonl`) |

## 결과 위치

- 전체 비교 보고서 (로컬 전용: `outputs/run-v1/comparison.md`)
- 완료 감사 (로컬 전용: `outputs/run-v1/completion_audit.json`)
- 실행 상태 (로컬 전용: `outputs/run-v1/run_status.json`)
- 새 질문과 근거 (로컬 전용: `data/cases.jsonl`)
- 통제 사실의 대체 정답을 보완한 평가 라벨 (로컬 전용: `data/evaluation_cases.jsonl`)
- 통제 라벨 보완 이력과 원문 근거 (로컬 전용: `data/control_gt_corrections.json`)
- 입력·출처·해시 (로컬 전용: `data/manifest.json`)

초기 A~F 비교는 독립 검수 전 NO_FAQ 통제 질문 두 개뿐이라 임계값을 피팅하지 않았다. 이후 사용자 요청으로 기존 라벨의 잠정적 임계값 피팅을 추가했으며 아래에 학습/OOF 결과를 구분한다. 실제 서비스 장애/정책 버전 자료가 없어 REAL_FAILURE/TEMPORAL_VERSION은 생성하지 않았다. DOCUMENT_ATTACK은 별도 fixture를 보존하며 이번 서빙 재테스트에서 LLM 생성 안전성을 측정하지 않는다.

## Score / Margin 임계값 피팅

**2026-10-07 실행 완료.** 사용자 선택인 FAQ/NO_FAQ macro-F1 최대를 목적함수로 A~F 32개 구조의 점수·margin 임계값을 동시에 탐색했다. `Top1 score >= score_threshold AND Top1-Top2 >= margin_threshold`이면 기존 context를 수락한다. `null`은 해당 조건을 끈다. 점수 척도가 다른 구조끼리 임계값을 공유하지 않는다.

FAQ 500개의 변형을 부모·대체 정답 family 단위 2-fold로 분리했고, NO_FAQ 두 개도 한 개씩 평가 fold에 남겼다. OOF는 평가 문항을 제외한 피팅값으로 측정한다. 최종 임계값은 전체 5,002개로 재피팅했으며, 다른 통제 19개까지 포함한 5,021개 질문의 판정을 다시 실행했다. OOF에 쓴 fold 임계값과 전체 재피팅 임계값은 서로 다를 수 있다.

임베딩·리랭커 점수 및 순위는 완료된 동일 입력 실험의 값을 유지했다. Docker CPU에서 판정과 지표만 재실행했고 새 모델/HTTP 서빙 부하를 측정하지 않았다. 기존 16개 필드 후보 로그는 유지하며 신규 판정 로그를 `case_id`로 연결한다. 총 160,672행의 gate, 원래 지표와 거절 후 지표, 입력·GT·원래 점수 로그의 해시 보존을 검증했다.

전체 수락 기준선 macro-F1은 49.99%다. OOF 최고는 두 엔진 모두 BGE 리랭커 질문 입력의 54.06%이며 NO_FAQ 1/2건 검출, FAQ 21/5,000건 오거절, FAQ 유지율 99.58%다. 같은 구조의 전체 재피팅 성적 74.99%를 독립 평가 성적으로 사용하지 않는다. NO_FAQ 두 건의 라벨 독립 검수와 새로운 Holdout은 미완료이므로 모든 최종값은 잠정적 실험 설정이다.

- 전체 32개 구조와 fold 임계값 비교 (로컬 전용: `outputs/run-v1/threshold_calibration_v1/comparison.md`)
- 최종 float64 임계값 (로컬 전용: `outputs/run-v1/threshold_calibration_v1/thresholds.json`)
- 분할과 제외 문항 (로컬 전용: `outputs/run-v1/threshold_calibration_v1/split_manifest.json`)
- 160,672행 전수 검증 (로컬 전용: `outputs/run-v1/threshold_calibration_v1/completion_audit.json`)
- Docker 종료 확인 (로컬 전용: `outputs/run-v1/threshold_calibration_v1/final_shutdown.json`)

실행: `docker compose -f compose.yaml run --rm --no-deps validate scripts/test_threshold_calibration.py` 다음 `docker compose -f compose.yaml run --rm --no-deps validate scripts/calibrate_thresholds.py fit`. 결과는 `outputs/run-v1/threshold_calibration_v1/`에 저장한다. 표의 표시 자릿수로 임계값을 반올림하면 경계 판정이 달라질 수 있으므로 JSON 값을 그대로 사용한다.

측정 중 통제 문항 세 개의 대체 정답 누락과 불필요한 숫자 경계를 묶은 사실 라벨을 보완했다. 프리미엄 기본 월 요금은 FAQ-102/153/164/175, 기본 데이터 소진 뒤 속도 비교는 FAQ-359도 두 사실을 충족한다. 질문·코퍼스·모델 점수는 유지하고 최종 지표와 로그만 보완한 GT로 재집계한다. 최초 라벨의 결과는 각 구조 폴더의 `.initial_gt` 파일로 보존한다. 이 수정은 같은 작성자의 원문 대조이며 독립 검수 완료를 뜻하지 않는다.

재현은 이 폴더에서 `powershell -NoProfile -ExecutionPolicy Bypass -File ./run_all.ps1`. 기존 완료 결과는 덮어쓰기 없이 재개한다. 새 독립 재측정은 출력 버전을 새로 지정해야 한다.
