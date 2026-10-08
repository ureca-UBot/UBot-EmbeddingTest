# v4 네 데이터셋 실제 엔진 테스트

입력은 상위 `fair_eval_v4/datasets`의 existing·kt·skt·lgu만 사용한다. 사용자 요청에 따라 독립 의미 검수 전 진단 모드로 실행하며 데이터셋·정답·목적은 변경하지 않는다. 기존 벤치마크 실행기나 v1~v3 출력은 읽지 않는다. 기존 Docker 이미지와 캐시된 모델 가중치만 재사용한다.

전체 결과 (로컬 전용: `outputs/run-v4/comparison.md`) · 진행 상태 (로컬 전용: `outputs/run-v4/dense_status.json`) · 후속 단계 (로컬 전용: `outputs/run-v4/remaining_status.json`) · 완료 감사 (로컬 전용: `outputs/run-v4/completion_audit.json`)

**2026-10-07 실행 완료:** 네 데이터셋 × 두 엔진, 각 13개 구조를 계산했다. 총 1,918질문을 각 엔진에 전부 입력했고 후보 로그 104개·1,216,070행을 전수 검사했다. 별도 부하 2,560요청은 실패 0건, 반복 800요청에서는 A 검색 순위 변화가 없었다. 초기 긴 문서 요청 오류 1건과 토크나이저 API 오류는 수정 후 재개했으며 `outputs/run-v4/incidents`에 기록을 남겼다. 컨테이너·포트 최종 종료 (로컬 전용: `outputs/run-v4/final_shutdown.json`)도 확인했다. 리랭커와 sparse/multi는 아래에 설명한 공통 보조 모델 측정이며, 임계값 피팅·생성 LLM 평가는 포함하지 않았다.

## 고정 범위

- 총 사용자 질문 1,918개를 Ollama와 vLLM에 각각 전부 입력한다. 각 데이터셋 전용 FAQ 코퍼스를 question / question+answer로 인코딩한다.
- 엔진·데이터셋별로 기동 → 실제 HTTP 요청 → 종료 → 컨테이너 및 포트 종료 확인을 반복한다. 엔진 포트는 127.0.0.1:11436 / 8002다.
- BGE-M3를 고정하고 Ollama GGUF F16, vLLM HF float16으로 실행한다. 가중치 포맷과 구현 차이는 남아 있으므로 완전히 같은 모델 바이너리의 실행 속도라고 표현하지 않는다.
- A 질문 Dense, B 질문+답변 Dense, C 원시 합집합과 최종20 제한을 따로 평가한다.
- 공통 BGE/Jina 리랭커 × 질문/질문+답변 4조합. 각 엔진에서 실제로 얻은 C 후보를 사용한다. 동일 쿼리·문서 쌍의 공통 리랭커 점수는 재사용한다.
- Sparse 및 ColBERT는 공통 FlagEmbedding native head로 해당 코퍼스 전체를 계산한다. 엔진 자체의 sparse/multi-vector API 측정은 아니다. Dense+sparse RRF의 dense 부분만 엔진별 결과다.
- 리랭커 입력은 1,024토큰 쌍 제한, 문서 구간 겹침 128토큰, 최대 raw logit으로 문서 점수를 정한다. 입력을 잘라 버리지 않는다. 구간 번호는 0부터, 순위는 1부터다.
- 동시성 1/4/8/16/32, 엔진별 같은 64질문, 요청당 1문장. 웜업 이후 짧은 진단 부하이며 운영 SLA가 아니다.
- 동일 입력 반복은 데이터셋마다 10질문 × 10회, 질문 Dense(A)의 순위·점수·Context hash를 기록한다.
- 문서 공격은 데이터셋마다 10개 독립 패치의 B 검색과 Context 구성까지 평가한다. 생성 LLM은 실행하지 않는다.
- NO_FAQ·부분 답변·개인 조회·명확화는 별도 보고한다. 임계값 피팅은 수행하지 않는다. holdout 포함 사전 정의 행렬의 진단이며, 결과를 보고 조정하면 이후 새로운 최종 holdout이 필요하다.

## 실행

```powershell
docker compose run --rm --no-deps validate run_v4.py preflight
./run_dense.ps1
./run_remaining.ps1
```

`run_remaining.ps1`은 dense 단계가 진행 중이면 완료를 기다린 뒤 실행한다. 입력 해시가 바뀌면 실행을 거부한다. 성공한 단계는 체크포인트로 재개하며 과거 실패 요청은 지우지 않는다. 모델·엔진은 동시에 GPU를 사용하지 않는다.

## 로그

`outputs/run-v4/<dataset>/<engine>/<structure>/candidate_log.jsonl.gz`에 아래 필드를 저장한다.

```text
query, model_query, expected_faq_id, expected_faq_ids, required_facts
faq_id, retrieval_rank, retrieval_cosine
retrieval_a_rank, retrieval_a_cosine, retrieval_b_rank, retrieval_b_cosine
rerank_rank, rerank_score, rank_delta
window_count, winning_window_index, winning_window_text, winning_window_score
dataset_id, engine, structure, question_id, split
```

`expected_faq_id`도 호환성을 위해 전체 허용 정답 ID 배열을 보존한다. 사실별 대안은 OR, 필요한 여러 사실은 AND다. `rank_delta = retrieval_rank - rerank_rank`이다. 리랭크하지 않은 구조의 리랭커·구간 필드는 null이다. Sparse/MaxSim/RRF를 cosine으로 표시하지 않으며 별도 `retrieval_representation_score`를 사용한다.

## 검사

`test_runner.py`는 통신사 혼합 거부, 전체 문서 구간 피복, 순위 변화, 복수 정답, 합집합 예산, HTTP 입력·정규화를 검증한다. `test_native_math.py`는 음수 유사도에서 padding이 최대값으로 선택되지 않는지, sparse 내적, 두 리랭커의 입력 쌍·padding과 공식 토크나이저 출력의 일치를 검사한다. 실제 native 계산은 공식 점수 함수와 대조한다. `compare_v4.py`는 모든 후보 로그를 읽어 스코프·정답·순위·요청 필드·구간 값을 전수 검사한다.
