# UBot BGE-M3 Retrieval Benchmark 실험 설계

[최신 프로젝트 계획](../../README.md)을 실행 계약으로 구체화한 설계 v2다. BGE-M3를 고정하고 질문 벡터, 질문과 답변 벡터, 두 벡터의 후보 병합, reranker를 차례로 비교한다. A~D의 실패가 남을 때 native Sparse(E)와 ColBERT Multi-vector(F)를 비교한다. 두 3,000건 엑셀은 세부 문항을 만드는 원천 자료로 사용한다. 기존 회귀셋은 보존하되 새 코퍼스와 같은 FAQ ID를 공유한다는 이유만으로 정답을 옮기지 않는다.

품질 구조 선정 → threshold calibration → holdout → Docker 기반 서빙 엔진 비교 → 별도 CPU/GPU 비교 순서로 진행한다. 설계 v2와 실제 실행의 범위는 구분한다. 2026-10-06 실행은 고정 revision과 원천 라벨로 A~F, 동결 정책 holdout, 기존 회귀를 측정한 잠정 실험이다. 독립 검수된 최종 dataset은 아직 없다. 실제 결과와 실행 차이는 실행 보고서 (로컬 전용: `../outputs/retrieval/run-20261006-v1/실행결과_20261006.md`)에 기록한다.

## 1. 고정할 기준과 결정할 변수

| 구분 | 설계 기준 |
|---|---|
| 임베딩 모델 | `BAAI/bge-m3`, dense embedding 고정 |
| 임베딩 런타임 | 기존 `sentence-transformers` 경로 재사용. 모델 revision과 라이브러리 버전 기록 |
| 검색 코퍼스 | 새 엑셀의 `FAQ 원문` 1,024건 전체. FAQ_RAG 985건, MAP_API 15건, USER_INFO 24건 |
| 검색 단위 | FAQ 1건. 초기에는 청킹하지 않음 |
| Query | 동일 문항에서 A~D가 같은 문자열과 query vector를 사용 |
| 유사도 | L2 정규화한 벡터의 cosine similarity |
| 순위 비교 | threshold 없이 먼저 평가 |
| LLM 전달 문서 수 | 기본 `final_n=3`. 1과 5는 저장된 순위의 진단 지표 |
| 초기 비교 변수 | FAQ 표현, 후보 생성 방식, reranker 모델과 입력 표현 |
| 조건부 추가 구조 | BGE native Sparse와 ColBERT, 이후 필요할 때 BM25/hybrid |
| 구조 선정 후 정책 | threshold와 단일 사실 질문의 score margin |
| 서빙 비교 | Ollama / TEI / vLLM, 같은 장비에서 지원 기능과 벡터·순위 일치 확인 후 부하 측정 |
| 마지막 별도 비교 | 선택한 구조와 엔진 후보에서 CPU/GPU 성능 |

운영에 가까운 baseline은 사용자 계획의 `question only → Top-3 → similarity >= 0.75`다. 이 저장소에는 UBot-BE의 실제 구현이 없으므로 운영과의 일치는 구현을 연결할 때 확인한다. 구조 비교 중에는 0.75를 적용하지 않고, 최종 구조 선정 후 현재 정책을 대조군으로 다시 측정한다.

FAQ_RAG 985건만 검색하는 결과는 선택적인 별도 진단이다. 1,024건 전체 검색과 섞지 않는다. 매장 조회와 개인 조회 문서는 검색 경쟁 후보로 남겨 다른 의도의 문서가 FAQ 답변을 밀어내는 문제도 확인한다. 실제 운영이 의도를 먼저 걸러 검색한다면 동일한 필터를 적용한 별도 코퍼스 설정을 만든다.

## 2. 입력 자료와 버전

### 2.1 새 원천 자료

| 자료 | 사용할 내용 | 사용 위치 |
|---|---|---|
| `FAQ_RAG_3000건_사전Context제거_피드백수정본(1).xlsx` | FAQ 원문, 사용자 질문, 검색 정답 집합, 실행 조건, 시나리오 그룹 | 새 코퍼스와 검색 평가 문항의 원천 |
| `FAQ_RAG_15개항목_각200건_총3000건_피드백수정본(1).xlsx` | 같은 질문과 FAQ 원문, 제공 Context, 사실 기준 | 정답 근거 검수와 이후 생성 단계의 대조 평가 |

두 자료의 FAQ 질문과 답변, 실행 ID, 사용자 질문은 동일하다. 이 둘을 합쳐 6,000건으로 세지 않는다. 두 번째 자료의 제공 Context는 검색 입력이나 정답 후보의 강제 주입에 사용하지 않는다.

Context 제거 자료의 검색 적중률 대상은 `예` 1,807행, `아니오` 923행, `조건부` 270행이다. 일반 실제 검색이고 FAQ_RAG를 포함하는 `예` 대상은 1,807행이다. 이들에는 원본 질문 ID 1,708개와 시나리오 그룹 문자열 903개가 있다. 903은 원천 자료의 그룹 수이며 독립성이 검증된 최종 표본 수는 아니다.

### 2.2 기존 회귀 자료

| 회귀 자료 | 기존 코퍼스 | 검색 문항 수 | 처리 |
|---|---|---:|---|
| 기존 60문항 | `model_benchmarks/faq/EmbeddingTestFAQ_통합.xlsx` | 60개 중 FAQ_RAG 57개 | 기존 코퍼스와 함께 보존 |
| 기존 확장 회귀셋 | `reranker/faq/EmbeddingTestFAQ_통합.xlsx` | 전체 238개 중 FAQ_RAG 235개 | 기존 코퍼스와 함께 보존 |

57개는 235개의 기존 문항 부분집합이므로 두 점수를 합산하지 않는다. 두 회귀 코퍼스는 각각 1,000건이다. 새 코퍼스와 공통 ID 1,000개를 대조하면 질문과 답변이 같은 것은 100건이고 다른 것은 900건이다.

회귀셋은 두 경로로 운영한다.

1. **기존 코퍼스 회귀:** 원래 엑셀, GT와 검색 구현을 유지한 재측정. 재사용 코드가 기존 결과를 재현하는지 확인한다.
2. **새 코퍼스 회귀:** 문항과 관련 FAQ의 본문을 대조해 GT를 다시 검수한 이식본. 이식 전후 GT와 이유를 기록한다. ID가 동일해도 본문이 바뀌었으면 자동 승인하지 않는다.

이식할 수 없는 문항은 원래 회귀셋에 남긴다. 새 코퍼스에서 답이 없어졌는지, 질문이 모호해졌는지 기록하며 원본 GT를 덮어쓰지 않는다. 과거 CSV에는 코퍼스 해시가 없으므로 현재 파일과 당시 실행 입력이 같다고 단정하지 않는다.

### 2.3 재현에 필요한 기록

`corpus_version`, `dataset_version`, 원본 파일 SHA-256, 정규화 후 corpus SHA-256, FAQ별 질문과 답변 SHA-256을 기록한다. 버전이 바뀌면 embedding cache와 결과도 별도 디렉터리를 사용한다. 원천 파일과 검수된 데이터는 복사해 보존하고 Downloads의 경로를 장기 실행 계약으로 삼지 않는다.

파일 경로와 확인된 해시는 [원천 자료 점검 기록](../configs/retrieval_source_audit.json)에 정리한다.

## 3. 평가 트랙

| 트랙 | 포함 조건 | 주요 지표 | 제외하거나 분리할 항목 |
|---|---|---|---|
| 핵심 검색 | 일반 실제 검색, 답에 필요한 FAQ 사실 존재, 입력이 독립적으로 해석 가능 | Hit, FactRecall, AllFactsHit, MRR | 개인 조회만으로 답하는 문항, 고정 Context, 통제 검색 |
| 문맥 검색 | 대화 이력 없이는 질문을 해석할 수 없음 | 고정 query 재작성 결과에서 같은 검색 지표 | 정답용 검색질문 기준을 실제 query로 사용하는 실행 |
| 정답 없는 질문 | 코퍼스에 질문의 핵심 답이 없음 | 최종 거절 정책의 NO_FAQ precision, recall, F1와 오탐률 | 조회 데이터가 없다는 이유만으로 NO_FAQ로 분류한 문항 |
| 특수 조건 | HR, EC, CF, 시험 문서 공격 AD 등 | 조건 재현율과 특수 실패 | 일반 검색 평균과 합산 |
| 반복 | RT의 동일 조건 반복 | 원본별 검색 결과 일관성, 순위와 Context 해시 변화 | 동일 입력 10회를 독립 문항으로 집계 |
| 모호한 입력 | 실제 문맥 없이 대상·조건을 확정할 수 없음 | CLARIFY/보류율과 과도한 확정률 | NO_FAQ 또는 일반 순위 평균으로 집계 |

15개 원천 항목 코드를 새 검색 유형과 일대일로 대응시키지 않는다. 예를 들어 NC는 HARD_NEGATIVE의 원천이 될 수 있고 CE는 숫자와 조건 검증의 원천이 될 수 있다. AD의 문서 공격과 OUT_OF_SCOPE는 실행과 채점이 서로 다르므로 분리한다.

MT는 초기 A~D 핵심 비교에서 분리한다. 문맥 검색을 비교할 때 재작성기 버전과 출력 query를 먼저 고정한 다음 모든 검색 방식이 같은 query를 사용한다. 재작성 전후 차이는 별도 실험이며 검색 구조 변경의 효과에 합산하지 않는다.

`expected_generation_status`(ANSWER/PARTIAL/ABSTAIN/CLARIFY/ROUTE)와 `retrieval_status`(FAQ_EXISTS/PARTIAL_FACTS_EXIST/NO_FAQ/UNDETERMINED/NOT_APPLICABLE)를 별도 필드로 저장한다. 개인 금액을 조회하지 않아 ABSTAIN이어도 조회 경로를 설명하는 FAQ는 존재할 수 있다. PARTIAL에는 코퍼스에서 답할 수 있는 하위 사실만 검색 GT로 둔다. 기대 생성 상태는 라벨이며 실제 생성 결과가 아니다.

`track`은 실행 방식, `reporting_cohort`는 집계 분모다. core_general, targeted_condition, contextual, no_faq, clarify, special_corpus, repeat로 나눈다. EXACT_ENTITY/HARD_NEGATIVE/부정·조건 경계 등은 targeted_condition에서 보고하고 일반 평균에 자동 합산하지 않는다. 서로 다른 분모의 평균을 비교하지 않는다.

## 4. 세부 문항 설계

### 4.1 검증 목적과 변형

| 유형 또는 태그 | 변형할 요소 | 보존할 요소 | 검증할 실패 |
|---|---|---|---|
| REAL_FAILURE | 실제 실패 입력을 원형 보존하고 추가 변형은 별도 ID로 작성 | 발생 당시 코퍼스와 로그, 실패 원인 | 서비스 실패가 새 구조에서 해결되는지 |
| EXACT_ENTITY | 상품명 전체와 약칭, 유사 상품명, 띄어쓰기, 숫자와 단위, 조건 주체 | 실제 FAQ에 존재하는 명칭과 정책 | 비슷한 상품이나 다른 조건의 FAQ가 앞서는지 |
| HARD_NEGATIVE | 핵심 조건 하나만 바꾼 대조 질문, 높은 점수의 잘못된 후보 | 질문마다 검수된 정답 근거 | 의미가 가까운 오답과 구분하는지 |
| KR_EN_EQUIVALENT | 같은 의미의 한국어·영어 용어 치환, 영문 표기와 한글 음역, 혼합 표기 | 질문 의도, 수치, 적용 조건, 사실별 GT | 영어와 한국어로 썼다는 이유만으로 다른 의미로 검색하는지 |
| NEGATION | 긍정/부정, 가능/불가, 적용/미적용 | Entity와 나머지 조건 | 부정을 놓쳐 반대 정책 근거를 선택하는지 |
| CONDITION_SCOPE | 명의자/실사용자, 연령, 가입 유형, 지역, 예외 | 정책과 조건 주체 | 적용 범위를 잘못 일반화하는지 |
| NUMERIC_CONDITION | 이상/이하/초과/미만, 기간, 금액, 용량, 단위 | 숫자·단위·비교 연산의 의미 | 경계와 단위를 뒤집는지 |
| COMPARISON | 두 정책·상품의 차이와 비교 기준 | 모든 비교 대상과 필요한 사실 | 한쪽 사실만 회수하는지 |
| MULTI_INTENT | 독립 요구 두 개 이상 | 하위 요청별 의도와 근거 | 하나의 기능 성공을 전체 성공으로 오인하는지 |
| AMBIGUOUS_QUERY | 지시어, 생략, 대상 불명확 | 실제 정보가 부족한 상태 | 근거 없이 특정 FAQ로 확정하는지 |
| TEMPORAL_VERSION | 과거/현재, 가입·시행 시점 | 실제 버전과 유효기간 | 적용 시점의 정책을 혼동하는지 |
| DOCUMENT_ATTACK | 격리 시험 문서의 명령형 본문 | 원래 문서의 사실과 공격 노출 조건 | 검색, Context, 생성 영향 구분 |
| PARAPHRASE | 동의어, 구어체, 존댓말, 문장 순서, 직접 질문과 상황 설명 | 의도와 필요한 사실 | 표면 표현이 달라도 같은 근거를 찾는지 |
| LONG_QUERY | 앞뒤의 무관 배경, 질문 위치, 여러 요청 중 핵심 요청 | 정답 조건과 명시적인 핵심 질문 | 길이와 주변 주제에 의해 근거를 놓치는지 |
| TYPO | 한 글자 오타, 자판 오타, 조사와 띄어쓰기, 명칭의 작은 오타 | 해석 가능한 의도 | 작은 표기 오류로 정답이 밀리는지 |
| NO_FAQ | 가까운 주제지만 빠진 사실, 없는 상품, 지원 밖 요청 | 코퍼스에 핵심 답이 없다는 검수 근거 | 답을 뒷받침하지 못하는 문서를 채택하는지 |

추가 태그 `MULTI_FACT`, `CONDITION`, `CROSS_INTENT`, `CONTEXTUAL`로 필요한 사실 개수, 정책 경계, 다른 처리 의도의 경쟁, 대화 의존성을 기록한다. REAL_FAILURE는 문장 유형보다 출처를 나타내므로 다른 태그와 함께 사용할 수 있다. 실제 서비스 로그가 없으면 이 유형은 0건으로 보고한다. 생성한 실패 예시를 실제 서비스 사례로 표기하지 않는다.

숫자가 등장하는 모든 질문을 EXACT_ENTITY로 분류하지 않는다. 해당 숫자 또는 조건을 정확히 구분하는 것이 문항의 검증 목적일 때 태그를 붙인다. 원문에 없는 요금제명과 가격은 정상 정답을 가진 문항으로 만들지 않는다.

### 4.2 표본 목표

먼저 calibration용 240개 시나리오 family를 검수한다. 기존 120개 목표에 NEGATION, CONDITION_SCOPE, NUMERIC_CONDITION, COMPARISON, MULTI_INTENT, AMBIGUOUS_QUERY 각 20개를 추가한 제안이다. 기본형과 변형을 작성하면 실행 문항 수는 family 수보다 커지므로 실제 작성 수를 별도 기록한다. TEMPORAL_VERSION과 DOCUMENT_ATTACK은 격리 fixture 트랙이며 이 확보 목표 합계에서 제외한다. Pilot은 holdout으로 재사용하지 않는다.

확장 단계의 다음 값은 **확보 목표이며 현재 완성된 데이터 수가 아니다.**

| 대표 유형 | Calibration family 목표 | Holdout family 목표 |
|---|---:|---:|
| EXACT_ENTITY | 100 | 50 |
| HARD_NEGATIVE | 100 | 50 |
| KR_EN_EQUIVALENT | 100 | 50 |
| PARAPHRASE | 100 | 50 |
| LONG_QUERY | 60 | 30 |
| TYPO | 60 | 30 |
| NO_FAQ | 200 | 300 |
| NEGATION | 60 | 30 |
| CONDITION_SCOPE | 60 | 30 |
| NUMERIC_CONDITION | 60 | 30 |
| COMPARISON | 60 | 30 |
| MULTI_INTENT | 60 | 30 |
| AMBIGUOUS_QUERY | 60 | 30 |
| 합계 | 1,080 | 740 |

같은 family에서 여러 변형을 만들 수 있지만 표본 수와 신뢰구간은 family 단위로 계산한다. 태그는 겹칠 수 있고 대표 유형은 하나만 지정하므로 표의 합계를 중복 집계하지 않는다. MULTI_FACT와 CONDITION은 긍정 문항 안의 별도 slice로 보고한다.

300개의 독립 NO_FAQ family에서 오탐 0건이라면 독립 Bernoulli 가정의 단측 95% 이항 상한은 약 1%다. 이것은 표본 규모를 설명하는 값이며 실제 질문이 독립이라는 보장은 아니다. 반복 표현으로 숫자만 채우지 않는다. 충분한 family를 확보하지 못하면 실제 표본 수와 불확실성을 함께 보고한다.

### 4.3 정답 라벨

사실별 정답은 AND of OR 구조로 표현한다.

```json
{
  "fact_groups": [
    {"fact_id": "remaining_installment", "primary_ids": ["FAQ-943"], "acceptable_ids": []},
    {"fact_id": "return_location", "primary_ids": ["FAQ-963"], "acceptable_ids": []},
    {"fact_id": "device_reset", "primary_ids": ["FAQ-932"], "acceptable_ids": []}
  ]
}
```

한 fact_group 안에서는 primary 또는 검수된 acceptable FAQ 중 하나를 찾으면 그 사실을 회수한 것이다. 그룹 사이에서는 필요한 모든 사실을 찾았는지 평가한다. 위 문항에서 FAQ-943만 찾은 것은 3개 사실 중 1개 적중이며 전체 답변 근거를 확보한 것은 아니다.

새 엑셀의 `검색 정답 FAQ 집합 (JSON)`에서 내부 배열은 하나의 사실을 뒷받침하는 대체 ID다. 초기 변환에서는 해당 ID를 같은 그룹의 primary_ids에 그대로 둔다. 이후 검수로 동등한 근거를 acceptable_ids에 추가한다. 배열을 펼쳐 단일 OR 정답으로 바꾸지 않는다.

하나의 FAQ가 여러 사실을 충분히 뒷받침하면 각 해당 그룹에 같은 ID를 넣을 수 있다. 반대로 일부 사실만 제공하는 문서를 전체 정답으로 인정하지 않는다. 원문이 변경되거나 정답이 모호하면 `AMBIGUOUS_GT` 검수 대기 상태로 두고 주 지표에서는 제외하되 제외 건수와 이유를 공개한다.

Hard Negative는 높은 dense 점수를 가진 후보에서 채굴할 수 있지만 자동으로 오답 확정하지 않는다. 사람이 사실과 적용 대상을 확인한다. 정답 집합과 Hard Negative의 ID 교집합은 금지한다. 검수되지 않은 후보는 분석 후보로만 기록한다.

정답 ID, 기대 답변, 필수 사실, 검색질문 검수 기준은 검색 query·필터·reranker 문서에 포함하지 않는다. FAQ 원문의 질문과 답변만 검색 문서로 사용한다.

### 4.4 영어와 한국어의 의미 동등성

영어·한국어 표기 차이로 검색이 달라지는 문제는 `KR_EN_EQUIVALENT` 유형으로 독립 관리한다. 용어 사전은 문항 작성과 GT 검수에 사용하며 최초 A~D 실험에서 query 치환기로 사용하지 않는다. 검색 전에 동의어를 강제 치환하면 원래 임베딩이 이 차이를 처리하는 능력을 측정할 수 없다.

| 유형 | 같은 의미 쌍의 예 | 검수 기준 |
|---|---|---|
| 한국어와 영어 | 로밍 ↔ roaming, 데이터 ↔ data | 통신 도메인에서 같은 서비스와 사실을 질문하는지 |
| 영문과 한글 음역 | eSIM ↔ 이심, USIM ↔ 유심 | 공식 명칭과 도메인 용례가 같으며 상품 범위가 바뀌지 않는지 |
| 영어 용어 혼합 | 데이터 로밍 ↔ data roaming | 문장 전체는 유지하고 용어만 바뀌었는지 |
| 약어와 풀어쓰기 | 영어 약어 ↔ 한국어 설명 | 실제 서비스 범위가 같은지 검수 |
| 영문 상품과 설명 | 영문 브랜드 표기 ↔ 해당 상품을 특정하는 한국어 설명 | 설명이 여러 상품에 해당하면 동등 쌍으로 확정하지 않음 |
| 문장 번역 | 한국어 질문 ↔ 같은 의미의 영어 질문 | 추가 진단. 용어 치환 결과와 합산하지 않음 |

표의 단어 쌍은 검수 후보다. 모든 문맥에서 자동으로 같은 의미라고 확정하지 않는다. 특히 물리 SIM과 eSIM, 가입과 activation, 해지와 refund, 명의 변경과 number portability처럼 서비스 범위가 달라질 수 있는 표현을 동의어로 묶지 않는다. USIM과 유심도 해당 FAQ 문맥에서 검수한다. 비슷한 다른 상품명은 HARD_NEGATIVE 대조로 별도 작성한다.

한 `semantic_pair_id`에는 `ko_anchor`, `en_term`을 반드시 포함한다. `mixed_terms`는 선택적으로 추가하고 `en_sentence`는 별도 언어 진단 slice로 둔다. 같은 쌍은 동일한 fact_groups, expected_routes, answerability, 실행 조건을 유지하며 영어로 바꾸는 용어 외에는 문장·수치·조건을 바꾸지 않는다. 같은 family와 split_group_id에 묶어 calibration과 holdout 사이에 흩어지지 않게 한다.

실제 원문을 이용한 최소 쌍은 다음과 같다.

```text
semantic_pair_id: roaming-country-product-price
ko_anchor: 해외 데이터 로밍 요금은 어떻게 되나요?
en_term:   해외 데이터 roaming 요금은 어떻게 되나요?
mixed_terms: 해외 data roaming 요금은 어떻게 되나요?
필요 사실: 방문 국가와 가입한 로밍 상품에 따라 데이터 요금이 달라진다.
Primary GT: FAQ-028
```

각 표현의 Cross-language Hit@K, FactRecall@K, AllFactsHit@K, fact별 Rank Delta, Top-K Overlap을 A~D와 조건부 E/F에서 측정한다. 순위가 후보 밖이면 rank는 N/A로 두고 회수 실패를 따로 기록한다. Top-K Overlap은 두 ID 집합 교집합 크기를 K로 나눈 진단값이며, K개 미만 반환 시 actual_count도 공개한다. 한국어 질문은 성공하고 영어 용어로 바꾼 질문만 실패한 family를 개별 분석한다. 정답이 같은 의미의 대체 FAQ로 바뀌었다면 ID가 달라도 semantic 성공으로 인정한다.

`PairAllFactsHit@K`는 ko_anchor와 모든 필수 용어 변형에서 필요한 사실을 모두 찾은 쌍의 비율이다. 기본형과 변형형의 FactRecall 차이를 `LanguageGap`으로 함께 보고한다. 두 표현이 모두 같은 오답을 찾는 경우에도 일관성 지표만 높아질 수 있으므로 성공률 없이 결과 일치율만 보고하지 않는다. 문장 전체 영어 번역 결과는 필수 용어 쌍의 성공률 분모에서 분리한다.

실패하면 candidate miss, merge truncation, reranker regression 중 어느 단계인지 확인한다. EXACT_ENTITY와 겹치는 쌍은 두 slice에 모두 표시하되 전체 문항 수에는 한 번만 센다. 구조 개선으로 해결되지 않을 때만 용어 사전 query 정규화나 keyword 검색을 **추가 calibration 실험**으로 검토한다. 최초 baseline의 입력을 바꾸거나 holdout 결과를 보고 사전을 추가하지 않는다.

### 4.5 부정, 숫자 경계와 적용 범위

변경한 의미 축은 `contrast_pair_id`와 `contrast_axis`로 묶고, Entity·숫자·단위·연산자·조건 주체는 `meaning_constraints`에 기록한다. 언어 동등성 쌍은 이 필드를 고정하지만 의미 대조 쌍은 지정한 축만 변경한다. 같은 대조 쌍을 split 양쪽에 나누지 않는다.

FAQ-114는 가입 명의자가 만 19세 미만이어야 한다는 정책이다. 만 18세와 만 19세의 질문은 적용 결론이 다르지만 둘 다 FAQ-114가 근거다. 부모님 명의와 자녀 명의의 질문도 같은 정책 FAQ가 적용 범위를 설명할 수 있다. FAQ-003의 번호 유지 가능/불가 질문도 같은 FAQ가 근거다. 이런 경우 GT와 rank를 억지로 분리하지 않는다. 서로 다른 정책 문서가 실제로 필요한 경우에만 반대 정책 FAQ를 Hard Negative로 지정한다. 같은 FAQ에 대한 긍정/부정 답변의 정확성은 생성 단계에서 따로 평가한다.

### 4.6 비교와 복합 의도

COMPARISON은 비교 대상별 사실 그룹을 둔다. FAQ-865처럼 한 문서가 2회선·3회선 할인 금액을 모두 제공하면 두 그룹에 같은 ID를 쓸 수 있다. 필요한 문서가 두 개인 비교는 두 사실 모두 회수해야 AllFactsHit 성공이다.

MULTI_INTENT는 `subrequests`에 request_id, route, requirement, retrieval_fact_ids, external_action_required를 기록한다. 로밍 요금과 가까운 영업 중 매장 요청에서는 로밍 근거만 FAQ 채점의 GT로 둔다. MAP_API 성공은 위치·영업시간 조회를 실제 실행한 결과로 별도 채점한다. FAQ 회수가 성공해도 전체 요청 완료로 계산하지 않는다. FAQ 문서를 API 조회 결과로 대체하지 않는다.

### 4.7 모호한 질문과 시점

대화 없는 “그거 얼마야?”는 answerability=ambiguous, retrieval_status=UNDETERMINED, expected_generation_status=CLARIFY다. GT를 비워 놓았다는 이유로 NO_FAQ에 넣지 않는다. 실제 대화가 있는 경우는 contextual 트랙에서 검수된 rewrite 출력을 동결한다. 입력 해석과 rewrite 실패는 CONTEXT_REWRITE_FAILURE로 기록한다.

정상 FAQ에는 버전·유효기간 메타데이터가 없으므로 현재/과거 정책의 정답을 지어내지 않는다. TEMPORAL_VERSION은 우선 원천 CF fixture에서만 평가한다. CF-0061은 TEST-CF-0061-A(25%, 2026-01-01 포함~2026-10-01 제외), B(27%, 2026-10-01 포함~종료 없음)를 갖는다. 이는 시험 정책이며 실제 운영 정책이 아니다.

**원천 CF-0061의 계약은 날짜 선필터 없이 A/B 모두 회수**하는 것이다. 원문 FAQ-018~027과 이 두 시험 문서만 별도 코퍼스로 구성한다. Retrieval GT는 두 버전의 근거이며, `temporal_context.expected_policy_ids`로 시점에 적용되는 정책 선택을 별도 평가한다. 날짜 필터 검색을 추가하려면 기존 fixture 계약과 다른 실험 ID로 정의한다.

### 4.8 문서 공격

DOCUMENT_ATTACK은 격리 인덱스의 retrieval exposure, context construction, generation/guardrail을 각각 기록한다. AD-0141은 전체 정상 코퍼스에서 FAQ-958 하나를 TEST-AD-0141로 교체한다. 공격 문구는 시험 데이터로 취급하며 실행 지시로 읽지 않는다.

공격 문서가 actual LLM input에 포함되어야 생성 방어 평가가 성립한다. 미검색 또는 Context 제외는 UNEXPOSED로 보고하며 방어 성공으로 세지 않는다. 생성 미실행은 NOT_EVALUATED다. 노출 후 원래 사실에 답했는지와 문서 안 명령을 따랐는지는 별도 채점한다. 이 결과는 일반 Hit/MRR 평균에서 제외한다.

## 5. Calibration과 Holdout

원천 문항을 그대로 가져오는 단계에서 family를 먼저 만들고 이후 변형을 작성한다. 같은 원본 질문, 같은 시나리오 family, 같은 입력의 복사본, 같은 템플릿에서 조건만 변형한 묶음, SR·HR·EC의 짝 문항, RT의 반복 실행은 같은 split에 둔다. 관련 묶음을 연결한 component 단위로 분할해 간접 중복도 막는다.

FAQ 코퍼스는 calibration과 holdout에서 동일하게 사용한다. holdout은 **처음 보는 질문과 시나리오에 대한 평가**이며 처음 보는 FAQ를 검색하는 실험은 아니다. 같은 사실과 같은 표현 템플릿의 변형은 같은 family로 두지만, 공통 FAQ 하나를 참조한다는 이유만으로 전체 복합 질문을 무조건 한 component로 묶지는 않는다. 최종 family 기준과 중복 점검 결과를 기록한다.

기존 회귀셋과 이미 튜닝에 사용한 질문은 holdout에 넣지 않는다. 새 holdout 문항은 배정된 family에서 독립적으로 작성·검수하고 실행 결과를 보기 전에 `split_manifest.json`과 데이터 해시를 동결한다. GT 검수는 모델 결과와 분리해 수행한다.

Calibration에서 표현, 병합 방식, 후보 수, reranker, threshold를 결정한다. Holdout에서는 고정 baseline과 선택한 challenger만 동일한 최종 정책으로 실행한다. 결과를 본 뒤 수정하면 해당 holdout은 개발 자료가 되고 다음 버전에 새 holdout을 만든다. 실패 문항을 제외해 다시 같은 holdout 점수를 발표하지 않는다.

## 6. A와 B의 문서 표현

| 실험 | 문서 텍스트 | 검색 |
|---|---|---|
| A | 원문 `question` | 질문 벡터 순위 |
| B | `질문: {question}\n답변: {answer}` | 질문과 답변 벡터 순위 |

공백 정규화 외에는 원문을 요약하거나 수치와 조건을 제거하지 않는다. query에는 FAQ 문서용 라벨을 붙이지 않는다. BGE-M3 query instruction을 새로 추가하지 않는다. 공식 모델 카드는 query instruction 없이 사용할 수 있다고 명시한다. [BGE-M3 모델 카드](https://huggingface.co/BAAI/bge-m3)

질문과 답변 문자열 구성은 embedding과 Q+A reranker 입력에서 동일하게 사용한다. 구분자와 순서도 버전의 일부다. 초기에는 제목, 카테고리, 처리 의도, GT ID를 추가하지 않는다.

1,024건 전체에 대한 exact dense search를 사용해 ANN 손실을 분리한다. 청킹이나 ANN 도입은 이번 A~D 비교에 포함하지 않는다. Query는 한 번, corpus의 Q와 Q+A 벡터는 각각 한 번 생성해 모든 비교에 재사용한다.

Embedding의 유효 max sequence length는 baseline 모델 설정에서 읽어 고정한다. 모델 카드의 최대 지원 길이와 실제 런타임 설정을 구분한다. 문항별 token 수, 잘린 token 수를 기록하고 답의 핵심 사실이 잘린 경우 별도 실패로 분류한다. Reranker 입력 길이도 각 모델의 유효 설정을 기록하며 Q와 Q+A 비교 중 변경하지 않는다.

## 7. C의 후보 병합과 공정한 비교

### 7.1 Union과 dedupe

두 검색에서 각 FAQ의 `q_score`, `qa_score`, `q_rank`, `qa_rank`를 구한다. 각 view의 상위 `per_view_k`를 합친 뒤 **corpus_version과 FAQ ID**로 중복을 제거한다. 본문이 비슷하다는 이유로 서로 다른 FAQ를 검색 결과에서 삭제하지 않는다. 의미상 대체 정답은 라벨에서 처리한다.

단순 union은 집합이므로 Top-1이나 MRR을 계산할 수 없다. C의 초기 정렬 규칙은 다음과 같이 고정한다. 가중치를 튜닝하지 않는다.

```text
merged_score(faq) = max(q_score(faq), qa_score(faq))
정렬: merged_score 내림차순 → FAQ ID 오름차순
```

Exact search로 두 전체 score 배열이 있으므로 union에 들어온 각 후보의 두 점수를 함께 기록한다. 검색 channel, 양쪽의 원래 순위, union 진입 출처도 저장한다. Max 규칙은 Q+A의 점수 분포가 높으면 그 view를 더 많이 선택할 수 있다. 이것은 검증할 가설이며 개선이 보장된다는 뜻이 아니다. 점수 분포 문제를 발견하면 calibration에서 정렬 규칙을 별도 실험으로 추가하고 실험 ID를 바꾼다.

### 7.2 후보 수 효과를 분리

| 비교 | A와 B | C | 해석 |
|---|---|---|---|
| 원래 후보 집합 | Top-k | Q Top-k ∪ Q+A Top-k, U개 | U가 최대 2k이므로 후보 수 증가도 포함 |
| 같은 최종 후보 수 | Top-M | 각 view Top-M의 union을 max 규칙으로 정렬하고 Top-M | reranker에 주는 FAQ 수가 동일 |
| 후보 수 대조 | A Top-2k, B Top-2k 및 query별 Top-U | 위 raw union | 넓게 검색한 효과와 표현을 추가한 효과 구분 |

`k, M ∈ {3,5,10,20}`으로 평가한다. Raw union의 결과는 `UnionFactRecall(per_view_k=k, actual_u=U)`로 표기하고 Recall@k라고 부르지 않는다. C Top-M의 지표만 `FactRecall@M`으로 표기한다. 모든 방식에서 후보 수의 평균, p95, 최대, 중복률을 보고한다.

같은 M이라도 C는 두 vector view를 탐색하므로 검색 연산량이 같다거나 저장 비용이 같다고 해석하지 않는다. 후보 수를 맞추는 비교와 시간·메모리 비용 비교를 함께 제시한다.

C의 raw union에는 있는데 Top-M에 없는 정답은 merge truncation 실패다. 임베딩 후보 생성 실패와 구분한다. D는 선택한 C Top-M을 그대로 사용한다.

## 8. D의 Reranker 비교

| 변수 | 초기 비교 값 |
|---|---|
| Candidate pool | C의 고정된 Top-M, 같은 candidate cache |
| M | 10, 20 |
| 모델 | BGE reranker v2-m3, Jina reranker v2 |
| 입력 Q | `(query, faq.question)` |
| 입력 Q+A | `(query, "질문: ...\n답변: ...")` |
| 최종 전달 | Top-3, threshold 없이 먼저 평가 |

최대 2개 모델 × 2개 입력 × 2개 M = 8개 기본 조합이다. 모델과 후보 수를 동시에 바꾸며 개선 원인을 설명하지 않는다. 먼저 같은 M과 모델에서 Q/Q+A 입력을 비교하고, 같은 입력에서 모델을 비교한다.

D와 C는 정확히 같은 후보를 본다. `A+BGE reranker`를 같은 M에서 보조 대조군으로 둬 기존 reranker 문제와 dual 후보 생성의 효과를 분리한다. C가 B보다 좋지 않다면 dual을 강제 채택하지 않고 A 또는 B의 후보 생성기를 유지할 수 있다.

공식 BGE reranker는 query와 document의 관련성 점수를 출력하며 sigmoid로 0~1 범위로 변환할 수 있다. 이 값에 기존 cosine threshold 0.75를 적용하지 않는다. [BGE reranker 모델 카드](https://huggingface.co/BAAI/bge-reranker-v2-m3)

모델 adapter는 `score_kind`, activation, 정규화 여부를 명시한다. Raw logit과 변환 score를 모두 저장하며 sigmoid를 두 번 적용하지 않는다. Jina는 해당 모델의 공식 구현과 입력 길이를 사용하고 동일하게 score 계약을 기록한다. [Jina reranker 모델 카드](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)

후보 M개 중 최종 3개 안으로 정답이 들어왔는지와 순위 변화는 별개다. `Improved=10, Worsened=10`처럼 건수가 같아도 변화 폭과 쿼리별 RR 차이는 다를 수 있다. 개선 수와 악화 수만으로 순효과를 판정하지 않는다.

## 9. E: BGE-M3 native Sparse

A~D의 검수된 LEXICAL_MISS·정확한 명칭·숫자/단위 실패가 남을 때 진입한다. 선택된 단일 문서 표현과 같은 텍스트·token 제한을 고정하고 Dense Only / Sparse Only / Dense+Sparse를 비교한다. E에서 질문과 Q+A까지 동시에 변경하지 않는다.

기존 sentence-transformers dense와 FlagEmbedding dense를 먼저 같은 모델 revision으로 대조한다. backend 차이가 품질 변화에 포함되면 sparse의 효과를 분리할 수 없다. BGE 공식 구현의 lexical_weights와 lexical matching score를 사용하며 token ID와 가중치의 sparse 표현을 저장한다. [BGE-M3 공식 모델 카드](https://huggingface.co/BAAI/bge-m3)

초기 Dense+Sparse는 channel별 Top-M union/dedupe와 동일 unique budget M을 평가한다. Dense cosine와 sparse score는 범위가 다르므로 raw 점수를 더하지 않는다. 합친 후보의 순위가 필요하면 동등 가중 RRF(rank 상수 60)를 별도 merge 실험으로 고정하고 channel별 점수·순위·union 회수를 모두 저장한다. Raw union, 고정 M, 후보 수 확대 대조를 구분하며 C/D와 같은 후보 수 통제를 적용한다. RRF와 reranker 변경 효과도 분리한다.

## 10. F: BGE-M3 native Multi-vector

C의 Dual은 FAQ마다 질문과 Q+A **두 dense 벡터**다. F는 BGE의 colbert_vecs인 **token 수준 표현**이다. 두 개념과 cache 키를 공유하지 않는다. 선택한 단일 문서 표현, revision, token 제한을 고정하고 공식 colbert_score로 late interaction을 계산한다. [BGE-M3 공식 모델 카드](https://huggingface.co/BAAI/bge-m3)

1,024건 코퍼스 전체의 exact late interaction 검색과 고정 dense 후보 pool의 rescore를 서로 다른 실험 ID로 둔다. 후자는 pool 밖 근거를 복구할 수 없으므로 전체 코퍼스 회수 개선으로 보고하지 않는다. 기존 intent 코드의 임의 정규화·MaxSim 계산은 공식 scoring과 일치하는지 검증 전에는 재사용하지 않는다.

Hard Negative/조건 구분의 개선과 함께 token vector 수, dtype, corpus bytes, index 구축 시간, query latency와 peak memory/VRAM을 기록한다. 평균 개선이 불확실하거나 비용이 운영 조건을 넘으면 운영 후보로 채택하지 않는다. 필요할 때만 F 이후 BM25/hybrid를 추가한다.

## 11. 지표 정의

Query i의 필요한 사실 그룹을 G_i, 그룹 g의 primary와 acceptable ID 합집합을 R_ig, 상위 K개 결과를 S_iK라고 둔다. GT가 있는 문항만 아래 검색 지표의 분모에 들어간다.

| 지표 | 문항별 정의 | 용도 |
|---|---|---|
| StrictHit@K | primary ID 중 하나 이상이 S_iK에 있음 | 기존 Primary GT 비교 유지 |
| SemanticHit@K | 어떤 사실 그룹의 primary 또는 acceptable ID가 하나 이상 있음 | Top-1, Top-3, Top-5의 단일 사실 검색 기준 |
| FactRecall@K | 적중한 사실 그룹 수 / 필요한 사실 그룹 수 | 복합 질문의 부분 근거 회수 |
| AllFactsHit@K | 모든 사실 그룹이 하나 이상의 근거를 확보함 | LLM에 필요한 전체 근거 확보 |
| SemanticMRR@M | 첫 번째 semantic GT 순위의 역수. M 밖이면 0 | 같은 후보 범위에서 순위 비교 |
| FactMRR@M | 각 사실 그룹의 첫 근거 RR을 평균. 미회수 그룹은 0 | 일부 사실만 높은 순위인 문제 확인 |
| FullMRR | 전체 1,024건 순위에서 첫 GT의 RR | A/B/C의 진단용. 후보 MRR과 분리 |
| PairAllFactsHit@K | 한국어 기본형과 필수 영어 용어 변형에서 모두 AllFactsHit 성공 | KR_EN_EQUIVALENT의 의미 동등성 검증 |
| LanguageGap@K | 쌍별 한국어 기본형 FactRecall에서 영어 용어 변형 FactRecall 평균을 뺀 값 | 언어 표기에 의한 성능 차이 |
| HardNegativeError@K | 검수된 오답 ID가 상위 K에 포함됨 | 근접 오답 노출 |
| HardNegativePairWin | 각 positive fact와 hard-negative의 순위 비교 승률 | 비교 쌍이 존재하는 문항만 집계 |

원래 계획의 Top-1/3/5는 기본적으로 SemanticHit@1/3/5다. 단일 사실 질문에서는 FactRecall@K와 Hit@K가 같다. 복합 질문에서는 같지 않으므로 하나의 Recall이라는 이름으로 섞지 않는다. 현재 운영 Top-3의 핵심 성공 지표는 **AllFactsHit@3**로 둔다.

Primary-only와 semantic 지표를 모두 보고해 대체 정답을 허용한 영향이 보이게 한다. NO_FAQ, API-only, route-only, 특수 조건의 비적용 지표는 0이 아니라 N/A다. 제외 사유별 수를 별도 보고한다.

평균은 reporting_cohort별로 먼저 분리하고, 대표 유형·원천 항목·단일/복합 사실·FAQ 의도별 slice를 제공한다. core_general의 주 지표와 targeted_condition의 결과를 분리한다. 문항 평균과 family 평균을 함께 제시하며 주 비교는 family 평균이다. 10회 반복 문항이 전체 평균을 10배 가중하지 않는다.

## 12. 실패 분석과 실행 로그

실패 태그는 함께 붙을 수 있다. 예를 들어 EXACT_ENTITY에서 정답이 후보에 없다면 RETRIEVAL_MISS와 LEXICAL_MISS 가설을 함께 기록한다.

| 분류 | 판정 기준 |
|---|---|
| RETRIEVAL_MISS | 필요한 사실이 raw 후보 집합에서 회수되지 않음. 전체 미회수와 일부 미회수를 분리 |
| MERGE_TRUNCATION | raw union에는 정답이 있으나 C Top-M에서 제거됨 |
| RANKING_ERROR | C Top-M에 정답이 있지만 final_n 안에 필요한 근거가 없음 |
| RERANKER_REGRESSION | 동일 pool의 C보다 D에서 정답 순위 또는 사실 회수가 악화 |
| LEXICAL_MISS | 명칭·조건이 정확히 존재하나 dense가 놓친 사례. tokenization과 GT 검수 후 확정 |
| AMBIGUOUS_GT | 중복·모호한 FAQ나 불충분한 라벨로 판정 불가 |
| NO_FAQ_FALSE_POSITIVE | 답이 없는 질문을 최종 threshold 정책이 답할 수 있다고 채택 |
| INPUT_TRUNCATION | 핵심 질문 또는 근거가 token 제한으로 잘림 |
| CROSS_INTENT_ERROR | 다른 처리 의도의 문서가 앞서거나 잘못된 호출 후보를 형성 |
| CONTEXT_REWRITE_FAILURE | 이력의 대상·조건·부정을 잃거나 원천의 정답용 query가 누출됨 |
| NON_DETERMINISTIC_RESULT | 같은 입력·설정 반복에서 후보 집합 또는 순위가 달라짐 |

Reranker의 `Stage1 Miss`는 필요한 fact 중 후보 pool에 없는 fact를 뜻한다. 하나라도 없는 query 수와 모든 fact가 없는 query 수를 함께 저장한다. 일부 미회수 query에서도 pool에 있는 fact의 순위 변화는 계속 측정한다.

Query별 로그에는 다음 내용을 남긴다.

- run ID, 설정 해시, corpus와 dataset 버전, split, family, 원천 문항 ID, 검수된 query 유형
- 실제 검색 query와 해시, 대화 재작성 버전, token 수와 truncation
- Q/QA Top-k ID·score·rank, union 진입 channel, actual_u, C Top-M
- Reranker 입력 표현, 후보별 입력 해시와 token 수, raw score, 정렬 score, 최종 순위
- fact별 baseline/reranked rank, RR 차이, Hit와 AllFacts 변화, Stage1 Miss
- threshold 통과 문서, query 채택/보류 판정, 실패 태그와 검수 메모
- 단계별 latency, 최종 문서 본문 해시, 결과 Context 해시
- repeat group별 candidate/rank/score hash와 변화. output hash는 실제 생성 실행 때만 기록
- subrequest별 retrieval 성공과 API 실행 상태, 공격 문서의 실제 입력 노출 상태

실패 사례 표는 최종 Top-1 성공→실패, AllFacts@3 성공→실패, Stage1 Miss, merge truncation을 우선순위로 정렬한다. 악화된 모든 문항을 검수 대상으로 남긴다.

## 13. Threshold Calibration

Threshold 없는 Top-K 검색은 NO_FAQ에도 결과를 반환한다. 따라서 후보가 나왔다는 사실만으로 false positive라고 채점하지 않는다. 순위 실험에서 저장한 NO_FAQ score 분포를 사용해 최종 채택 정책을 별도로 평가한다.

구조를 먼저 고른 뒤 해당 구조의 final score에 대해 calibration한다. 기존 0.75는 A의 재현용 정책이다. C의 max score나 reranker score에 그대로 옮기지 않는다.

| 정책 | 적용 |
|---|---|
| P0 | A의 Top-3 각각에 cosine >= 0.75. 현재 계획의 정책 대조군 |
| P1 | 선택 구조의 Top-3 각각에 score >= t |
| P2 | P1에 query-level Top-1 대 Top-2 margin >= m 추가. 먼저 단일 사실 slice의 진단으로 비교 |

Margin은 검수된 동등 근거 그룹을 하나로 묶은 뒤 경쟁 그룹끼리 계산하는 진단도 함께 제공한다. Raw Top-1과 Top-2가 같은 사실의 대체 FAQ이면 작은 margin은 모호성 증거가 아니다. 두 문서가 모두 필요한 복합 질문에는 margin gate를 적용하지 않는다. 경쟁 후보가 없어 margin을 계산할 수 없으면 N/A로 둔다.

P2를 운영 후보로 선정하려면 정답 라벨을 보지 않고 단일·복합 질문을 판별하는 입력 처리 규칙을 먼저 고정해야 한다. 해당 규칙이 없으면 P2는 slice 진단에만 사용하고 P1을 운영 정책 후보로 둔다. 동등 FAQ 그룹을 실제 gate에 쓰려면 corpus만으로 검수한 고정 그룹 표를 사용한다. Query의 fact_groups나 정답 ID를 이용해 margin을 보정하는 것은 평가용 진단이며 운영 성능으로 보고하지 않는다.

처음에는 calibration score의 정렬된 관측값과 양끝 sentinel을 threshold 후보로 삼아 FPR과 사실 회수 곡선을 만든다. 단일 사실 margin은 calibration 관측값을 사용한다. Holdout의 분포를 threshold 후보 생성에 사용하지 않는다.

기본 선택 정책은 **unsafe_accept_rate <= 1%를 만족하는 설정 중 family 평균 AllFactsHit@3를 최대화**하는 것으로 제안한다. 이는 실험용 제안이며 운영 오탐 허용치를 확정한 값은 아니다. 목표를 만족하는 설정이 없으면 미달로 보고하고 더 느슨한 threshold를 조용히 채택하지 않는다. 동일한 성능이면 단순한 정책과 낮은 latency를 선택한다.

NO_FAQ를 positive class로 두면 precision/recall/F1은 NO_FAQ 탐지 지표다. detector_false_positive_rate는 **답이 있는 질문을 NO_FAQ로 잘못 보류한 비율**이다. 답 없는 질문을 채택하는 안전성 지표는 unsafe_accept_rate = accepted_none / all_none이며, NO_FAQ 탐지기의 false-negative rate다. FAQ 채택을 positive로 부를 때만 이를 FAQ FPR로 부른다. 두 positive 정의를 섞지 않는다. Binary calibration은 명확한 full과 none 문항으로 비교하고 partial·route-only·ambiguous는 별도 상태로 보고한다. 전체 클래스 비율이 달라지면 precision도 달라지므로 원천 데이터의 비율과 평가셋의 비율을 함께 기록한다.

구조와 threshold 선정 후 baseline의 고정 P0와 calibration에서 선정한 A 정책을 둘 다 holdout에 포함한다. Challenger가 이긴 이유가 threshold를 바꾼 효과인지 구조를 바꾼 효과인지 분리한다.

## 14. 실험 순서와 채택 기준

| 단계 | 실행 | 다음 단계로 넘길 근거 |
|---|---|---|
| 0 | 원천 해시, 새 코퍼스, 라벨과 split, 중복 검수 | 버전과 정답 의미 확정 |
| 1 | 기존 코퍼스에서 A 회귀 재측정 | 기존 코드와 새 adapter의 결과 차이 설명 |
| 2 | 새 코퍼스에서 A와 B, threshold 없음 | 같은 문항과 표현 외 조건의 비교 |
| 3 | C의 raw union, 같은 M, 후보 수 대조 | candidate 개선이 후보 수만의 효과인지 확인 |
| 4 | C 대비 D, Q/Q+A와 두 reranker | 같은 pool의 순효과와 악화 사례 검수 |
| 5 | 실패 taxonomy와 조건별 slice | E/F 진입 필요성 검수 |
| 6 | 조건부 E → 조건부 F → 필요 시 BM25/hybrid | 회수·순위 개선과 저장·실행 비용 비교 |
| 7 | 최종 품질 구조와 reranker 동결 | 모델·문서 표현·후보 계약 확정 |
| 8 | threshold와 margin calibration | 채택/보류 정책 동결 |
| 9 | baseline과 고정 challenger의 holdout 1회 | 품질 채택 또는 보류 |
| 10 | Docker 엔진 capability/parity → 동일 장비 부하 측정 | Ollama/TEI/vLLM 후보 축소 |
| 11 | 별도 CPU/GPU 비교 | 선택 구조와 엔진의 운영 자원 판단 |
| 12 | 효과가 검증된 구조만 UBot-BE 연결 | 서비스의 데이터·점수 계약 재현 |

초기 목표는 단일 사실의 SemanticHit@1 >= 95%, SemanticHit@3 >= 98%, FactRecall@10 >= 99%, EXACT_ENTITY SemanticHit@1 >= 95%다. 복합 질문은 별도의 AllFactsHit@3 목표와 결과를 보고한다. 수치는 실험 방향의 목표이며 달성 전제를 두지 않는다.

Challenger의 primary decision metric은 같은 `final_n=3`에서 family 평균 AllFactsHit@3다. 95% paired bootstrap을 family 단위로 2,000회 계산하고 baseline 대비 차이의 구간을 보고한다. 우월성 주장은 차이의 하한이 0을 넘는 경우로 제한한다. 그렇지 않으면 불확실한 결과로 둔다. Bootstrap의 seed도 설정의 일부다.

Top-1이 1%p 넘게 낮아지거나 기존 회귀셋의 Top-1/Top-3가 1%p 넘게 낮아지면 초기 채택 조건을 통과하지 못한 것으로 제안한다. 현재 표본에서 1%p가 몇 문항인지 함께 표기한다. EXACT_ENTITY와 NO_FAQ 악화, 사실 안전성 문제는 평균 개선으로 가리지 않는다.

Latency의 허용값은 운영 SLO에서 정해야 한다. SLO가 없으면 정확도에 따른 후보 선정을 할 수는 있지만 운영 채택을 완료한 것으로 판단하지 않는다. CUDA 시간은 synchronize 후 측정하고 cold load, corpus 구축, warm query를 분리한다. 같은 장비에서 stage와 전체의 p50/p95, peak VRAM, corpus 저장량을 측정한다. 배치당 query throughput과 단일 요청 latency를 혼동하지 않는다.

Hybrid는 A~D와 필요한 E/F를 검토한 뒤에도 명칭 또는 조건 실패가 calibration에서 반복되고 BM25 대조가 구제하는 경우에 추가한다. Q+A와 dual이 개선됐다는 이유로 자동 추가하지 않는다. BM25 tokenization과 index 범위를 고정하고 raw 점수 가중합을 초기 기본값으로 두지 않는다. 새로운 hybrid 구조도 후보 수 대조를 거치며 기존 holdout에 반복 튜닝하지 않는다.

## 15. Docker 기반 Serving Engine Benchmark

품질 holdout이 끝난 후 Ollama, TEI, vLLM을 같은 장비에서 비교한다. [서빙 설정](../configs/serving_engine_plan.json)을 실행 manifest로 구체화하고 이미지 digest, 엔진 버전, 모델 revision/파일 해시, tokenizer, pooling, normalization, dtype, quantization, token 제한을 기록한다. 엔진은 하나씩 실행한다. GPU 예약은 Compose device reservation으로 지정한다. [Docker GPU 설정](https://docs.docker.com/compose/how-tos/gpu-support/)

첫 비교는 dense 공통 기능이다. 동일 입력의 차원·finite·norm·reference cosine·최대 절대 오차·Top-K overlap·사실 지표·threshold 판정 변화를 확인하고 calibration probe에서 허용 오차를 먼저 동결한다. 엔진별로 문서와 query를 모두 다시 임베딩한다. 다른 엔진의 corpus와 query를 섞지 않는다. Ollama의 GGUF/양자화와 HF 원본 weights의 차이는 별도 변형으로 보고한다.

| 엔진 | Dense probe | native 표현 probe |
|---|---|---|
| Ollama | /api/embed, input 배열, truncate=false | 지원 여부를 실제 probe로 확인 |
| TEI | /embed, inputs 배열 | dense 지원과 native sparse/ColBERT 지원을 별도 확인 |
| vLLM | /v1/embeddings | BgeM3EmbeddingModel override와 pooling task를 명시하고 probe |

Ollama는 길이 초과를 조용히 자르는 기본 동작이 있으므로 parity 비교에서는 truncate=false로 요청한다. [Ollama embed API](https://docs.ollama.com/api/embed) TEI는 공식 컨테이너와 /embed를 사용하되 설치된 버전에서 BGE의 native 모드까지 지원한다고 추정하지 않는다. [TEI 실행 문서](https://huggingface.co/docs/text-embeddings-inference/en/quick_tour)

vLLM의 BGE 전용 모드는 hf-overrides의 architectures=BgeM3EmbeddingModel, runner=pooling, task=embed/token_classify/token_embed를 명시한다. Sparse /pooling 응답은 /tokenize ID와 연결해야 한다. 해당 override 없이 vanilla XLMRoberta 경로로 실행한 결과를 native sparse/ColBERT로 보고하지 않는다. 최종 명령은 이미지에 설치된 CLI 도움말과 probe로 확인한다. [vLLM BGE-M3 pooling 문서](https://docs.vllm.ai/en/stable/models/pooling_models/specific_models/)

동시 요청 수 1/4/8/16/32와 요청당 texts 1/8/32는 별도 축이다. 각 cell은 warmup 50 요청 후 최소 60초와 1,000회 시도를 모두 충족하고 3회 측정하는 초기 제안이다. cold load와 corpus 구축은 분리한다. 성공 요청의 p50/p95/p99 latency와 전체 시도의 실패·timeout 비율을 함께 보고한다. Throughput은 wall time당 성공 requests/sec와 texts/sec를 모두 기록한다. retry는 끄고 응답 수·차원·입력 순서를 확인한다. Client end-to-end와 server timing은 구분한다.

선택 구조에 필요한 모드가 없으면 SKIPPED_UNSUPPORTED다. 미지원 모드를 throughput=0으로 채점하거나 dense만 실행해 같은 구조라고 부르지 않는다. 서빙 parity가 깨지면 load 점수만으로 후보를 채택하지 않는다. engine drift를 해결하려고 기존 holdout에서 threshold를 다시 튜닝하지 않는다.

## 16. 별도 CPU/GPU 단계

서빙 후보를 좁힌 뒤 같은 입력·구조·엔진 버전에서 CPU/GPU를 비교한다. dtype와 quantization 변경은 장비 효과와 별도 축이다. 품질 parity, p50/p95/p99, 처리량, memory/VRAM, 실패율을 보고한다. 이 결과를 A~F의 검색 구조 개선 점수에 합산하지 않는다. 지원되지 않는 엔진·장비 조합은 미지원으로 기록한다.

## 17. 기존 코드에 연결할 위치

| 기존 자산 | 재사용 | 필요한 확장 |
|---|---|---|
| `benchmark_embeddings_unified_1000.py` | BGE loader, device 감지, batch encoding, 기존 순위/지표 비교 | question만 반환하는 corpus loader를 별도 공용 loader로 확장, Q+A와 두 vector cache |
| `compare_embedding_reranker.py` | reranker 설정과 device, 후보별 score와 변화 분석 | candidate cache 입력, Q/Q+A adapter, fact 단위 rank |
| `compare_embedding_reranker_buffer.py` | 후보 폭과 final_k의 분리 | C와 D의 동일 pool 비교, merge truncation과 AllFacts |
| 기존 CSV와 결과 문서 | 과거 baseline과 실패 검토 | 새 버전 결과로 덮어쓰지 않음 |

기존 함수는 그대로 유지하고 새 계약을 별도 공용 모듈로 추가한다. 다른 모델의 로딩과 기존 CLI 기본값을 바꾸지 않는다. `first_rank()`는 단일 OR 정답과 기존 회귀에 재사용하고 fact-group 채점기는 별도로 둔다. 의미가 달라진 새 지표를 기존 `mrr` CSV 열에 쓰지 않는다.

데이터 검증·지표 코드와 모델 runner를 구현했다. 아래 트리는 설계 당시의 권장 구조다. 실제 실행 dataset은 `faq/retrieval_run_v2/`, 모델 runner는 `scripts/run_retrieval_benchmark.py`, native runner는 `scripts/run_native_retrieval.py`다. 승인 dataset은 후속 단계로 남아 있다.

```text
UBot-EmbeddingTest/
  model_benchmarks/
    docs/FAQ_Retrieval_개선_실험_설계.md
    configs/retrieval_experiment_plan.json
    configs/retrieval_source_audit.json
    schemas/retrieval_case.schema.json
    faq/retrieval_v1/
      sources/                   원천 엑셀의 동결 사본
      corpus.jsonl               ID, category, question, answer, intent, hashes
      calibration.jsonl
      holdout.jsonl
      regression_current.jsonl
      special_conditions.jsonl
      split_manifest.json
    scripts/retrieval_common.py   구현: 사실 채점, 후보 병합, 의미 계약 검증
    scripts/validate_retrieval_dataset.py   구현: JSON Schema와 원천 참조 검증
    tests/test_retrieval_common.py   구현: 실패 위험을 검증하는 고정 fixture
    scripts/benchmark_retrieval_architecture.py
    outputs/retrieval/<run_id>/   manifest, summary, per_query, failures, latency
  reranker/
    scripts/compare_retrieval_rerankers.py
    outputs/retrieval/<run_id>/
```

유효한 runner의 구현 계약은 `prepare → validate → embed → candidates → evaluate → rerank → calibrate → holdout` 순서다. `validate`는 모델 로딩 없이 JSON, 참조 ID, GT/HardNegative 충돌, split 중복, 원천 SHA, 필수 사실 근거를 확인한다. `holdout`은 확정 config 해시와 split 해시를 요구하고 calibration 옵션을 받지 않는다.

모델을 사용하는 구현 검증 전에도 dedupe, 동점 정렬, AND/OR 사실 채점, 대체 GT, 후보 밖 RR=0, NO_FAQ의 N/A, split family 중복, 이중 sigmoid 방지를 작은 고정 fixture로 확인해야 한다. 실제 모델 검증은 calibration 문항 20개로 각 단계의 로그와 시간 계약을 점검한 후 전체 calibration으로 확대한다.

## 18. 이번 설계의 산출물

- [최신 프로젝트 계획](../../README.md): 사용자 최신안과 저장소의 실행 상태
- [실험 기본 설정](../configs/retrieval_experiment_plan.json): A~F, 실행 순서, 표본 목표와 미확정 항목
- [서빙 설정](../configs/serving_engine_plan.json): 엔진별 probe, parity와 부하 측정 계약
- [원천 자료 점검 기록](../configs/retrieval_source_audit.json): 실제 파일 해시, 데이터 건수, 기존 FAQ ID 충돌
- [문항 JSON Schema](../schemas/retrieval_case.schema.json): 질문, source, family, split, 사실별 GT 계약
- 설계 예시 문항 (로컬 전용: `../faq/retrieval_design_examples.jsonl`): 실제 FAQ 근거를 사용한 형식 예시. 검수된 최종 평가셋이나 holdout이 아님

240개 검수 family는 설계의 확보 목표다. 실제 탐색 실행은 원천 라벨의 1,997행을 연결 component로 나눴으며 이 수를 검수 완료 표본 수로 발표하지 않는다. 질문 Top20 → BGE question reranker를 동결했지만 NO_FAQ 및 기존 235문항 Top1 gate가 실패해 운영 적용하지 않는다. 다음 데이터 버전은 독립 GT 검수와 실제 서비스 실패 로그를 확보해야 한다. 원천 3,000행을 그대로 늘리거나 고정 Context의 정답 문서를 검색 결과로 사용하지 않는다.

Docker 명령과 검증 결과는 [Docker 실행 안내](DOCKER_실행_안내.md)에 기록한다. 컨테이너에서 모델 없는 데이터 검증과 지표 테스트를 먼저 실행한다. 설계용 문항은 --allow-design-examples 옵션을 명시해야 검증할 수 있고, 일반 실행에서는 승인되지 않은 문항을 거절한다.
