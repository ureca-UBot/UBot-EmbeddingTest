# UBot BGE-M3 Retrieval Benchmark

Source: SK Telecom / KT / LG U+ public FAQ pages, used for non-commercial research/evaluation

**공개 범위:** 코드·평가 방법·집계 수치만 제공한다. FAQ 본문, 평가 질문·근거, 상세 로그, 데이터 ZIP은 private/local로 관리한다. 공개용 자료는 [평가 방법](EVALUATION.md), [집계 결과](public_results/README.md), [데이터 관리 안내](DATA_POLICY.md)에서 확인한다. 아래 설계·이력에서 참조하는 데이터·검수 화면·원시 출력 경로는 로컬에만 존재할 수 있다.

> 후속 실험의 설계와 실행 자료를 관리한다. 모델·검색 방식 비교와는 별도 평가셋을 사용하며, 전체 구성과 결과 요약은 [프로젝트 안내](../README.md)를 참고한다. 최신 결과와 4개 데이터셋은 [fair_eval_v4](readme_benchmark/readme_serving_retest/fair_eval_v4/README.md), 독립 실행 환경은 [v4 runtime](readme_benchmark/readme_serving_retest/fair_eval_v4/distribution/ubot-v4-runtime/README.md)을 기준으로 확인한다. 아래에는 설계 원문과 작성·실행 이력도 보존돼 있다.

v4는 Excel 원천 FAQ·KT·SKT·LG U+를 분리한 4개 데이터셋, 사용자 질문 총 1,918개로 Ollama·vLLM 측정을 완료했다. [상세 결과 비교](public_results/comparison.md)에서 구조별 검색 지표·부하·한영 표현·반복 실행 결과를 확인할 수 있다.

후속 실험의 작성·실행 이력은 **[readme_benchmark](readme_benchmark/README.md)**에 있다. 아래 설계 원문의 모델 비교 평가셋 재사용 항목은 적용하지 않는다. 후속 실험의 평가 데이터와 실행 결과는 별도로 관리한다.

UBot FAQ 검색 품질과 임베딩 서빙 구조를 검증하기 위한 실험 프로젝트입니다.

이번 실험에서는 **임베딩 모델을 `BAAI/bge-m3`로 고정**합니다.

목표는 새로운 임베딩 모델을 찾는 것이 아니라, BGE-M3를 기준으로 다음을 검증하는 것입니다.

- FAQ를 어떤 형태로 임베딩해야 하는가
- Candidate Retrieval을 어떻게 구성해야 하는가
- Reranker가 실제 검색 품질을 개선하는가
- Dual Vector가 Candidate Recall을 개선하는가
- BGE-M3의 Sparse / ColBERT-style Multi-vector가 필요한가
- Ollama / TEI / vLLM 중 어떤 서빙 엔진이 적합한가

CPU/GPU 성능 비교는 **최종 Retrieval 구조와 Serving Engine 후보를 좁힌 뒤 별도 단계에서 수행**하며, 초기 검색 품질 비교에는 포함하지 않습니다.

---

## 1. 현재 UBot 검색 구조

현재 기본 검색 구조는 다음과 같습니다.

```text
User Question
      ↓
   BGE-M3
      ↓
FAQ Question Vector
      ↓
Cosine Similarity
      ↓
    Top-K
      ↓
Threshold
      ↓
     LLM
```

FAQ Vector는 기본적으로 FAQ의 `question`만 사용합니다.

```text
faq.vector = embed(faq.question)
```

이 구조는 단순하고 빠르지만 다음 문제가 발생할 수 있습니다.

- FAQ 질문에 없는 핵심 정보가 답변에만 존재하는 경우
- 의미가 매우 비슷한 FAQ가 다수 존재하는 경우
- 상품명·서비스명 등 정확한 Entity가 존재해도 다른 FAQ가 더 높은 순위를 받는 경우
- 정답 FAQ가 Candidate Set에는 존재하지만 Top-1이 아닌 경우
- 코퍼스에 정답이 없는데 가장 비슷한 FAQ가 강제로 선택되는 경우

따라서 이번 실험은 **BGE-M3 모델 선택이 아니라 Retrieval Architecture 개선**을 대상으로 합니다.

---

## 2. 실험 원칙

### 2.1 BGE-M3 고정

이번 실험에서 Embedding Model 비교는 하지 않습니다.

```text
Embedding Model
= BAAI/bge-m3
```

기존 모델 선정 실험 결과는 참고 자료로 유지합니다.

### 2.2 기존 자산 최대 활용

기존 `UBot-EmbeddingTest`의 데이터와 코드를 최대한 재사용합니다.

예:

```text
model_benchmarks/
reranker/
intent/
clustering/
```

기존 평가 Query 역시 삭제하지 않고 Regression Test 용도로 유지합니다.

### 2.3 한 번에 하나의 구조 변경

여러 개선안을 처음부터 한꺼번에 적용하지 않습니다.

```text
Question-only Baseline
        ↓
Question + Answer
        ↓
Dual Vector
        ↓
Dual Vector + Reranker
        ↓
필요 시 Sparse / Multi-vector
```

각 단계가 실제 품질 향상에 기여했는지 개별적으로 확인합니다.

### 2.4 Threshold는 마지막에 조정

현재 운영 Threshold를 먼저 조정하지 않습니다.

```text
Retrieval Architecture 결정
        ↓
Reranker 결정
        ↓
Score / Margin 분포 확인
        ↓
Threshold Calibration
```

검색 구조가 바뀌면 score 분포 자체가 바뀔 수 있기 때문입니다.

---

# 3. 평가 체계

검색 테스트는 단일 평균 점수로 합치지 않고 **검색 목적과 실행 방식에 따라 분리**합니다.

| 검색 유형 | 정의 | 주요 지표 | 제외 / 주의 |
|---|---|---|---|
| 핵심 검색 | 일반 실제 검색. 답에 필요한 FAQ 사실이 코퍼스에 존재하고 입력 자체를 독립적으로 해석할 수 있음 | Hit, FactRecall, AllFactsHit, MRR | 개인 조회만으로 답하는 문항, 고정 Context, 통제 검색 |
| 문맥 검색 | 대화 이력이 없으면 현재 질문을 독립적으로 해석할 수 없음 | 고정된 query rewrite 결과에 대해 동일 검색 지표 사용 | 정답용 검색질문을 실제 query로 직접 넣는 실행 |
| 정답 없는 질문 | 코퍼스에 질문의 핵심 답이 존재하지 않음 | 최종 NO_FAQ precision, recall, F1, false-positive rate | 조회 데이터가 없다는 이유만으로 NO_FAQ 처리한 문항 |
| 특수 조건 | HARD_NEGATIVE, EXACT_ENTITY, COUNTERFACT, 시험 문서 공격 등 특정 실패 조건 | 조건별 재현율, 조건별 실패 수 | 일반 검색 평균에 무조건 합산하지 않음 |
| 반복 | 동일한 입력과 조건을 여러 번 반복 | 검색 결과 일관성, rank 변화, context hash 변화 | 동일 입력 10회를 독립 테스트 문항 10개로 집계하지 않음 |

## 3.1 기존 원천 항목과 새 검색 유형은 일대일 매핑하지 않음

기존 15개 원천 항목 코드를 새 검색 유형과 기계적으로 일대일 대응시키지 않습니다.

예:

```text
NC
→ HARD_NEGATIVE를 만드는 원천이 될 수 있음

CE
→ 숫자 / 조건 검증 사례의 원천이 될 수 있음

AD
→ 문서 공격(Document Attack)과 OUT_OF_SCOPE는
   실행 방식과 채점 기준이 다르므로 분리
```

기존 데이터의 원래 목적을 보존하면서 새 평가 체계에 필요한 태그를 추가합니다.

---

## 3.2 문맥 검색(MT)은 초기 핵심 비교에서 분리

MT 계열 문맥 검색은 초기 A~D Retrieval 구조 비교에서 분리합니다.

문맥 검색을 비교할 때는 먼저 다음을 고정합니다.

```text
Rewrite Model / Rule Version
        ↓
Rewritten Query
        ↓
모든 Retrieval 방식이 동일 Query 사용
```

즉 다음 두 실험을 섞지 않습니다.

```text
A. Query Rewrite 품질 비교

B. Retrieval 구조 품질 비교
```

재작성 전후 성능 차이는 별도 실험이며 Retrieval 구조 변경 효과에 합산하지 않습니다.

---

## 3.3 생성 상태와 검색 정답 존재 여부를 분리

다음 생성 상태:

```text
ANSWER
PARTIAL
ABSTAIN
CLARIFY
...
```

와 다음 검색 상태:

```text
FAQ_EXISTS
NO_FAQ
PARTIAL_FACTS_EXIST
```

는 서로 다른 필드로 관리합니다.

예:

```text
사용자의 개인 금액을 직접 조회할 수 없어
최종 응답은 ABSTAIN

하지만
"이용 금액 조회 경로"
를 설명하는 FAQ는 존재할 수 있음
```

이 경우 생성 결과는 `ABSTAIN`이어도 Retrieval GT는 존재할 수 있습니다.

`PARTIAL` 문항은 코퍼스에서 실제로 답할 수 있는 하위 사실만 Retrieval Ground Truth로 지정합니다.

---

# 4. 세부 문항 설계

## 4.1 검증 목적과 변형

| 유형 또는 태그 | 변형할 요소 | 보존할 요소 | 검증할 실패 |
|---|---|---|---|
| REAL_FAILURE | 실제 실패 입력은 원형 보존. 추가 변형은 별도 ID로 작성 | 발생 당시 코퍼스와 로그, 실패 원인 | 서비스 실패가 새 구조에서 해결되는지 |
| EXACT_ENTITY | 상품명 전체/약칭, 유사 상품명, 띄어쓰기, 숫자·단위, 조건 주체 | 실제 FAQ에 존재하는 명칭과 정책 | 비슷한 상품이나 다른 조건의 FAQ가 앞서는지 |
| HARD_NEGATIVE | 핵심 조건 하나만 바꾼 대조 질문, 높은 점수의 잘못된 후보 | 질문마다 검수된 정답 근거 | 의미가 가까운 오답과 구분하는지 |
| KR_EN_EQUIVALENT | 한글↔영문 의미 동등 표현, 영문 원어, 한글 음역, 약어, 혼합 표기, 영문 상품명과 한국어 설명의 교차 | 질문 의도, 수치, 적용 조건, 사실별 GT | 같은 의미인데 언어·표기 차이 때문에 다른 FAQ를 찾는지 |
| NEGATION | 긍정↔부정, 가능↔불가, 적용↔미적용, 사용↔미사용 | 대상 Entity와 나머지 조건 | 부정을 놓쳐 반대 의미 FAQ를 상위에 두는지 |
| CONDITION_SCOPE | 대상 연령, 가입자 유형, 지역, 요금제, 적용 대상, 예외 조건 | 핵심 정책과 비교 대상 | 조건 범위를 일반화해 잘못된 FAQ를 찾는지 |
| NUMERIC_CONDITION | 이상/이하/초과/미만, 기간, 용량, 금액, 횟수, 단위 | 숫자와 비교 조건의 의미 | 숫자는 맞지만 경계 조건이나 단위를 틀리는지 |
| COMPARISON | 두 상품/정책/조건을 비교하는 표현, 우위 질문, 차이 질문 | 비교 대상과 비교 기준 | 한쪽 FAQ만 찾거나 비교 기준과 무관한 문서를 찾는지 |
| MULTI_INTENT | 한 질문에 두 개 이상의 독립 요구 포함 | 각 요구의 원래 의미와 필요한 사실 | 하나의 의도만 검색해 필요한 근거 일부를 놓치는지 |
| AMBIGUOUS_QUERY | 지시어, 생략, 주어 없음, 대상 불명확 표현 | 실제로 정보가 부족한 상태 | 근거가 부족한데 특정 FAQ를 과도하게 확정하는지 |
| TEMPORAL_VERSION | 현재/과거/특정 시점, 변경 전후, 가입 시점, 정책 버전 | 시점 조건과 정책 사실 | 최신 정보와 과거 정책을 혼동하는지 |
| PARAPHRASE | 동의어, 구어체, 존댓말, 문장 순서, 직접 질문/상황 설명 | 의도와 필요한 사실 | 표면 표현이 달라도 같은 근거를 찾는지 |
| LONG_QUERY | 앞뒤 무관 배경, 질문 위치, 여러 요청 중 핵심 요청 | 정답 조건과 명시적인 핵심 질문 | 길이와 주변 주제로 인해 근거를 놓치는지 |
| TYPO | 한 글자 오타, 자판 오타, 조사·띄어쓰기, 명칭의 작은 오타 | 해석 가능한 의도 | 작은 표기 오류로 정답이 밀리는지 |
| NO_FAQ | 가까운 주제지만 빠진 사실, 없는 상품, 지원 밖 요청 | 코퍼스에 핵심 답이 없다는 검수 근거 | 답을 뒷받침하지 못하는 문서를 채택하는지 |
| DOCUMENT_ATTACK | FAQ/문서 내부의 명령문, 프롬프트 인젝션성 표현, 정책 무시 지시 | 검색 정답성과 원래 문서 의미 | 검색·Context 구성·생성 단계에서 공격성 문구가 영향을 주는지 |

---

## 4.2 REAL_FAILURE

실제 서비스 테스트에서 발생한 검색 실패를 그대로 보존합니다.

원본은 수정하지 않습니다.

```text
REAL_FAILURE_001
original_query: ...
corpus_version: ...
expected_faq_ids: [...]
actual_top_k: [...]
failure_reason: ...
```

추가적인 패러프레이즈나 조건 변경은 새로운 ID로 생성합니다.

```text
REAL_FAILURE_001
REAL_FAILURE_001_P1
REAL_FAILURE_001_P2
```

이를 통해 실제 장애가 새 구조에서 해결됐는지 직접 확인합니다.

---

## 4.3 EXACT_ENTITY

정확한 상품명·서비스명·요금제명이 Query에 존재하는데 Semantic Retrieval이 다른 FAQ를 높게 평가하는 문제를 검증합니다.

예:

```text
로밍패스 가격 얼마야?

5G 프리미어 에센셜 요금 알려줘

유쓰 5G 슬림이랑 유쓰 5G 라이트 차이 알려줘
```

검증 항목:

```text
Full Entity Name
Abbreviation
Spacing Variation
Similar Product Name
Number
Unit
Condition Owner
```

---

## 4.4 HARD_NEGATIVE

정답 FAQ와 의미적으로 매우 가까운 오답 FAQ를 의도적으로 포함합니다.

예:

```text
FAQ A
로밍패스 이용요금

FAQ B
해외 로밍 기본요금

FAQ C
로밍 데이터 추가요금
```

Query가 `로밍패스`를 명시한 경우 FAQ A가 경쟁 FAQ보다 위에 오는지 확인합니다.

---

## 4.5 NO_FAQ

NO_FAQ는 일반 Retrieval Accuracy와 별도로 평가합니다.

검색기는 항상 가장 가까운 문서를 반환할 수 있기 때문에, 코퍼스에 정답이 없는 질문을 제대로 거절하는 능력을 따로 측정해야 합니다.

주요 지표:

```text
NO_FAQ Precision
NO_FAQ Recall
NO_FAQ F1
False Positive Rate
```

## 4.6 KR_EN_EQUIVALENT

한국어와 영어가 섞이는 실제 사용자 입력을 별도 의미 동등성 테스트로 관리합니다.

단순 번역 문장만 만들지 않고 다음 경우를 포함합니다.

```text
한글 일반명 ↔ 영어 일반명
영문 상품명 ↔ 한국어 설명
영어 약어 ↔ 한글 풀어쓰기
영문 원어 ↔ 한글 음역
한글 + 영어 혼합
```

예:

```text
"데이터 로밍 차단"
↔
"block data roaming"

"eSIM 재발급 방법"
↔
"이심 다시 발급받는 법"

"international roaming 요금"
↔
"해외 로밍 가격"
```

검증 핵심은 문자열 번역 품질이 아니라 **동일한 의미를 가진 Query가 동일한 사실 근거를 검색하는지**입니다.

따라서 다음을 고정합니다.

```text
Intent
Entity
Number / Unit
Condition
Required Facts
```

언어 표현만 바꾸고 다른 의미 요소는 변경하지 않습니다.

측정:

```text
Cross-language Hit@K
Cross-language FactRecall
Cross-language AllFactsHit
Rank Delta
Top-K Overlap
```

가능하면 한 쌍을 다음처럼 관리합니다.

```text
pair_id: KE_001

query_ko:
"eSIM 재발급 방법 알려줘"

query_en:
"How can I get my eSIM reissued?"

required_facts:
- FAQ-123
```

한국어 Query와 영어 Query가 동일한 GT를 공유하도록 하여 언어 차이에 따른 검색 Rank 변화만 확인합니다.

---

## 4.7 NEGATION

Dense Retrieval은 단어 대부분이 동일한 긍정문과 부정문을 매우 가깝게 볼 수 있으므로 별도 검증합니다.

예:

```text
"로밍패스를 사용하면 요금이 발생해?"
"로밍패스를 사용하지 않으면 요금이 발생해?"
```

또는:

```text
"이 요금제에서 테더링 가능해?"
"이 요금제에서 테더링 불가능해?"
```

검증:

```text
부정 표현을 놓쳐 반대 정책 FAQ를 선택하는지
긍정/부정 쌍의 Rank가 충분히 분리되는지
Hard Negative가 Top-1을 차지하는지
```

---

## 4.8 NUMERIC_CONDITION

숫자 자체뿐 아니라 비교 연산과 단위를 함께 검증합니다.

예:

```text
100GB 이상
100GB 이하
100GB 초과
100GB 미만

월 8만원 이하
월 8만원 이상

30일 이내
30일 이후
```

보존:

```text
Entity
Number
Unit
Comparison Operator
Policy Context
```

검증:

```text
숫자는 맞지만 이상/이하를 뒤집는지
단위가 다른 FAQ를 잘못 선택하는지
기간 조건을 놓치는지
```

---

## 4.9 CONDITION_SCOPE

정책의 적용 범위를 구분할 수 있는지 검증합니다.

예:

```text
만 29세 이하 가입 가능
만 29세 미만 가입 가능

신규 가입자만 적용
기존 가입자도 적용

국내에서만 가능
해외에서도 가능
```

Dense Search가 전체 주제 유사도만 보고 적용 대상이 다른 FAQ를 위로 올리는지를 확인합니다.

---

## 4.10 COMPARISON

두 개 이상의 Entity를 비교하는 질문을 별도로 평가합니다.

예:

```text
"A 요금제랑 B 요금제 중 데이터 더 많은 건 뭐야?"
"A와 B의 로밍 혜택 차이가 뭐야?"
```

하나의 FAQ만으로 답할 수 없는 경우 Fact 단위 GT를 사용합니다.

```text
required_facts:
- FAQ_A
- FAQ_B
```

평가:

```text
FactRecall
AllFactsHit
첫 번째 관련 문서 Rank
두 Entity 모두 Candidate에 포함되는지
```

---

## 4.11 MULTI_INTENT

한 질문 안에 두 개 이상의 요구사항이 있는 경우를 별도 평가합니다.

예:

```text
"로밍 요금 알려주고 가까운 매장도 찾아줘"
```

이 경우 Retrieval 평가에서는 각 하위 요구를 분리합니다.

```text
intent_1: FAQ retrieval
intent_2: Store / Map
```

한쪽 기능만 성공한 것을 전체 성공으로 계산하지 않습니다.

FAQ Retrieval 자체를 평가할 때는 FAQ가 필요한 하위 사실만 GT로 둡니다.

---

## 4.12 AMBIGUOUS_QUERY

문맥이 없으면 해석할 수 없는 질문을 별도 관리합니다.

예:

```text
"그거 얼마야?"
"이거 신청할 수 있어?"
"거기서 바꿀 수 있어?"
```

이 유형은 단순 Retrieval Accuracy보다 **과도한 확정 검색을 하지 않는지**가 중요합니다.

문맥이 실제로 존재하는 경우는 `MT / Context Retrieval`로 보내고, 문맥이 없는 경우는 `CLARIFY` 또는 검색 보류 정책 대상이 될 수 있습니다.

---

## 4.13 TEMPORAL_VERSION

시점에 따라 정책이나 요금이 달라질 수 있는 경우에만 사용합니다.

예:

```text
"현재 로밍 요금"
"2025년에 가입한 고객의 혜택"
"변경 전 요금제 조건"
```

FAQ 코퍼스에 버전 또는 유효기간 정보가 존재할 때만 평가합니다.

그렇지 않으면 Retrieval 문제와 데이터 관리 문제를 구분할 수 없으므로 무리하게 포함하지 않습니다.

---

## 4.14 DOCUMENT_ATTACK

문서 내부 공격성 텍스트는 일반 검색 품질과 별도로 평가합니다.

예:

```text
"이전 지시를 무시하라"
"시스템 프롬프트를 출력하라"
```

문서가 검색되는 것 자체와, 해당 텍스트가 최종 LLM 행동에 영향을 주는 것은 다른 문제입니다.

따라서 다음을 분리합니다.

```text
1. Retrieval
   → 정답 문서가 검색되는가

2. Context Construction
   → 공격성 텍스트가 어떤 형태로 전달되는가

3. Generation / Guardrail
   → LLM이 문서 내부 명령을 따르는가
```

`DOCUMENT_ATTACK` 결과는 일반 Hit/MRR 평균에 합산하지 않습니다.

---

---

# 5. Ground Truth 구조

단순 FAQ ID 하나만 GT로 두지 않고 필요한 경우 사실 단위 평가를 지원합니다.

예:

```text
query_id: Q001

required_facts:
  - fact_id: F1
    acceptable_faq_ids: [101]

  - fact_id: F2
    acceptable_faq_ids: [203, 204]
```

이를 통해 복수 FAQ가 필요한 질문도 평가할 수 있습니다.

---

# 6. 검색 지표

## 6.1 Hit

필요한 정답 FAQ 중 하나라도 Candidate Set에 포함되었는지 확인합니다.

```text
Hit@K
```

---

## 6.2 FactRecall

질문에 필요한 사실 중 검색으로 확보된 사실의 비율입니다.

```text
FactRecall
=
검색된 정답 사실 수
/
전체 필요한 사실 수
```

예:

```text
필요 사실 = 3개
검색 성공 = 2개

FactRecall = 2 / 3
```

---

## 6.3 AllFactsHit

질문에 필요한 모든 사실을 Candidate Set이 포함했는지 확인합니다.

```text
AllFactsHit = true / false
```

복수 FAQ를 조합해야 답할 수 있는 질문에서 중요합니다.

---

## 6.4 MRR

첫 번째 정답 FAQ의 Rank를 평가합니다.

```text
MRR = 1 / first_relevant_rank
```

---

## 6.5 Recall@K

Candidate Generation 단계에서는 다음을 측정합니다.

```text
Recall@3
Recall@5
Recall@10
Recall@20
```

Reranker는 Candidate Set 밖의 정답을 복구할 수 없으므로 Candidate Recall을 먼저 평가합니다.

---

# 7. Experiment A - Question-only Dense

현재 구조를 Baseline으로 사용합니다.

```text
FAQ
└── question_vector
    = embed(question)
```

검색:

```text
User Query
    ↓
BGE-M3
    ↓
Query Vector
    ↓
Question Vector Search
    ↓
Top-K
```

모든 이후 실험의 기준점입니다.

---

# 8. Experiment B - Question + Answer Dense

FAQ의 질문과 답변을 하나의 문서로 임베딩합니다.

```text
content_vector
=
embed(
    question
    +
    answer
)
```

예:

```text
Q. 로밍패스 이용요금은 얼마인가요?

A. 로밍패스는 하루 기준 ...
```

Question-only Baseline과 동일한 Query Set으로 비교합니다.

---

# 9. Experiment C - Dual Vector

FAQ 하나를 두 개의 관점으로 표현합니다.

```text
FAQ
├── question_vector
│   └── embed(question)
│
└── content_vector
    └── embed(question + answer)
```

검색:

```text
                    ┌→ Question Vector Top-K
User Query → BGE-M3
                    └→ Content Vector Top-K
                              ↓
                         Candidate Merge
                              ↓
                            Dedupe
```

초기 실험에서는 복잡한 Weight를 추가하지 않습니다.

```text
Question Top-K
+
Content Top-K
      ↓
    Union
      ↓
    Dedupe
```

첫 번째 검증 목적은 **정답 FAQ가 Candidate Set 안으로 들어오는 비율을 높이는 것**입니다.

---

# 10. A/B/C Candidate Recall 비교

Reranker 적용 전에 다음을 비교합니다.

```text
A. Question-only

B. Question + Answer

C. Dual Vector
```

주요 지표:

```text
Hit@K
FactRecall@K
AllFactsHit@K
Recall@3
Recall@5
Recall@10
Recall@20
MRR
```

예:

```text
Question-only Recall@10
97.0%

Dual Vector Recall@10
99.5%
```

이 경우 Dual Vector가 Candidate Generation을 개선했다고 판단할 수 있습니다.

반대로 Recall 차이가 거의 없다면 Ranking 단계 개선의 우선순위가 높아집니다.

---

# 11. Experiment D - Reranker

Candidate Retrieval 이후 Reranker를 적용합니다.

기존 실험 자산을 최대한 재사용합니다.

초기 후보:

```text
BAAI/bge-reranker-v2-m3

jinaai/jina-reranker-v2-base-multilingual
```

## 11.1 Reranker 입력 비교

기존 방식:

```text
Query
↔
FAQ Question
```

추가 방식:

```text
Query
↔
FAQ Question + Answer
```

두 표현을 별도로 비교합니다.

## 11.2 반드시 기록할 변화

평균 Accuracy만 비교하지 않습니다.

각 Query마다 다음을 저장합니다.

```text
baseline_rank
reranked_rank
rank_delta

IMPROVED
WORSENED
UNCHANGED
STAGE1_MISS
```

기존 Reranker 실험에서 전체 Top-1이 오히려 하락한 실행이 있었으므로 Regression을 반드시 개별 분석합니다.

---

# 12. 실패 원인 분류

검색 실패는 최소 다음 범주로 분류합니다.

```text
RETRIEVAL_MISS
→ 정답이 Candidate Set에 없음

RANKING_ERROR
→ 정답이 Candidate Set에는 있지만 상위 Rank가 아님

RERANKER_REGRESSION
→ Reranker 적용 후 정답 Rank가 악화됨

LEXICAL_MISS
→ 정확한 Entity / Keyword가 Query에 존재하지만 정답을 놓침

AMBIGUOUS_GT
→ FAQ 또는 GT 자체가 중복되거나 모호함

NO_FAQ_FALSE_POSITIVE
→ 핵심 답이 코퍼스에 없는데 관련 없는 FAQ를 채택함

CONTEXT_REWRITE_FAILURE
→ 문맥 검색용 Query Rewrite 단계에서 의미가 손실됨

NON_DETERMINISTIC_RESULT
→ 동일 입력 반복에서 Rank / Candidate가 불안정하게 변함
```

---

# 13. Experiment E - BGE-M3 Sparse

A~D 결과만으로 충분하지 않을 경우 BGE-M3의 Sparse/Lexical 표현을 실험합니다.

목적:

```text
Exact Entity
Keyword
Product Name
Number / Unit
```

등 Dense Semantic Search가 놓치는 Lexical Signal을 보완하는 것입니다.

초기 비교:

```text
Dense Only

vs

Sparse Only

vs

Dense + Sparse
```

Dense + Sparse의 Merge 방식은 별도 실험으로 관리합니다.

---

# 14. Experiment F - BGE-M3 Multi-vector

Dual Vector와 BGE-M3 native Multi-vector는 다른 개념입니다.

## Dual Vector

FAQ 하나를 두 개의 문서 표현으로 생성합니다.

```text
question_vector

content_vector
```

## BGE-M3 Multi-vector

BGE-M3의 ColBERT-style token-level representation을 이용합니다.

```text
Document
↓
Token / Late-interaction Representation
↓
Query와 세밀한 Matching
```

A~D에서 해결되지 않는 Hard Negative나 세밀한 조건 비교에 효과가 있는지 별도 평가합니다.

Multi-vector는 구현 복잡도와 저장 비용이 증가하므로 **효과가 측정된 경우에만 운영 후보로 고려**합니다.

---

# 15. Hybrid Search 검토

처음부터 BM25 / Hybrid를 도입하지 않습니다.

우선:

```text
A. Question-only
B. Question + Answer
C. Dual Vector
D. Dual Vector + Reranker
E. 필요 시 Sparse
F. 필요 시 Multi-vector
```

순서로 평가합니다.

이후에도 다음 문제가 남는 경우 Hybrid Search를 검토합니다.

```text
LEXICAL_MISS
EXACT_ENTITY 실패
상품명 / 서비스명 충돌
숫자 / 단위 구분 실패
```

후보 구조:

```text
Dense
+
Sparse / BM25
      ↓
Candidate Merge
      ↓
Reranker
```

---

# 16. Threshold Calibration

Retrieval 구조가 결정된 뒤 Threshold를 다시 조정합니다.

단순 Top-1 Score뿐 아니라 다음을 같이 확인합니다.

```text
Top1 Score

Top1 - Top2 Margin
```

예:

```text
Case A
Top1 = 0.78
Top2 = 0.77

→ 절대 점수는 높지만 모호함
```

```text
Case B
Top1 = 0.74
Top2 = 0.50

→ 절대 점수는 조금 낮지만 상대적으로 명확함
```

NO_FAQ Set을 함께 사용하여 Threshold 변화에 따른 False Positive / False Negative를 확인합니다.

---

# 17. Serving Engine Benchmark

Retrieval 구조가 어느 정도 결정된 후 동일한 BGE-M3를 다음 엔진으로 서빙합니다.

```text
Ollama
TEI
vLLM
```

이 단계의 목적은 검색 알고리즘 정확도 비교가 아니라 **Serving 특성 비교**입니다.

초기 측정 항목:

```text
p50 latency
p95 latency
p99 latency

requests/sec
texts/sec

Concurrency
1 / 4 / 8 / 16 / 32

Memory usage
VRAM usage
Failure rate
Batching behavior
```

동일 입력에 대해 엔진별 Embedding 결과와 검색 Rank가 비정상적으로 달라지는지도 확인합니다.

CPU / GPU 성능 비교는 이 단계의 후보를 좁힌 뒤 별도 실험으로 수행합니다.

---

# 18. Calibration Set / Holdout Set

평가 데이터를 다음과 같이 구분합니다.

```text
Regression Set
→ 기존 동작이 깨지는지 확인

Calibration Set
→ Retrieval / Reranker / Threshold 조정

Holdout Set
→ 최종 구조 선택 후 마지막 평가
```

Holdout 결과를 반복적으로 보면서 파라미터를 조정하지 않습니다.

---

# 19. 반복 테스트

동일 입력을 반복 실행하는 RT 계열은 독립 문항으로 평균에 넣지 않습니다.

예:

```text
Q001
run 1
run 2
run 3
...
run 10
```

측정:

```text
Top-K Candidate Set 변화
Rank 변화
Score 변화
Context Hash 변화
Output Hash 변화
```

검색기가 결정적이어야 하는 조건에서는 동일 결과가 유지되는지 확인합니다.

---

# 20. 초기 성공 기준

초기 목표는 다음과 같이 둡니다.

| 지표 | 초기 목표 |
|---|---:|
| Recall@10 | 약 99% 이상 |
| Top-1 / Hit@1 | 95% 이상 |
| Top-3 / Hit@3 | 98% 이상 |
| EXACT_ENTITY | 별도 95% 이상 목표 |
| AllFactsHit | 복수 사실 문항 별도 추적 |
| NO_FAQ | Precision / Recall / F1 별도 평가 |
| Reranker Regression | 개별 Query 단위 기록 |
| 반복 안정성 | 동일 조건에서 Rank / Context 변화 최소화 |

이 수치는 절대적인 제품 요구사항이 아니라 초기 비교 기준입니다.

---

# 21. 전체 진행 순서

```text
1. 실제 FAQ 약 1,000개 Corpus 확정

2. 기존 Regression Query 정리

3. 실제 서비스 실패 Query 추가

4. 평가 유형 / 태그 / Ground Truth 정리
   - 핵심 검색
   - 문맥 검색
   - NO_FAQ
   - REAL_FAILURE
   - EXACT_ENTITY
   - HARD_NEGATIVE
   - KR_EN_EQUIVALENT
   - NEGATION
   - NUMERIC_CONDITION
   - CONDITION_SCOPE
   - COMPARISON
   - MULTI_INTENT
   - AMBIGUOUS_QUERY
   - 필요 시 TEMPORAL_VERSION
   - DOCUMENT_ATTACK
   - 반복

5. BGE-M3 Question-only Baseline 재측정

6. Question + Answer Dense 측정

7. Dual Vector 측정

8. Candidate Recall / FactRecall / AllFactsHit 비교

9. Reranker 재평가
   - Question
   - Question + Answer

10. 실패 유형 분석

11. 필요 시 BGE-M3 Sparse 실험

12. 필요 시 BGE-M3 Multi-vector 실험

13. 필요 시 Hybrid Search 검토

14. 최종 Retrieval 구조 선정

15. Threshold Calibration

16. Holdout 최종 평가

17. Ollama / TEI / vLLM Serving Benchmark

18. 이후 별도 CPU / GPU Benchmark

19. 효과가 검증된 구조만 UBot-BE에 반영
```

---

# 22. 핵심 원칙

```text
BGE-M3는 고정한다.

모델 교체보다 Retrieval Architecture를 검증한다.

검색과 생성 평가를 섞지 않는다.

문맥 재작성 품질과 Retrieval 품질을 섞지 않는다.

NO_FAQ를 일반 검색 정확도에 묻어버리지 않는다.

복수 사실 질문은 FAQ ID 하나가 아니라 Fact 단위로 평가한다.

실제 실패 사례를 최우선 Regression 자산으로 유지한다.

영문/한글 표기 차이는 별도 KR_EN_EQUIVALENT 쌍으로 검증한다.

부정, 숫자 경계, 조건 범위는 일반 PARAPHRASE에 묻지 않고 별도 실패 유형으로 추적한다.

Reranker는 평균 성능뿐 아니라 Regression 사례를 본다.

Dual Vector와 BGE-M3 Multi-vector를 구분한다.

Serving Engine 비교는 Retrieval 구조 검증 이후 진행한다.

CPU/GPU 비교는 마지막 별도 단계에서 진행한다.

실험에서 효과가 입증된 복잡도만 운영 코드에 반영한다.
```


## 이 저장소의 실행 설계와 현재 상태

모델 비교 평가셋에서 파생한 `run-20261006-v1`은 후속 평가셋의 실험이 아니며 참고용으로 보존합니다. 모델 비교 평가셋의 질문·분류·비율·GT 라벨을 후속 데이터에 가져오지 않습니다.

새 테스트는 [readme_benchmark](readme_benchmark/README.md) 폴더 안에서 진행합니다. 코드·데이터·결과를 모두 그 안에 두고 기존 프로젝트 폴더와 파일은 원래 위치에 보존합니다. 새 폴더 밖에는 이 README 안내만 추가합니다.

새 데이터는 사용자 README의 유형·평가 원칙과 FAQ 원문 사실만으로 작성했습니다. 새 문항 603행과 반복 실행 40행을 검증하고 split을 동결해 모델·엔진 테스트를 실행했습니다. 이전 결과로 새 구조나 threshold를 결정하지 않았습니다. 상세 결과는 신규 테스트 실행 보고서 (로컬 전용: `readme_benchmark/outputs/run-v1/실행결과.md`)에 기록합니다.

후속 요청의 3,000쌍×10변형 데이터는 [30,000행 변형 데이터](readme_benchmark/variations_30000/README.md), 세 엔진별 전체 Calibration 실행은 [30,000행 Calibration 비교](readme_benchmark/engine_calibration_30000/README.md)에 분리했습니다. 이전 결과·벡터·임계값을 재사용하지 않습니다. Calibration 완료와 상세 비교는 해당 실행 폴더의 결과 파일로 확인합니다.

사용자가 새로 제공한 `faqs.jsonl`로 작성한 3,000쌍×10개 문항은 [새 FAQ 기반 30,000행](readme_benchmark/faqs_based_30000/README.md)에 추가했습니다. 이전 변형 질문·정답은 작성 기반으로 사용하지 않습니다. 기존 Calibration은 사용자 요청으로 중지된 상태이며 새 데이터 생성 중에는 서빙 측정을 재개하지 않습니다.

새 FAQ 데이터로 요청한 Ollama·vLLM 측정은 [새 FAQ 엔진 비교](readme_benchmark/faqs_engine_ollama_vllm_30000/README.md)에서 실행합니다. 본 평가 30,000행과 한영 비교의 원문 기준 3,000행을 구분하고, 엔진별 기동·확인·전체 실행·종료 확인을 진행합니다. 이전 Calibration은 보존하며 TEI는 이번 범위에서 제외합니다.
