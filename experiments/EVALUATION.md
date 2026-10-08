# 후속 실험 평가 방법

Source: SK Telecom / KT / LG U+ public FAQ pages, used for non-commercial research/evaluation

FAQ 원문과 평가 질문은 private/local 입력이다. 공개되는 것은 아래 방법과 집계 수치이며, 본문과 사용자 질문·정답 행은 포함하지 않는다.

## 독립 데이터셋

| 데이터셋 | FAQ 수 | 사용자 질문 | 개발 / Calibration / Holdout | 검색 지표 채점 문항 |
|---|---:|---:|---|---:|
| Excel 원천 FAQ | 1,024 | 340 | 170 / 85 / 85 | 255 |
| KT | 1,099 | 769 | 385 / 192 / 192 | 747 |
| SKT | 1,305 | 619 | 309 / 155 / 155 | 597 |
| LG U+ | 131 | 190 | 95 / 48 / 47 | 168 |

통신사별 코퍼스는 임베딩 입력 단계부터 분리한다. FAQ와 사용자 질문도 별도 파일로 관리하며, 각 질문은 정답 FAQ ID 집합을 가진다. 같은 사실을 만족하는 대체 FAQ는 OR, 질문에 필요한 여러 사실은 AND로 채점한다. 연결된 변형 문항은 같은 split에 둔다.

## 검증 목적

일반 패러프레이즈, 정확한 명칭, 유사 오답, 한영 동등 표현, 부정, 조건 범위, 수치 조건, 비교, 복수 의도, 모호한 질문, 긴 질문, 오타, 정답 부재, 부분 답변, 대화 문맥을 구분한다. 개인·실시간 조회가 필요한 요구는 별도 분류한다. 반복 입력과 문서 공격은 독립 fixture로 실행한다. 실제 장애 로그 및 시간별 정책 버전이 확보되지 않은 항목을 실제 검증 완료로 주장하지 않는다.

## 검색 구조

- A: FAQ 질문의 dense 검색.
- B: FAQ 질문+답변의 dense 검색.
- C-fixed20: A·B 후보를 합쳐 최대 cosine으로 정렬한 뒤 20개로 제한.
- C-raw-union: A 20개와 B 20개의 합집합, 최대 40개. 일반 top-1과 후보 예산이 다르다.
- D: C 원시 합집합에 BGE 또는 Jina reranker 적용. 질문 / 질문+답변 입력을 각각 비교.
- E: sparse 검색 및 dense+sparse RRF.
- F: 전체 코퍼스에 대한 ColBERT 방식 multi-vector 검색.

임베딩은 BGE-M3로 고정한다. Ollama는 GGUF F16, vLLM은 HF float16을 사용한다. Reranker는 공통 HF/PyTorch 모델, sparse·multi-vector는 공통 FlagEmbedding native head로 실행한다. 후자의 결과를 각 서빙 엔진이 해당 모델을 직접 제공한 성능으로 해석하지 않는다.

Reranker 입력 창은 query/document 쌍 기준 최대 1,024토큰, 문서 겹침은 128토큰이다. 창별 raw logit 최댓값으로 문서 점수를 정하며, 동률이면 먼저 나온 창을 선택한다. 긴 문서는 창이 많아지는 효과가 있다.

## 지표와 측정 범위

- Hit@K: 상위 K개에 허용 정답 FAQ가 하나 이상 포함된 비율.
- FactRecall@K: 필요한 지원 사실 중 검색된 사실의 비율.
- AllFactsHit@K: 필요한 지원 사실을 모두 검색한 문항의 비율.
- MRR@K: 첫 정답 순위의 역수 평균.

NO_FAQ·모호한 질문·순수 개인 조회는 검색 정확도 분모에서 제외한다. PARTIAL은 코퍼스가 지원하는 사실만 채점한다. 문항 평균과 연결 그룹 평균을 별도로 제공한다. 이 지표는 최종 생성 답변의 정확도를 측정하지 않는다.

엔진은 데이터셋마다 기동 → 확인 → 측정 → 종료 → 종료 확인 순으로 실행했다. HTTP 임베딩 부하는 웜업 후 64개 질문으로 동시성 1/4/8/16/32를 측정했다. 전체 RAG의 종단 지연이나 장시간 SLA 실험은 아니다. 반복 검사는 엔진·데이터셋별 10질문×10회다.

정답 라벨의 독립 의미 검수는 미완료이며, v4에서는 임계값을 피팅하지 않았다. Holdout 수치를 확인한 진단 결과이므로 이를 이용해 튜닝한 뒤 같은 holdout을 최종 검증처럼 제시하지 않는다.

[집계 결과](public_results/README.md) · [실행 환경](readme_benchmark/readme_serving_retest/fair_eval_v4/distribution/ubot-v4-runtime/README.md) · [데이터 공개 정책](DATA_POLICY.md)
