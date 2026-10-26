# UBot-EmbeddingTest

UBot의 FAQ 검색에 사용할 임베딩 모델과 검색 방식을 비교한 실험 기록입니다. 모델 5종 비교, 리랭커, 의도별 검색 기준, 미응답 질문 묶기, HTTP 서빙 엔진의 측정값을 보관합니다.

실험에서는 **BGE-M3 질문 벡터 검색을 기본으로 사용하고 별도 리랭커는 도입하지 않는 방향**을 선택했습니다. [UBot-BE](https://github.com/ureca-UBot/UBot-BE)는 이 구성을 사용하며, Ollama와 vLLM의 벡터를 프로필별로 구분합니다. HDBSCAN 실험과 유사도 기준 후보를 그대로 적용한 것은 아니므로, 아래 [BE 반영 내용](#be-반영-내용)에 구현과 차이를 함께 적었습니다.

정리 기준: 실험 저장소 `ca61493`, BE `develop`의 `4a7a546f`, 확인일 2026-10-26. 표의 수치는 저장된 CSV·JSON과 보고서를 기준으로 정리했습니다.

## 실험과 원본 자료

| 항목 | 비교한 내용 | 보고서·결과 |
|---|---|---|
| 임베딩 모델 | Qwen3-Embedding, BGE-M3, E5, GTE, EmbeddingGemma | [초기 60문항 보고서](model_benchmarks/docs/임베딩모델_벤치마크_최종비교.md), [보정 후 235문항 CSV](model_benchmarks/outputs/benchmark_results_1000_fixed_summary.csv) |
| 리랭커 | 임베딩 단독과 BGE·Jina 재정렬, 후보 수·버퍼 크기 | [235문항 최종 정정](reranker/docs/임베딩_vs_reranker_비교결과_1000.md#9-최종-정정-수정-후-결과-235개-쿼리-기준), [버퍼 비교](reranker/docs/reranker_버퍼_테스트_결과.md) |
| 의도별 검색 | 코퍼스 600·1,000·1,385건과 검색 방식 5종 | [종합 보고서](intent/docs/intent_routing_test_summary.md), [원본 실행 기록](intent/outputs/) |
| 미응답 질문 묶기 | BGE-M3와 HDBSCAN, 질문 38개 | [보고서](clustering/docs/BGE-M3_클러스터링_최종결과.md), [측정 CSV](clustering/outputs/clustering_results_summary.csv) |
| 서빙 엔진 | Ollama·TEI·vLLM의 HTTP 임베딩 처리 | [비교 조건·결과](serving-benchmark/README.md), [측정 JSON](serving-benchmark/results/) |

## 임베딩 모델 선택

FAQ 1,000건과 질문 235개를 사용한 **정답 라벨 보정 후 결과**입니다. 같은 의미의 FAQ가 여러 ID로 존재하는 경우를 정답으로 인정하도록 평가 데이터를 수정했습니다. 모델은 GPU에서 `sentence-transformers`로 직접 실행했습니다.

| 모델 | Top-1 | Top-3 | Top-5 | MRR | 임베딩 p95 |
|---|---:|---:|---:|---:|---:|
| BGE-M3 | 94.5% | 97.4% | 98.3% | 0.963 | 27ms |
| multilingual-e5-large | 93.2% | 95.7% | 97.0% | 0.948 | 37ms |
| EmbeddingGemma-300M | 91.9% | 96.2% | 96.6% | 0.942 | 72ms |
| gte-multilingual-base | 90.6% | 94.0% | 96.2% | 0.928 | 36ms |
| Qwen3-Embedding-0.6B | 89.4% | 94.0% | 96.2% | 0.921 | 44ms |

Top-K는 정답 FAQ가 상위 K개 안에 포함된 비율이고, MRR은 정답의 순위 역수를 평균한 값입니다. 모두 FAQ 검색 순위를 평가한 지표입니다. 이 데이터에서는 BGE-M3가 검색 정확도와 MRR에서 가장 높았습니다. 다만 BGE-M3도 기존 Hard·Medium 질문의 Top-1은 각각 66.7%·44.4%여서 전체 평균만으로 어려운 질문의 품질을 판단할 수는 없습니다.

출처는 [요약 CSV](model_benchmarks/outputs/benchmark_results_1000_fixed_summary.csv)와 [질문별 결과](model_benchmarks/outputs/benchmark_results_1000_fixed_detail.csv)입니다. 초기 [60문항 결과](model_benchmarks/outputs/benchmark_results_unified_summary.csv)의 BGE-M3 Top-1은 86.7%입니다. FAQ 규모·질문·정답 라벨이 달라진 실험이므로 86.7% → 94.5%를 모델 개선 폭으로 해석하지 않습니다. 위 지연도 HTTP 서빙 시간과는 별도 측정입니다.

## 리랭커를 추가하지 않은 이유

초기 57문항에서는 BGE 리랭커의 MRR이 0.846 → 0.862로 올라 도입을 권장했습니다. 이후 235문항으로 늘리고 중복 FAQ의 정답 라벨을 보정하자 결론이 바뀌었습니다.

| 방식 | Top-1 | MRR | 순위 개선 / 악화 |
|---|---:|---:|---:|
| BGE-M3 단독 | 94.5% | 0.963 | 기준 |
| BGE-M3 + BGE reranker v2-m3, 후보 10개 | 91.9% | 0.954 | 10 / 10건 |
| BGE-M3 + Jina reranker v2, 후보 10개 | 92.8% | 0.953 | 5 / 9건 |

[최종 정정 보고서](reranker/docs/임베딩_vs_reranker_비교결과_1000.md#9-최종-정정-수정-후-결과-235개-쿼리-기준)와 [원본 CSV](reranker/outputs/reranker_comparison_summary_1000.csv)의 보정 후 `run=2,3`을 기준으로 정리했습니다. 같은 CSV의 `run=1`은 보정 전 결과입니다. 초기 보고서의 도입 권장 문장만 읽으면 최종 결론과 반대로 이해하게 됩니다.

[버퍼 실험](reranker/docs/reranker_버퍼_테스트_결과.md)에서는 최종 5개를 고를 때 BGE 리랭커에 후보 10개를 주면 Hit@5가 98.3% → 99.6%로 높아졌지만, MRR은 0.961 → 0.953으로 낮아졌습니다. 후보 포함률의 이득은 있었으나 정답을 앞에 놓는 성능까지 좋아진 것은 아니었습니다. 현재 BE의 FAQ 후보 검색에는 이 리랭커·버퍼 단계를 넣지 않았습니다.

## 의도별 검색과 유사도 기준

일반·매장·사용자 정보·무관 질문을 각각 100개씩, 총 400개 사용했습니다. 코퍼스 600·1,000·1,385건에서 dense, hybrid, hybrid+reranker, hybrid+ColBERT, dense+reranker를 비교했습니다.

코퍼스 1,000건의 dense 검색에서 보고한 기준 후보는 다음과 같습니다.

| 유사도 기준 | 일반 재현율 | 매장 재현율 | 사용자 정보 재현율 | 무관 질문 배제율 |
|---|---:|---:|---:|---:|
| 0.605 | 93% | 91% | 98% | 82% |
| 0.730 | 71% | 70% | 80% | 100% |

이 실험에서는 dense 검색을 선택하고, 재현율을 우선할 때 0.605, 무관 질문 배제를 우선할 때 0.730을 후보로 남겼습니다. hybrid는 점수 범위가 달라 기존 기준을 그대로 적용한 비교에 한계가 있었고, 나머지 재정렬 방식도 각자의 점수와 기준으로 평가했습니다. 모든 검색 방식에 같은 유사도 숫자를 적용할 수는 없습니다.

출처: [의도 라우팅 보고서](intent/docs/intent_routing_test_summary.md). 0.605·0.730은 당시 실험의 후보이며, **현재 BE의 `CHAT_CONFIDENCE_THRESHOLD` 기본값은 0.79**입니다.

## 미응답 질문 묶기

BGE-M3 벡터와 HDBSCAN으로 주제 4개·노이즈 8개를 포함한 질문 38개를 묶었습니다.

| 최소 묶음 크기 | 생성 묶음 | 노이즈 | ARI | NMI |
|---|---:|---:|---:|---:|
| 2 | 8 | 12 | 0.532 | 0.747 |
| 3 | 3 | 16 | 0.370 | 0.573 |
| 4 | 2 | 21 | 0.188 | 0.389 |
| 5·6 | 0 | 38 | 0.000 | 0.000 |

최소 크기 2에서는 서로 다른 주제를 섞지 않았지만, 정답 주제 4개가 8개로 나뉘고 실제 주제 질문 4개가 노이즈로 빠졌습니다. [실험 결론](clustering/docs/BGE-M3_클러스터링_최종결과.md)은 동작 확인이며 운영 파라미터 확정은 보류했습니다.

**BE에는 HDBSCAN을 적용하지 않았습니다.** 미응답 질문은 같은 임베딩 프로필의 대기·보류 묶음 중 가장 가까운 중심 벡터를 찾고, `UNANSWERED_GROUP_THRESHOLD` 이상이면 합칩니다. 기준을 넘는 묶음이 없으면 새 묶음을 만들며 중심 벡터는 질문 벡터의 평균으로 갱신합니다. 기본 기준은 0.6이고 HDBSCAN의 최소 묶음 크기와는 다른 설정입니다.

## HTTP 서빙 엔진 비교

Ollama와 vLLM에 질문 묶음 4종 × 동시성 1·4·8·16·32 × 구간별 64건을 보냈습니다. 총 40개 구간, 2,560건에서 관측 실패는 0건이었습니다.

동시성 32의 요청 처리량입니다. RPS는 초당 완료한 HTTP 요청 수입니다.

| 질문 묶음 | Ollama RPS | vLLM RPS | 비율 |
|---|---:|---:|---:|
| Existing | 44.52 | 132.73 | 2.98배 |
| KT | 33.20 | 116.42 | 3.51배 |
| SKT | 38.47 | 124.15 | 3.23배 |
| LG U+ | 60.76 | 226.68 | 3.73배 |

출처: [원본 측정 JSON](serving-benchmark/results/ollama-vllm-load-metrics.json). 모델 계열은 BGE-M3지만 형식·정밀도가 같지 않아 이 비율을 서빙 엔진 하나만의 효과로 볼 수는 없습니다.

별도 [Ollama·TEI·vLLM 탐색](serving-benchmark/results/ollama-tei-vllm-exploratory.json)에서는 기준 벡터 일치 조건을 TEI·vLLM이 통과했고 Ollama는 통과하지 못했습니다. 조건을 통과한 엔진 중 요청당 문장 1개의 최대 처리량을 기준으로 vLLM이 첫 후보였습니다. 각 구간 1회인 단기 측정이며 원본에도 운영 채택은 `false`로 기록돼 있습니다. 현재 BE의 연결 구현은 아래 코드에서 확인할 수 있습니다.

## BE 반영 내용

아래는 BE `develop`의 `4a7a546f`에서 확인한 구현입니다. 모델·검색 방식 실험과 현재 구현의 대응을 적었으며, BE에서 추가한 프로필 관리와 후속 검색 실험도 함께 연결했습니다.

| 실험·검토 항목 | BE에서 확인한 내용 | 코드·설정 |
|---|---|---|
| BGE-M3 선택 | Ollama는 `bge-m3:567m`, vLLM은 `BAAI/bge-m3`를 사용합니다. 저장 벡터는 1,024차원입니다. | [.env.example](https://github.com/ureca-UBot/UBot-BE/blob/4a7a546faa189946bb48bb14e364750d653b0774/.env.example), [EmbeddingClient](https://github.com/ureca-UBot/UBot-BE/blob/4a7a546faa189946bb48bb14e364750d653b0774/src/main/java/com/ubot/embedding/client/EmbeddingClient.java) |
| dense 검색과 리랭커 제외 | 활성 FAQ 질문 표현의 `QUESTION` 벡터로 검색하며 별도 리랭커 없이 후보 순서를 유지합니다. | [FaqVectorService](https://github.com/ureca-UBot/UBot-BE/blob/4a7a546faa189946bb48bb14e364750d653b0774/src/main/java/com/ubot/faq/service/FaqVectorService.java), [FaqRetrievalService](https://github.com/ureca-UBot/UBot-BE/blob/4a7a546faa189946bb48bb14e364750d653b0774/src/main/java/com/ubot/faq/retrieval/FaqRetrievalService.java) |
| 의도별 유사도 실험 | 후보 Top-K 기본값은 3, 질문 유사도 기준 기본값은 0.79입니다. 실험 후보 0.605·0.730과 다릅니다. | [application.yml](https://github.com/ureca-UBot/UBot-BE/blob/4a7a546faa189946bb48bb14e364750d653b0774/src/main/resources/application.yml), [ChatAnswerProcessor](https://github.com/ureca-UBot/UBot-BE/blob/4a7a546faa189946bb48bb14e364750d653b0774/src/main/java/com/ubot/chat/service/ChatAnswerProcessor.java) |
| vLLM 서빙 연결 | OpenAI 호환 임베딩 클라이언트와 별도 모델 컨테이너를 사용합니다. 기본 모드는 Ollama이고 `AI_MODE=vllm`으로 전환합니다. | [EmbeddingConfig](https://github.com/ureca-UBot/UBot-BE/blob/4a7a546faa189946bb48bb14e364750d653b0774/src/main/java/com/ubot/embedding/config/EmbeddingConfig.java), [모델 Compose](https://github.com/ureca-UBot/UBot-BE/blob/4a7a546faa189946bb48bb14e364750d653b0774/infra/llm/docker-compose.yml) |
| 엔진별 벡터 차이 관리 | 공급자·API 모델명·차원·프로필 버전으로 벡터를 분리합니다. 엔진 전환 후 현재 프로필의 누락 벡터를 백필합니다. | [EmbeddingProfileService](https://github.com/ureca-UBot/UBot-BE/blob/4a7a546faa189946bb48bb14e364750d653b0774/src/main/java/com/ubot/embedding/service/EmbeddingProfileService.java), [백필 안내](https://github.com/ureca-UBot/UBot-BE/blob/4a7a546faa189946bb48bb14e364750d653b0774/docs/how-to/llm-module.md#백필) |
| HDBSCAN 검토 | 알고리즘 자체는 미적용입니다. BE는 유사도 기반 최근접 묶음 선택과 중심 벡터 평균 갱신을 사용합니다. | [UnansweredQuestionService](https://github.com/ureca-UBot/UBot-BE/blob/4a7a546faa189946bb48bb14e364750d653b0774/src/main/java/com/ubot/unanswered/service/UnansweredQuestionService.java), [묶음 벡터 저장소](https://github.com/ureca-UBot/UBot-BE/blob/4a7a546faa189946bb48bb14e364750d653b0774/src/main/java/com/ubot/unanswered/repository/UnansweredQuestionVectorRepository.java) |

BE의 FAQ 검색은 이후 별도 측정을 거쳐 pgvector HNSW의 `ef_search=200`, `iterative_scan=strict_order`를 사용합니다. 프로필·FAQ 버전·유형·삭제 여부를 조건으로 적용하고, 같은 FAQ의 질문 표현을 중복 제거한 뒤 후보가 모자라면 탐색 후보 수를 늘립니다. 이 변경의 근거는 이 저장소의 모델 벤치마크가 아니라 BE의 [FAQ 검색·답변 실험](https://github.com/ureca-UBot/UBot-BE/blob/4a7a546faa189946bb48bb14e364750d653b0774/docs/FAQ_검색_테스트_보고서.md)과 후속 [HNSW 실험](https://github.com/ureca-UBot/UBot-BE/blob/4a7a546faa189946bb48bb14e364750d653b0774/docs/FAQ_HNSW_검색_테스트_보고서.md)에 있습니다.

## 실험 실행

저장소 루트에서 실행하는 예시입니다. 필요한 패키지는 `torch`, `sentence-transformers`, `FlagEmbedding`, `numpy`, `requests`, `openpyxl`, `psutil`, `scikit-learn`이며, 저장된 측정은 GPU 환경을 사용했습니다. 의존성 버전 고정 파일은 포함돼 있지 않습니다. 원본 `outputs/`에 실행 결과를 덧붙이지 않도록 별도 출력 폴더를 사용합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch sentence-transformers FlagEmbedding numpy requests openpyxl psutil scikit-learn
New-Item -ItemType Directory -Force local-results

python model_benchmarks/scripts/benchmark_embeddings_unified_1000.py --xlsx model_benchmarks/faq/EmbeddingTestFAQ_통합.xlsx --models bge-m3 --output local-results/model-bge-m3

python reranker/scripts/compare_embedding_reranker.py --xlsx reranker/faq/EmbeddingTestFAQ_통합.xlsx --top-k 10 --rerankers bge-reranker-v2-m3 --output local-results/reranker

python intent/scripts/intent_classification_bge.py --corpus intent/faq/faq_intent_corpus_1000.csv --test-set intent/faq/intent_test_set.csv --no-hybrid --k 3 --thresholds 0.600:0.795:0.005 --output-dir local-results/intent

python clustering/scripts/test_clustering_bge_m3.py --xlsx clustering/faq/EmbeddingTestFAQ_통합.xlsx --sheet '미등록 클러스터링' --min-cluster-size '2,3,4,5,6' --output local-results/clustering
```

모델·리랭커 스크립트는 통합 문서에 실제로 들어 있는 질문 시트를 읽습니다. 재실행 때는 출력된 FAQ 수·질문 수와 정답 라벨을 기록해야 위 표와 비교할 수 있습니다. 파일명만 같다고 과거 60·57·235문항 실험과 입력이 같아지는 것은 아닙니다.

`intent/scripts/IntentClassificationAnalysis.java`는 BE의 Spring·DB 테스트 문맥을 사용하는 자료로, 독립 Java 실행 프로그램은 아닙니다. HTTP 서빙 비교는 결과 JSON을 보존한 것으로 당시 실행기는 포함하지 않습니다.
