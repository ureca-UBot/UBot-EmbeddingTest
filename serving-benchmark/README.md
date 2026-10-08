# 임베딩 서빙 엔진 비교

이 폴더는 BGE-M3 임베딩의 **HTTP 서빙 엔진 비교 결과만** 보존합니다. 검색 구조, 리랭커, 임계값 조정, 평가 데이터 생성과 공개 정책은 이 커밋의 범위에 포함하지 않습니다.

## 보존한 결과

### Ollama와 vLLM 단기 부하 비교

[원본 측정값](results/ollama-vllm-load-metrics.json)은 네 개의 분리된 질문 묶음에서 두 엔진을 비교한 결과입니다.

- 엔진: Ollama, vLLM
- 모델 계열: BGE-M3
- 요청: 질문 묶음별·엔진별·동시성별 64건
- 동시성: 1, 4, 8, 16, 32
- 전체 측정: 40개 구간, 2,560건
- 기록 항목: 처리량, p50·p95·p99 지연, 성공·실패, Top-1 일치와 최소 임베딩 cosine
- 관측 실패: 0건

동시성 32에서 관측한 요청 처리량은 다음과 같습니다.

| 질문 묶음 | Ollama RPS | vLLM RPS | vLLM / Ollama |
|---|---:|---:|---:|
| Existing | 44.52 | 132.73 | 2.98배 |
| KT | 33.20 | 116.42 | 3.51배 |
| SKT | 38.47 | 124.15 | 3.23배 |
| LG U+ | 60.76 | 226.68 | 3.73배 |

### Ollama·TEI·vLLM 탐색 비교

[탐색 결과](results/ollama-tei-vllm-exploratory.json)는 원본 서빙 계획에서 엔진 정보, 측정 결과와 한계만 분리한 자료입니다.

- BGE-M3 dense question embedding 구성만 비교했습니다.
- 각 엔진은 동시성 5개와 요청당 문장 수 3개를 조합한 15개 구간을 실행했습니다.
- Ollama는 기준 벡터와의 일치 조건을 통과하지 못했고, TEI와 vLLM은 통과했습니다.
- 해당 탐색의 선정 규칙에서는 vLLM이 첫 후보였지만 Production 채택 결과는 아닙니다.

## 해석 범위

- 두 결과는 짧은 HTTP 임베딩 부하 진단입니다. 전체 RAG 응답 시간이나 장시간 처리량을 측정하지 않았습니다.
- Ollama와 vLLM의 모델 형식·정밀도가 같지 않으므로 엔진 하나만의 효과로 해석하지 않습니다.
- 첫 탐색은 구간별 1회 측정이며 운영 SLO가 없습니다.
- 모델 콜드 스타트, 서버 내부 배칭 추적, 리랭커 서빙과 UBot 종단간 지연은 측정 범위 밖입니다.
- 이 커밋에는 당시 비공개 입력과 결합된 실행기가 없습니다. JSON은 기존 측정값과 조건을 보존하는 근거 자료입니다.

## 출처

- 원본 커밋: `8ad2aab49095ad2efd8387a649d4d5fd1d76533f`
- Ollama/vLLM 원본 경로: `experiments/public_results/load_metrics.json`
- 세 엔진 탐색 원본 경로: `experiments/model_benchmarks/configs/serving_engine_plan.json`
- BGE-M3 리비전: `5617a9f61b028005a4858fdac845db406aefb181`
