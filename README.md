# UBot Embedding Test

FAQ 검색을 위한 임베딩 모델·검색 방식 비교와, 별도 평가셋으로 진행한 후속 실험을 정리한다.

## 모델·검색 방식 비교

### 임베딩 모델 비교

FAQ 100개·평가 질문 60개를 대상으로, 각 모델을 sentence-transformers와 GPU로 실행해 검색 성능을 비교했다.

| 모델 | Top-1 | Top-3 | Top-5 | MRR |
|---|---:|---:|---:|---:|
| **BGE-M3** | **86.7%** | **95.0%** | **98.3%** | **0.916** |
| EmbeddingGemma-300M | 83.3% | 93.3% | 96.7% | 0.894 |
| multilingual-e5-large | 81.7% | 93.3% | 95.0% | 0.877 |
| gte-multilingual-base | 76.7% | 86.7% | 91.7% | 0.832 |
| Qwen3-Embedding-0.6B | 75.0% | 90.0% | 96.7% | 0.844 |

이 모델 선정 실험에서는 BGE-M3가 검색 정확도 지표에서 가장 높았다. [모델 비교 보고서](model_benchmarks/docs/임베딩모델_벤치마크_최종비교.md)

### 임베딩 단독과 Reranker 비교

FAQ 1,000개·평가 질문 235개, `top_k=10` 조건의 **정답 보정 후 결과**다. 중복 의미의 FAQ ID를 함께 정답으로 인정하도록 수정한 최종 정정 수치를 사용했다.

| 구성 | Top-1 | MRR |
|---|---:|---:|
| **BGE-M3 단독** | **94.5%** | **0.963** |
| BGE-M3 + BGE reranker v2-m3 | 91.9% | 0.954 |
| BGE-M3 + Jina reranker v2 | 92.8% | 0.953 |

이 평가셋에서는 임베딩 단독이 가장 높았다. 57문항 초기 결과에서 제시했던 reranker 도입 권장은 확장·정답 보정 후 변경됐다. [최종 정정 결과](reranker/docs/임베딩_vs_reranker_비교결과_1000.md#9-최종-정정-수정-후-결과-235개-쿼리-기준)

별도 버퍼 실험에서는 후보 10개를 BGE reranker로 재정렬해 최종 5개를 선택했을 때 Hit@5가 **98.3% → 99.6%**, MRR은 **0.961 → 0.953**이었다. 이 수치는 위 `top_k=10` 비교와 별도 실험 결과다. [버퍼 실험 보고서](reranker/docs/reranker_버퍼_테스트_결과.md)

### 의도 라우팅·클러스터링

| 실험 | 조건 | 결과 |
|---|---|---|
| 의도 라우팅 | BGE-M3 dense-only, 코퍼스 1,000개, 질문 400개, threshold 0.730 | 일반·매장·사용자 재현율 71%·70%·80%, 무관 정확도 100% |
| 클러스터링 | BGE-M3 + HDBSCAN, 질문 38개, min_cluster_size=2 | 클러스터 8개, 노이즈 12개, ARI 0.532, NMI 0.747 |

세부 조건은 [의도 라우팅 보고서](intent/docs/intent_routing_test_summary.md)와 [클러스터링 보고서](clustering/docs/BGE-M3_클러스터링_최종결과.md)를 참고한다. 클러스터링 결과는 소규모 동작 확인이며 운영 파라미터 확정 결과는 아니다.

## 후속 실험

별도로 구성한 4개 평가셋으로 BGE-M3 검색 구조와 Ollama·vLLM 서빙을 비교했다. 위의 모델·검색 방식 비교와는 코퍼스·질문·정답 체계가 다른 실험이며, 두 실험 사이의 수치 차이는 개선 폭을 의미하지 않는다.

v4에서 **질문 검색과 질문+답변 검색의 후보를 합친 뒤 BGE reranker로 질문+답변을 재정렬하는 구성**이 네 평가셋 모두 Hit@1이 가장 높았다.

| 평가셋 | 질문+답변 Dense Hit@1 (vLLM) | BGE reranker 적용 Hit@1 (vLLM) |
|---|---:|---:|
| Excel 원천 FAQ | 65.49% | **68.24%** |
| KT | 68.27% | **82.33%** |
| SKT | 80.23% | **92.29%** |
| LG U+ | 86.31% | **88.10%** |

해당 reranker 구성의 Hit@1은 Ollama와 vLLM이 같았다. 동시 요청 32개 조건의 임베딩 HTTP 처리량은 vLLM이 Ollama보다 약 **2.98~3.73배** 높았다. Reranker는 공통 HF/PyTorch 모델로 실행했으며, 처리량은 임베딩 요청만 측정한 값이다.

Hit@1은 답변 근거가 있는 문항에서 첫 번째 검색 결과가 정답 FAQ인 비율이다. 정답 라벨은 독립 검수 전이며, 임계값은 피팅하지 않았다.

상세 내용은 [전체 결과 비교](experiments/public_results/comparison.md), [평가 방법·검증 기준](experiments/EVALUATION.md), [실행 환경·재현 방법](experiments/readme_benchmark/readme_serving_retest/fair_eval_v4/distribution/ubot-v4-runtime/README.md)을 참고한다.

## 데이터 출처와 공개 범위

Source: SK Telecom / KT / LG U+ public FAQ pages, used for non-commercial research/evaluation

공개 저장소에는 테스트 스크립트, 평가 방법, 실행 환경과 본문 없는 집계 수치를 제공한다. FAQ 원문·평가 질문·근거·상세 로그·데이터 ZIP은 private/local로 관리하며 재배포하지 않는다. 공개 결과에는 FAQ 원문을 인용하지 않았다.

문항 추적에는 [FAQ ID와 source URL](experiments/public_results/faq_sources.json)만 제공한다. 재실행에는 별도로 준비한 로컬 데이터가 필요하다. [데이터 관리 안내](experiments/DATA_POLICY.md)

## 저장소 구성

```text
UBot-EmbeddingTest/
├── README.md                  # 프로젝트 안내
├── clustering/                # 클러스터링 조건 비교
├── intent/                    # 의도 라우팅 방식 비교
├── model_benchmarks/           # 임베딩 모델 비교
├── reranker/                  # Reranker 적용·후보 수 비교
└── experiments/               # 후속 실험
```

모델·검색 방식 비교 코드와 보고서는 각 주제 폴더에, 후속 실험의 코드·환경·평가 방법은 `experiments/`에 있다. 공개 집계 결과는 `experiments/public_results/`에서 확인한다.

## 자료·실행 안내

- [후속 실험 설계·자료](experiments/README.md)
- [실행 코드·환경 ZIP](experiments/readme_benchmark/readme_serving_retest/fair_eval_v4/distribution/UBot_v4_실행코드_환경.zip)
- [집계 결과·출처 목록](experiments/public_results/README.md)
- [Private/local 데이터 관리·공개 전 검사](experiments/DATA_POLICY.md)

실행은 각 실험 폴더에서 진행한다. 입력 해시 보존을 위해 `experiments/`는 Git 체크아웃의 줄바꿈 변환을 적용하지 않는다.
