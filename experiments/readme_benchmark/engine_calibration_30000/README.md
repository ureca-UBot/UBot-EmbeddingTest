# 30,000행 엔진별 Calibration 구조 비교

사용자가 지정한 3,000쌍에서 각 10개씩 생성한 `variations_30000` 전체를 사용한다. 원본 15유형을 각각 2,000행으로 보존하고, RT 2,000행도 실제 재계산한다. 기존 603문항 및 원본 3,000문항 실행의 벡터·점수·임계값은 가져오지 않는다.

`run_all.ps1`은 Ollama → vLLM → TEI 순서로 각 엔진을 기동하고 readiness를 확인한다. 각 엔진 HTTP API에서 30,000 query와 1,024 FAQ의 두 view를 계산한다. 서빙 모델 종료·HTTP 폐쇄를 확인한 뒤 같은 엔진의 후보로 공통 CUDA 보조 모델을 실행하고, 해당 세트가 끝나면 다음 엔진으로 넘어간다. GPU 메모리를 확보하기 위해 dense 서빙 모델과 보조 모델을 동시에 상주시켜 두지 않는다.

A 질문 dense, B 질문+답변 dense, C dual Top-20, D BGE/Jina×질문/질문+답변 네 reranker 조합, E sparse 단독 및 dense+sparse RRF, F full 및 dense-pool MaxSim까지 엔진당 11개 구조다. D는 조합당 600,000 pairs를 입력 중복 제거 없이 계산한다. E sparse와 F full은 공통 native 대조군이다. D/E/F를 해당 엔진 자체 API가 서빙했다고 주장하지 않는다.

D reranker는 세 엔진 모두 공통 CUDA FP16으로 실행하며 FP32 대비 표본 raw-logit 차이를 기록한다. E/F는 공통 CUDA FP32다. FP16과 FP32가 같은 점수 또는 순위를 준다고 가정하지 않는다. [BGE 공식 모델 카드](https://huggingface.co/BAAI/bge-reranker-v2-m3)는 FP16 설정의 속도 및 품질 차이 가능성을 설명한다. 긴 질문을 줄이지 않고 전체 재정렬 720만 쌍을 실행하기 위해 D의 정밀도를 공통으로 고정했다.

질문을 자르지 않는다. 최대 602토큰 입력을 모두 보존하기 위해 이번 실행의 검증 길이는 1,024토큰이다. Ollama/vLLM은 1,024토큰 설정, TEI는 원래 8,192토큰 모델 설정을 쓰되 입력을 공통 범위로 검증한다. Reranker pair도 1,024토큰 이내인지 확인하고 truncation을 비활성화한다. 512토큰 초과 2,340행을 제거하거나 질문 앞부분만 잘라 넣지 않는다.

입력에는 생성된 질문과 고정 원본 대화 이력만 들어간다. 제공 Context·정답·source ID·API JSON·persona는 검색 입력이 아니다. 문서 공격 66개 fixture와 생성/API 동작의 성공 여부는 이 검색 실험으로 검증하지 않는다.

추적 FAQ ID를 이용한 SourceHit·SourceCoverage·AllSourcesHit를 집계한다. 독립 검수된 atomic-fact 정답 집합은 아니며, 변형 질문 독립 의미 검수도 아직 완료되지 않았다. Calibration 대상으로 전체 30,000행을 실행하고 일반·조건·실제 NO_FAQ 15,900행에서 score/margin 임계값을 맞춘다. 같은 데이터에서 얻은 Calibration 수치이며 별도 Holdout 결과가 아니다. 반복은 안정성 그룹별로 집계하고 일반 성능 평균에 합치지 않는다.

중간 API 벡터·reranker 점수·native 점수는 이 새 폴더에 checkpoint로 저장한다. 다른 엔진의 inference 결과는 재사용하지 않는다. 각 엔진별 selection·policy·유형/목적별 결과와 최종 비교 보고서는 `outputs/run-v1`에 생성한다.
