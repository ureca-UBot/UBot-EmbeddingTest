# UBot Retrieval Benchmark Docker 실행 안내

A~F → 구조·threshold 동결 → holdout 1회 → 엔진별 켜기·확인·측정·끄기·확인 → CPU/GPU 별도 비교 순서로 실행한다. 실제 결과는 실행 보고서 (로컬 전용: `../outputs/retrieval/run-20261006-v1/실행결과_20261006.md`), 원천 모든 행의 처리 상태는 3,000행 ledger (로컬 전용: `../outputs/retrieval/run-20261006-v1/source-3000-ledger.jsonl`)에 기록한다.

실험은 원천 라벨의 잠정 평가다. NO_FAQ holdout과 기존 235문항 Top1 gate가 실패해 운영 반영하지 않는다. 생성 모델·외부 API는 호출하지 않았다. 독립 도메인 검수는 미완료다.

## 환경과 데이터

Docker Desktop 4.90.0 / Linux Engine 29.7.2 / RTX 3060 12,288 MiB / driver 595.95. 품질 image는 `ubot-retrieval-benchmark:run-v1`, Jina image는 `ubot-retrieval-jina:run-v1`이다. [모델 revision](../configs/resolved_model_revisions.json)을 사용한다.

실제 dataset: `model_benchmarks/faq/retrieval_run_v2/`, 출력: `model_benchmarks/outputs/retrieval/run-20261006-v1/`. 1,024문서와 FAQ 검색 1,997행이다. 두 엑셀의 3,000질문은 중복되므로 6,000건으로 합산하지 않는다.

## 데이터·지표 검증과 품질 실행

```powershell
# 저장소 루트에서 초기 개선 실험 폴더로 이동한다.
Set-Location ./experiments
docker compose --profile validation run --rm validate
docker compose --profile validation run --rm unit-tests
docker compose --profile diagnostics run --rm gpu-check
```

`validate`는 설계 예시 20건의 스키마·의미 계약과 4개 원천 SHA를 검사한다. `--allow-design-examples`는 설계용 문항 확인 옵션이다. 실제 잠정 dataset의 manifest·split·GT ID·hash도 별도 확인하며 승인 데이터로 바꾸지 않는다.

아래는 이미 실행한 단계의 재현 명령이다. 현 결과를 덮어쓰거나 확인한 holdout을 튜닝하지 않는다. 새 실험은 별도 run ID와 dataset 버전으로 scripts의 고정 경로를 먼저 변경한 뒤 calibration부터 시작한다.

```powershell
docker compose --profile quality run --rm benchmark model_benchmarks/scripts/run_retrieval_benchmark.py A
.\run_quality_sequence.ps1
.\run_native_sequence.ps1
.\run_frozen_holdout.ps1
docker compose --profile quality run --rm benchmark model_benchmarks/scripts/assess_quality_gates.py
```

B/C, 기존 두 회귀셋, BGE/Jina 질문 및 질문+답변 reranker를 차례로 실행했다. Native E/F는 잠정 탐색 진단이고 BM25는 검수된 lexical miss가 없어 보류했다. 선택·threshold는 `selection.json`으로 동결했다. `evaluate_frozen_holdout.py`는 정책 결과가 있으면 재평가를 거절한다.

Jina custom Python은 네트워크 차단·읽기 전용 root·capability 제거·model/data/scripts 읽기 전용·결과만 쓰기로 실행한다. Transformers 4.51.3 adapter를 사용하므로 BGE와의 속도 차이를 모델 효과만으로 해석하지 않는다.

## 하나씩 켜고 끄는 엔진

| 엔진 | image | 호스트 주소 | 모델 |
|---|---|---|---|
| Ollama | 0.35.0, digest 고정 | http://127.0.0.1:11435 | GGUF F16, artifact hash 별도 |
| TEI | cuda-1.9 digest 고정, 실제 1.9.4 | http://127.0.0.1:8081 | HF FP32, CLS, 최대 512 |
| vLLM | 0.30.0, digest 고정 | http://127.0.0.1:8001 | HF FP32, BgeM3EmbeddingModel, embed task |

TEI/vLLM safetensors는 원본 모든 tensor와 `torch.equal`로 확인했다. TEI 512 metadata artifact는 weights/tokenizer를 동일 파일로 연결한다. Ollama F16은 기준 FP32와 구분한다. 프로젝트별 모델 volume을 보존하고 `down -v`로 삭제하지 않는다.

```powershell
# Ollama: 준비된 bge-m3 artifact 사용
docker compose --profile ollama up -d ollama
.\wait_serving_ready.ps1 -Engine ollama -Url http://localhost:11435/api/tags
docker compose --profile quality run --rm benchmark model_benchmarks/scripts/benchmark_serving_client.py --engine ollama --url http://ollama:11434
docker compose --profile ollama logs --no-color ollama
docker compose stop ollama
.\verify_engine_stopped.ps1 -Engine ollama -Url http://localhost:11435/api/tags

# Ollama 종료 확인 후 TEI
docker compose --profile tei up -d tei
.\wait_serving_ready.ps1 -Engine tei -Url http://localhost:8081/health
docker compose --profile quality run --rm benchmark model_benchmarks/scripts/benchmark_serving_client.py --engine tei --url http://tei:80
docker compose --profile tei logs --no-color tei
docker compose stop tei
.\verify_engine_stopped.ps1 -Engine tei -Url http://localhost:8081/health

# TEI 종료 확인 후 vLLM
docker compose --profile vllm up -d vllm
.\wait_serving_ready.ps1 -Engine vllm -Url http://localhost:8001/health
docker compose --profile quality run --rm benchmark model_benchmarks/scripts/benchmark_serving_client.py --engine vllm --url http://vllm:8000
docker compose --profile vllm logs --no-color vllm
docker compose stop vllm
.\verify_engine_stopped.ps1 -Engine vllm -Url http://localhost:8001/health
```

별도 터미널에서 `.\monitor_serving_memory.ps1 -Engine tei`처럼 RAM sampling을 시작하면 엔진 종료 시 끝난다. 부하 client는 VRAM을 1초마다 기록한다. GPU 메모리는 Windows display 프로세스도 포함해 종료 후 0이 되지는 않는다. 엔진은 동시에 켜지 않는다.

embedding dimension·유한값·batch 순서·벡터 cosine·rank·사실 지표·P0 drift를 확인한 뒤 동시성 1/4/8/16/32 × batch 1/8/32를 측정한다. 코퍼스도 엔진마다 재임베딩한다. 범위는 dense 컴포넌트이며 reranker 포함 UBot 전체 서비스는 아니다.

실제 1차 부하는 각 10초 이상·최소 100 attempt·1회·워밍업 5회·timeout 30초·재시도 없음이다. 60초·1,000요청·3회는 설계 기본값이며 실행 완료로 표시하지 않는다. 성공 latency와 전체 attempt 실패율을 별도로 저장한다. 빠르게 거절된 요청 뒤의 새 요청은 같은 요청의 재시도가 아니다.

TEI 한도 512에서 동시성 32 × batch 32의 HTTP 429가 발생했다. 첫 sweep은 `serving/tei-default-capacity-512/`에 보존한다. 종료 확인 후 한도 2,048로 재시작해 15조건을 다시 측정했다. cache 보존과 입력 최대 길이 변경이 함께 적용되어 시작 시간 차이를 한 원인의 개선으로 발표하지 않는다.

## 엔진 후보를 좁힌 뒤 CPU/GPU 별도 측정

```powershell
docker compose --profile hardware run --rm hardware-cpu model_benchmarks/scripts/shortlist_serving.py
docker compose --profile quality run --rm benchmark model_benchmarks/scripts/run_selected_language_pairs.py
docker compose --profile hardware run --rm hardware-cpu
docker compose --profile quality run --rm benchmark model_benchmarks/scripts/benchmark_cpu_gpu.py --device cuda
docker compose --profile hardware run --rm hardware-cpu model_benchmarks/scripts/combine_hardware.py
docker compose --profile hardware run --rm hardware-cpu model_benchmarks/scripts/write_execution_report.py
docker compose ps --all
```

CPU는 GPU reservation이 없는 컨테이너다. CPU 종료 후 GPU 컨테이너를 실행하고 device마다 모델을 다시 로드한다. 동일 FP32 SentenceTransformer dense 컴포넌트 batch 1/8/32 각 30요청이다. 엔진 자체의 CPU/GPU end-to-end 비교는 이번 결과에 포함되지 않는다.

`shortlist.json`은 cosine 최소 0.9999와 15조건 무오류를 통과한 엔진 중 batch=1 throughput으로 1차 후보를 좁힌다. 운영 SLO와 장시간 반복 부하 검증이 없으므로 제품 도입 결론으로 쓰지 않는다.
