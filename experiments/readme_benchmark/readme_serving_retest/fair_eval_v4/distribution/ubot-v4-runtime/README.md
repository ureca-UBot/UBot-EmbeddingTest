# UBot v4 실행 코드와 환경

Excel 원천 FAQ·KT·SKT·LG U+ 네 데이터셋을 Ollama와 vLLM에서 비교한 v4 실행 패키지입니다. 질문 생성기는 포함하지 않습니다. 별도로 관리하는 private/local 데이터 ZIP의 동결 JSONL을 읽어 실행합니다.

Source: SK Telecom / KT / LG U+ public FAQ pages, used for non-commercial research/evaluation

FAQ 원문·사용자 질문·정답은 공개 저장소나 이 코드 패키지에 포함되지 않습니다. `-DatasetZip`에는 접근 권한을 가진 비공개 데이터 경로를 지정해야 합니다. 데이터 ZIP을 공개 배포하거나 Git에 추가하지 마세요. 생성되는 상세 로그와 문서 창도 private/local로 관리합니다.

## 포함 범위

- v4 데이터 로더, 평가 함수, HTTP 임베딩·부하·반복·문서 공격 실행기
- A/B/C, BGE/Jina 리랭커, Sparse/ColBERT, Dense+Sparse RRF의 13개 구조
- 후보 로그 생성, 전수 감사, 비교 보고서, 단일 질문 로그 조회
- Docker Compose, Dockerfile 2개, 실제 환경에서 추출한 의존성 버전과 설치 잠금 파일
- 데이터 가져오기, 모델 준비, 전체 실행, 종료 스크립트

**제외:** 질문/FAQ 본문, 모델 가중치, Docker 이미지 바이너리, 임베딩 캐시, 측정 결과와 로그. `manifest.json`은 입력 개수·분할·해시를 담은 검증용 메타데이터입니다. 이 파일의 `serving_runs_performed: false`는 데이터셋 작성 시점 기록이며 새 실행 진행 상태가 아닙니다.

## 실행 환경

- Windows PowerShell 5.1 또는 PowerShell 7
- Docker Desktop의 Linux 컨테이너와 NVIDIA GPU 사용이 가능한 환경
- 실제 측정 장비: RTX 3060 12GB. Docker 이미지·모델·출력 저장 공간도 필요합니다.
- 첫 이미지 빌드와 모델 다운로드는 인터넷 연결이 필요합니다. 추론 worker는 다운로드 후 오프라인 캐시를 사용합니다.
- Python은 컨테이너에 포함되므로 호스트 Python 설치는 필요하지 않습니다.
- 포트: Ollama `127.0.0.1:11436`, vLLM `127.0.0.1:8002`. 두 엔진은 차례로 기동하고 종료합니다.

## 새 PC에서 실행

이 ZIP을 푼 `ubot-v4-runtime` 폴더에서 실행하세요. 비공개 데이터 ZIP은 `jsonl/<dataset>/faq_pairs.jsonl`과 `user_questions.jsonl`을 포함하고 동결 manifest의 해시와 일치해야 합니다.

```powershell
.\Prepare.ps1 -DatasetZip 'C:\경로\테스트질문셋_v4_4종.zip'
.\Run.ps1
```

준비 단계는 이미지 빌드 → 설치 패키지 확인 → 데이터·fixture 복원 → 고정 모델 다운로드 → Ollama 모델 해시 확인 → 실행 전 검사 순서입니다. 실행 단계는 데이터셋마다 Ollama·vLLM HTTP 측정과 종료 확인을 반복한 뒤, 공통 리랭커·native head·전수 감사를 수행합니다.

기본 모델 볼륨은 `ubot-v4-runtime-*`으로 분리됩니다. Compose 프로젝트 이름도 `ubot-v4-runtime`으로, 기존 테스트 프로젝트와 다릅니다. 단, 포트는 같으므로 기존 엔진이 꺼진 상태에서 실행하세요.

## 기존 측정 PC의 이미지·모델 재사용

`serving_test/.env.example`을 `serving_test/.env`로 복사하고 아래 값으로 바꿉니다.

```dotenv
HF_MODELS_VOLUME=ubot-retrieval-benchmark_hf-benchmark-cache
OLLAMA_MODELS_VOLUME=ubot-retrieval-benchmark_ollama-cache
HF_RUNTIME_VOLUME=ubot-readme-benchmark_hf-runtime
```

```powershell
.\Prepare.ps1 -DatasetZip 'C:\경로\테스트질문셋_v4_4종.zip' -ReuseLocalImages -SkipModelDownload
.\Run.ps1
```

재사용 스위치는 기존 이미지 ID가 실제 측정 당시 ID와 같은지 확인한 뒤 v4 전용 태그를 추가합니다. 기존 태그와 볼륨을 삭제하지 않습니다. 모델 다운로드를 생략하면 고정 HF 스냅샷, serving 가중치의 텐서 동일성, Ollama digest를 검증합니다.

## 실행 범위와 재개

```powershell
.\Run.ps1 -Stage Validate   # 데이터 및 길이 검사, 엔진 측정 없음
.\Run.ps1 -Stage Dense      # 네 데이터셋의 두 엔진 HTTP 측정
.\Run.ps1 -Stage Auxiliary  # Dense 완료 뒤 리랭커, native head, 전수 감사
.\Stop.ps1                 # 컨테이너 종료 및 포트 확인
```

`serving_test/outputs/run-v4` 아래에 진행 상태와 체크포인트를 저장합니다. 중단 뒤 같은 명령을 다시 실행하면 완료 입력을 재사용합니다. 같은 출력 폴더에서 모델·정답·검색 설정을 바꾸지 마세요. 다른 설정의 실험은 이 패키지를 새 폴더에 풀어 별도로 실행합니다. `Stop.ps1`은 모델 볼륨과 결과를 보존합니다.

## 입력과 모델 고정

| 데이터셋 | 사용자 질문 | FAQ |
|---|---:|---:|
| existing | 340 | 1,024 |
| kt | 769 | 1,099 |
| skt | 619 | 1,305 |
| lgu | 190 | 131 |

가져오기 도구는 ZIP의 알려진 JSONL 8개만 읽고 SHA-256을 확인합니다. 원래 v4의 반복·문서 공격 설정은 원본과 같은 규칙으로 재구성한 뒤 해시까지 대조합니다. 다른 데이터셋이나 수정된 정답을 조용히 받아들이지 않습니다.

HF 모델 3개의 revision, Ollama F16 GGUF 해시, 원래 이미지 ID는 `environment/runtime-lock.json`에 있습니다. Docker API 이미지도 digest로 고정했습니다. `docker/requirements-*.lock`은 고정 base 이미지에 실제로 추가·변경된 패키지 전부를 버전 고정한 설치 목록입니다. `environment/*.json`은 설치 후 대조하는 전체 패키지 기록입니다.

Ollama 다운로드는 `bge-m3:latest` 태그를 사용한 뒤 **측정 당시 digest 및 GGUF 해시와 일치하는지 검사**합니다. 배포자가 태그를 변경하면 준비 단계가 중단됩니다. 이 경우 원래 모델 캐시를 사용해야 하며 다른 모델을 같은 실험으로 처리하지 않습니다.

## 결과와 로그

출력 위치: `serving_test/outputs/run-v4/`

- `comparison.md`, `results.json`: 구조별·목적별·분할별 성적
- `load_results.json`: 동시성 1/4/8/16/32의 HTTP 임베딩 처리량과 지연
- `<dataset>/<engine>/<structure>/candidate_log.jsonl.gz`: 후보별 전체 로그
- `completion_audit.json`, `final_shutdown.json`: 로그 검사와 종료 확인

```powershell
Set-Location serving_test
docker compose run --rm --no-deps validate inspect_log.py --dataset kt --engine vllm --question Q3-S0001-01
```

후보 로그는 query, expected_faq_id/expected_faq_ids 전체 목록, retrieval_rank/cosine, retrieval_a_rank/cosine, retrieval_b_rank/cosine, rerank_rank/score, rank_delta, window_count, winning_window_index/text/score를 포함합니다. `rank_delta = retrieval_rank - rerank_rank`입니다.

## 측정 범위

Ollama/vLLM은 BGE-M3 dense 임베딩을 서빙합니다. BGE/Jina 리랭커와 sparse/ColBERT는 공통 HF/PyTorch·FlagEmbedding 보조 환경에서 실행합니다. 임계값 피팅과 생성 LLM 답변 평가는 포함하지 않습니다. 정답은 독립 의미 검수 대기 상태이므로 결과는 합성 진단입니다.

이 패키지는 실제 v4 실행 코드를 기반으로 경로·이미지·볼륨 설정과 준비 기능을 분리한 배포본입니다. 배포 검증 범위는 `PACKAGE_VALIDATION.json`에 기록됩니다. 새 PC에서의 네트워크 다운로드·전체 GPU 재측정까지 완료됐다는 의미는 아닙니다.
