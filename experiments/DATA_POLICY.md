# 후속 실험의 공개 자료와 로컬 데이터

Source: SK Telecom / KT / LG U+ public FAQ pages, used for non-commercial research/evaluation

이 정책은 `experiments/`의 후속 실험에만 적용한다. 후속 실험은 테스트 코드, 평가 방법, 실행 환경과 집계 결과를 공개하고, 통신사 FAQ 질문·답변 본문은 재배포하지 않는다.

루트의 `clustering/`, `intent/`, `model_benchmarks/`, `reranker/`는 모델·검색 방식 비교 자료다. 해당 폴더의 원본 파일 내용과 Git 추적 상태는 그대로 유지하며, 아래 제외 정책을 적용하지 않는다.

## 공개 범위

- 임베딩·검색·리랭커·서빙 측정 스크립트, 평가 함수, Docker 설정과 패키지 버전.
- 평가셋의 크기, 분할, 검증 목적, 정답 채점 방식, 입력 해시.
- 본문을 포함하지 않는 구조별·목적별·분할별 집계 성적과 부하 수치.
- [FAQ ID와 source URL](public_results/faq_sources.json). 질문, 답변, 제목, 발췌문, 이미지 URL은 포함하지 않는다. 일부 FAQ의 출처는 개별 문항 주소가 아닌 목록 페이지다.

공개 결과에는 FAQ 인용문을 넣지 않는다. 설명에 필요한 예시는 특정 통신사 원문을 복제하지 않은 추상 예시를 사용한다.

## Private/local 범위

후속 실험의 원본 FAQ, 사용자 질문·정답·근거 데이터, 평가셋 복사본, 검수 화면, Excel·데이터 ZIP, 후보 로그, 질의별 상세 결과, 벡터 캐시와 문항 작성 자료는 Git에서 제외한다. 실행 스크립트의 경로를 유지하기 위해 로컬 파일은 원래 경로에 보존한다. `experiments/.gitignore`는 해당 폴더 안에서 기본 제외 후 공개 파일 종류와 검토한 결과 파일만 허용한다.

주요 로컬 경로:

- `experiments/**/data/`, `datasets/`, `source/`, `sources/`, `authoring/`, `review/`, `exports/`.
- `experiments/**/outputs/`: 공개용 결과는 원본 출력에서 별도 추출한 `public_results/`를 사용한다.
- 루트 `.private/`: 작업 전 파일 목록, 로컬 보존 확인 및 변경 전 문서 백업.

공개 저장소를 clone하는 것만으로 후속 실험의 FAQ·평가 데이터가 제공되지는 않는다. 후속 실험 재실행에는 접근 권한을 가진 로컬 FAQ·평가 데이터가 필요하다. 동결 v4 runner는 입력 해시를 검사하므로 다른 데이터로 실행하려면 별도 데이터셋 버전과 manifest를 준비해야 한다.

독립 실행 패키지의 `Prepare.ps1 -DatasetZip ...`에는 비공개로 관리하는 데이터 ZIP 경로를 전달한다. 코드·환경 ZIP에는 FAQ 본문이 없으며, 데이터 ZIP은 공개 배포 대상이 아니다.

## 공개 결과 생성과 검사

저장소 루트에서:

```powershell
python experiments/publication/export_public.py
python experiments/publication/check_public.py
```

첫 명령은 로컬 v4 결과에서 허용한 숫자 필드와 FAQ ID/source URL만 추출한다. 두 번째 명령은 `experiments/`와 루트 안내 파일(`README.md`, `.gitignore`, `.gitattributes`) 중 Git 추적 파일 및 추가 가능한 파일을 검사한다. 모델·검색 방식 비교 폴더는 검사 대상에서 제외하며, 출력의 `out_of_scope_files`에 그 수를 표시한다. FAQ 원문이 로컬에 있으면 본문과 일치하는 긴 문자열도 검사한다. `--tracked`는 Git 인덱스에 올라간 내용만 검사하므로 commit 전에 사용한다.

`.gitignore`는 이미 추적하는 파일이나 과거 커밋을 제거하지 않는다. 모델·검색 방식 비교 폴더의 원본 데이터도 Git 추적을 유지한다. 후속 실험의 데이터 제외를 위해 원본 비교 자료를 제거하거나 Git 이력을 재작성하지 않는다.
