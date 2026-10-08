# 공개 자료와 로컬 데이터

Source: SK Telecom / KT / LG U+ public FAQ pages, used for non-commercial research/evaluation

이 저장소는 테스트 코드, 평가 방법, 실행 환경과 집계 결과를 공개한다. 통신사 FAQ 질문·답변 본문은 재배포하지 않는다.

## 공개 범위

- 임베딩·검색·리랭커·서빙 측정 스크립트, 평가 함수, Docker 설정과 패키지 버전.
- 평가셋의 크기, 분할, 검증 목적, 정답 채점 방식, 입력 해시.
- 본문을 포함하지 않는 구조별·목적별·분할별 집계 성적과 부하 수치.
- [FAQ ID와 source URL](public_results/faq_sources.json). 질문, 답변, 제목, 발췌문, 이미지 URL은 포함하지 않는다. 일부 FAQ의 출처는 개별 문항 주소가 아닌 목록 페이지다.

공개 결과에는 FAQ 인용문을 넣지 않는다. 설명에 필요한 예시는 특정 통신사 원문을 복제하지 않은 추상 예시를 사용한다.

## Private/local 범위

원본 FAQ, 사용자 질문·정답·근거 데이터, 평가셋 복사본, 검수 화면, Excel·데이터 ZIP, 후보 로그, 질의별 상세 결과, 벡터 캐시와 문항 작성 자료는 Git에서 제외한다. 실행 스크립트의 경로를 유지하기 위해 로컬 파일은 원래 경로에 보존한다. `.gitignore`는 기본 제외 후 공개 파일 종류와 검토한 결과 파일만 허용한다.

주요 로컬 경로:

- 루트 각 비교 폴더의 `faq/`, 질문·본문을 포함하는 `outputs/` 상세 파일.
- `experiments/**/data/`, `datasets/`, `source/`, `sources/`, `authoring/`, `review/`, `exports/`.
- `experiments/**/outputs/`: 공개용 결과는 원본 출력에서 별도 추출한 `public_results/`를 사용한다.
- `.private/`: 게시 전 파일 목록, 제거 대상의 로컬 보존 확인 및 변경 전 문서.

공개 저장소를 clone하는 것만으로 FAQ 데이터가 제공되지는 않는다. 재실행에는 접근 권한을 가진 로컬 FAQ·평가 데이터가 필요하다. 동결 v4 runner는 입력 해시를 검사하므로 다른 데이터로 실행하려면 별도 데이터셋 버전과 manifest를 준비해야 한다.

독립 실행 패키지의 `Prepare.ps1 -DatasetZip ...`에는 비공개로 관리하는 데이터 ZIP 경로를 전달한다. 코드·환경 ZIP에는 FAQ 본문이 없으며, 데이터 ZIP은 공개 배포 대상이 아니다.

## 공개 결과 생성과 검사

저장소 루트에서:

```powershell
python experiments/publication/export_public.py
python experiments/publication/check_public.py
```

첫 명령은 로컬 v4 결과에서 허용한 숫자 필드와 FAQ ID/source URL만 추출한다. 두 번째 명령은 Git 추적 파일 및 추가 가능한 파일에 비공개 경로나 원시 데이터가 포함되는지 검사한다. FAQ 원문이 로컬에 있으면 본문과 일치하는 긴 문자열도 검사한다. `--tracked`는 Git 인덱스에 올라간 내용만 검사하므로 commit 전에 사용한다.

`.gitignore`는 이미 추적하는 파일이나 과거 커밋을 제거하지 않는다. 현재 인덱스의 데이터 파일은 `git rm --cached`로 제외하고 로컬 파일을 유지한다. 과거 커밋·GitHub에 이미 게시된 파일의 제거는 별도의 이력 정리와 원격 반영이 필요하다. 이 작업에서는 이력 재작성이나 강제 푸시를 수행하지 않는다.
