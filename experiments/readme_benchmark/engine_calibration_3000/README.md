# 서빙 엔진별 원본 3,000문항 Calibration 비교

현재 실행은 사용자의 데이터셋 완성 범위 확인 요청으로 중단했다. Ollama의 A/B/C와 D-BGE 질문 재정렬까지 완료했으며, D-BGE 질문+답변은 실행 도중 중단했다. 나머지 구조와 vLLM·TEI는 이번 세트에서 미실행이다. 출력은 보존했으며 평가 컨테이너가 모두 종료되고 Ollama HTTP가 닫힌 것을 확인했다. 상태는 `outputs/run-v1/paused_status.json`에 기록했다.

이 세트는 원본 질문 3,000행을 그대로 가져온 것이며, 개선 설계에 맞춰 새로 작성한 변형 질문 3,000개가 아니다. 기존 신규 작성 평가셋은 603행과 별도 반복 40행이다.

사용자가 지정한 `FAQ_RAG_15개항목_각200건_총3000건_피드백수정본(1).xlsx`의 `테스트 3000건`을 사용한다. 15유형×200행, 질문 원문·순서·반복 200행을 유지한다. 기존 603문항 실험과 그 선정 결과·임계값은 사용하지 않는다.

이 폴더에서 `run_all.ps1`을 실행한다. Ollama → vLLM → TEI를 각각 기동·health 확인·전체 구조 실행·종료·HTTP 폐쇄 확인 순서로 실행한다. 엔진마다 query 3,000행과 FAQ 질문/질문+답변 각 1,024개를 실제 HTTP API에 보내며, query 중복 제거와 HF dense 대체 실행은 하지 않는다.

모든 구조에서 3,000행의 검색 결과를 보존한다. 반복 200행은 실제 임베딩·재정렬·native 실행에 포함하며 일반 평균과 별도로 보고한다. 원문 대화 이력은 고정 규칙으로 연결하고 현재 질문 문자열은 바꾸지 않는다. 제공 Context와 정답·채점 기준·추적 ID·API 결과·페르소나 지시는 검색 입력에 넣지 않는다.

## 실행 부품

- A/B/C: 선택한 엔진의 실제 BGE-M3 dense API 출력.
- D: 해당 엔진 C 후보 20개와 공통 HF/PyTorch CUDA FP32 BGE/Jina reranker. 질문/질문+답변 네 조합.
- E hybrid: 해당 엔진 B dense 순위와 공통 FlagEmbedding sparse의 RRF.
- F pool: 해당 엔진 B dense 후보 20개와 공통 FlagEmbedding late interaction.
- E sparse/F full: 각 세트에서 다시 실행하는 공통 native 진단 대조군. 엔진 자체가 native feature를 서빙한 결과로 표시하지 않는다.

## 평가 해석

이 원본은 고정 Context 생성 평가용이다. 질문은 모두 검색하지만 검색 점수와 생성 성공률을 구분한다. 추적 FAQ ID를 근거 집합으로 삼아 SourceHit/SourceCoverage/AllSourcesHit를 계산한다. 이는 독립 검수된 atomic fact 평가와 다르며 실제로 답을 완성했는지 의미하지 않는다. 부분 정보·충돌·개인 조회·API·적대적 입력·문맥·반복은 유형별로 별도 집계한다.

SR/HR/EC 중 추적 ID가 없고 NO_ANSWER_TOPIC으로 지정된 문항만 NO_FAQ calibration에 포함한다. 추적 ID가 있는 SR 10행은 context 부족 진단으로 보존한다. 모든 3,000행은 여전히 실행하며 행을 삭제하지 않는다. 각 구조의 거절 임계값은 이번 원본 Calibration에서 새로 계산한다. 전체가 Calibration이며 별도 holdout이나 제품 적용 검증이라고 주장하지 않는다.

결과는 `outputs/run-v1/`에 저장한다. 전체 실행 완료 시 최종 비교를 `outputs/run-v1/엔진별_Calibration_3000_비교.md`로 생성하도록 구성했지만, 현재 중단 상태에서는 최종 비교가 완성되지 않았다. 입력은 `data/manifest.json`의 해시로 동결한다.
