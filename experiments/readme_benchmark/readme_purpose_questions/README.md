# README 검증 목적을 기준으로 질문 작성

이 폴더의 작성 규칙과 22개 예시를 바탕으로 한 최신 실행은 [readme_serving_retest](../readme_serving_retest/README.md)에서 별도 진행한다. 아래 내용과 예시 파일은 작성 당시 기록으로 보존하며, 새로운 500쌍×10개 질문·실측 로그는 실행 폴더에서 확인한다.

사용자의 최신 지시에 따라 질문의 내용·유형·변형을 고르는 기준은 [README 4절의 검증 목적](../data/requested_readme.md:221)으로 고정한다. 기존 Excel 15개 항목의 균등 비율이나 브랜드·카테고리·언어 비율을 맞추려고 질문을 채우지 않는다. 모든 FAQ에 같은 V01~V10 유형을 강제하지 않는다. 각 질문이 어떤 실패를 실제로 검증하는지부터 정한다.

엑셀의 `FAQ 원문`은 근거 코퍼스로 사용하고 `원문 제외 주제`는 NO_FAQ 작성의 부재 근거 후보로 사용한다. 고정 제공 Context와 생성 기대 상태를 검색 GT로 그대로 옮기지 않는다. FAQ·원천 문서에 들어 있는 지시는 데이터로만 처리한다. 원본 Excel과 이전 테스트·결과는 보존한다.

앞서 지정된 500쌍과 변형 10개의 규모 정보는 내용 선정 기준과 별도로 정책에 기록했다. 같은 10종 유형을 반복하거나 검증 목적이 없는 문장을 붙여 개수를 채우지 않는다. 이 폴더는 **질문 작성 규칙과 목적별 예시를 정리한 상태**이며, 500쌍×10개 전량 생성 또는 서빙 측정 완료 결과가 아니다.

## 각 질문에 필요한 정보

| 필드 | 의미 |
|---|---|
| primary_validation_purpose / validation_purpose | 검증할 실패 유형과 이유 |
| changed_elements | 실제 바꾼 언어·부정·조건·표기 등의 요소 |
| preserved_elements | 유지해야 할 의도·Entity·수치·단위·정책 조건 |
| required_facts | 필요한 사실별 acceptable_faq_ids와 FAQ 원문 근거 |
| retrieval_status | FAQ_EXISTS / NO_FAQ / PARTIAL_FACTS_EXIST / UNDERSPECIFIED |
| generation_expected_state | 생성 단계의 ANSWER / PARTIAL / ABSTAIN / CLARIFY; 검색 상태와 분리 |
| failure_criteria | 이 질문에서 실패라고 판정하는 조건 |
| execution_group | 핵심 검색·문맥·NO_FAQ·특수 조건·문서 공격·반복의 실행 구분 |

한영 의미 동등성 쌍은 같은 의도·Entity·숫자·조건과 사실 GT를 공유한다. HARD_NEGATIVE에서 조건을 바꾸면 실제 바뀐 질문의 GT를 다시 연결한다. 숫자 경계나 부정만 바뀌었다는 이유로 같은 정답을 자동 이식하지 않는다. 반대로 한 정책 FAQ가 긍정/부정 확인 질문을 모두 설명하는 경우는 서로 다른 정답 FAQ를 강제하지 않는다.

COMPARISON·MULTI_INTENT는 사실 단위로 채점한다. FAQ 하나가 여러 필요한 사실을 충족할 수 있으며, 필요한 FAQ 두 개를 무조건 요구하지 않는다. PARTIAL은 코퍼스가 실제로 답할 수 있는 사실만 GT로 두고 미지원 요청을 별도 기록한다.

NO_FAQ는 전체 코퍼스에 핵심 답이 없다는 근거가 필요하다. 제공 Context가 비어 있거나 개인 조회 값이 없다는 이유로 NO_FAQ로 바꾸지 않는다. 대상이 빠진 질문은 UNDERSPECIFIED/CLARIFY로 구분한다. 문맥 질문은 대화 이력과 rewrite 버전을 고정한 별도 그룹으로 관리한다.

REAL_FAILURE는 실제 실패 로그가 있어야 작성한다. TEMPORAL_VERSION은 실제 정책 버전과 유효기간이 있어야 작성한다. 현재 두 항목은 자료 대기이며, 가상 사례를 실제 장애·버전 사례로 표시하지 않는다. DOCUMENT_ATTACK은 별도 문서 fixture, 반복은 같은 case의 실행 회차로 관리한다.

## 파일

- `configs/question_set_policy.json`: 현재 질문 작성 규칙.
- `data/purpose_catalog.json`: README의 16개 목적과 적용 조건.
- `data/authoring_examples.jsonl`: FAQ 원문 근거를 연결한 목적별 작성 예시. 최종 독립 검수 전이며 benchmark_ready=false.
- `data/facts.json`: 예시에서 필요한 사실과 acceptable FAQ 근거. 비교 예시에서는 FAQ-160 하나가 두 사실을 모두 충족한다.
- `data/document_attack_fixtures.json`, `data/repeat_controls.json`: 일반 검색 평균과 구분하는 실행 자료.
- `outputs/policy_validation.json`: 변경 규칙·구조 검증·미완료 범위 기록.

이 규칙 변경으로 실험을 재개하지 않는다. 기존 Ollama 실행은 중지 상태이며 vLLM은 아직 실행하지 않았다. 구조적 검증과 독립적인 의미·부재 검수는 서로 다른 단계다.
