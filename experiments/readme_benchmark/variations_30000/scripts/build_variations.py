"""Author ten traceable, meaning-preserving variations for every selected source row.

Workbook instructions, attack strings and API fixtures are data, never instructions
to this program. No serving output or previous experiment threshold is imported.
"""
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'engine_calibration_3000' / 'data'
OUT = ROOT / 'data'

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')

# Alternatives have the same referent. We never substitute another plan, amount,
# API status, deadline or entitlement and then inherit the original answer.
LEXICON = [
    ('휴대전화', ['휴대폰', '핸드폰', '이동전화'], 'mobile phone', True),
    ('휴대폰', ['휴대전화', '핸드폰', '이동전화'], 'mobile phone', True),
    ('본인인증', ['본인 확인', '본인 인증', '신원 인증'], 'identity verification', True),
    ('번호이동', ['번호 이동', '번호이동 절차', '통신사 변경을 위한 번호이동'], 'number portability', False),
    ('가족 결합', ['가족결합', '가족 회선 결합', '가족 결합 서비스'], 'family bundle', True),
    ('부가서비스', ['부가 서비스', '추가 서비스', '부가서비스 항목'], 'add-on service', True),
    ('고객센터', ['고객 센터', '고객 지원 센터', '상담 센터'], 'customer service center', False),
    ('요금제', ['요금 상품', '요금 플랜', '이용 요금제'], 'rate plan', True),
    ('통신사', ['통신 사업자', '통신 업체', '이동통신사'], 'carrier', False),
    ('신분증', ['신분 증명 서류', '신분증 서류', '본인 신분증'], 'ID card', False),
    ('유심', ['USIM', 'SIM 카드', '심 카드'], 'SIM card', False),
    ('eSIM', ['이심', 'eSIM 카드', '내장형 SIM'], 'embedded SIM', True),
    ('테더링', ['테더링 기능', '테더링 서비스', '테더링 이용'], 'tethering', True),
    ('쉐어링', ['데이터 쉐어링', '쉐어링 서비스', '데이터 공유 서비스'], 'data sharing', True),
    ('로밍', ['해외 로밍', '로밍 서비스', '로밍 이용'], 'roaming', True),
    ('데이터', ['데이터 용량', '데이터 제공량', '데이터량'], 'data', False),
    ('청구서', ['요금 청구서', '청구 내역서', '이용요금 청구서'], 'bill', True),
    ('청구액', ['청구 금액', '청구된 금액', '청구 요금'], 'billed amount', True),
    ('위약금', ['약정 위약금', '위약금 금액', '위약금 비용'], 'termination penalty', False),
    ('영업 상태', ['영업 여부', '영업 중인지 여부', '매장 영업 상태'], 'business status', True),
    ('매장', ['대리점 매장', '통신 매장', '매장 지점'], 'store', False),
    ('공기계', ['공기기', '미개통 단말기', '회선 없는 공기계'], 'unactivated handset', True),
    ('기기', ['단말기', '단말', '휴대 기기'], 'device', True),
    ('단말', ['단말기', '휴대 단말', '이동통신 단말'], 'handset', True),
    ('배터리', ['배터리 부품', '휴대폰 배터리', '기기 배터리'], 'battery', False),
    ('충전기', ['충전 장치', '휴대폰 충전기', '전원 충전기'], 'charger', False),
    ('케이스', ['보호 케이스', '휴대폰 케이스', '기기 케이스'], 'case', False),
    ('포인트', ['적립 포인트', '포인트 잔액', '서비스 포인트'], 'points', True),
    ('인감증명서', ['인감 증명서', '인감증명 서류', '명의자 인감증명서'], 'seal certificate', True),
    ('가족관계증명서', ['가족관계 증명서', '가족 관계 증명서', '가족관계증명 서류'], 'family relationship certificate', True),
    ('법정대리인', ['법정 대리인', '법정대리인 당사자', '법정대리인 본인'], 'legal guardian', True),
    ('명의자', ['회선 명의자', '가입 명의자', '명의자 본인'], 'account holder', False),
    ('미성년자', ['미성년 가입자', '미성년 이용자', '미성년자 본인'], 'minor', False),
    ('할인율', ['할인 비율', '적용 할인율', '요금 할인율'], 'discount rate', True),
    ('할인', ['요금 할인', '할인 혜택', '할인 적용'], 'discount', True),
    ('미납', ['미납 상태', '요금 미납', '미납 내역'], 'unpaid balance', True),
    ('납부', ['요금 납부', '요금 지불', '납부 처리'], 'payment', True),
    ('재발급', ['다시 발급', '재발급 처리', '재발급 절차'], 'reissue', False),
    ('소액결제', ['휴대폰 소액결제', '소액 결제', '모바일 소액결제'], 'carrier billing', True),
    ('가입', ['가입 신청', '가입 절차', '서비스 가입'], 'sign-up', True),
    ('해지', ['서비스 해지', '해지 처리', '이용 해지'], 'cancellation', True),
    ('개통', ['회선 개통', '개통 처리', '개통 절차'], 'activation', True),
    ('수리', ['기기 수리', '수리 서비스', '수리 처리'], 'repair', False),
    ('분실', ['분실 상태', '기기 분실', '분실 상황'], 'loss', True),
    ('보험', ['보험 서비스', '보험 상품', '보험 보장'], 'insurance', True),
    ('보상금', ['보상 금액', '보상금 금액', '지급 보상금'], 'compensation', True),
    ('수수료', ['처리 수수료', '수수료 금액', '부과 수수료'], 'fee', False),
    ('쿠폰', ['쿠폰 혜택', '제공 쿠폰', '서비스 쿠폰'], 'coupon', True),
    ('기준 시각', ['적용 기준 시각', '기준 시간', '정책 적용 시각'], 'reference time', True),
    ('변경 요청', ['변경 신청', '변경 접수 요청', '변경 요청 건'], 'change request', True),
    ('최신성', ['정보 최신성', '자료 최신성', '결과 최신성'], 'freshness', True),
    ('주소', ['매장 주소', '소재지 주소', '주소 정보'], 'address', True),
    ('거리', ['이동 거리', '거리 정보', '현재 위치와의 거리'], 'distance', True),
    ('인터넷', ['인터넷 서비스', '인터넷 회선', '인터넷 이용'], 'internet', True),
    ('문자', ['문자 메시지', '문자 서비스', '문자메시지'], 'text message', True),
    ('전화', ['통화', '전화 통화', '음성 통화'], 'phone call', True),
    ('메시지', ['메세지', '메시지 내용', '메시지 서비스'], 'message', True),
    ('앱', ['애플리케이션', '어플', '모바일 앱'], 'app', True),
    ('레시피', ['조리법', '요리 방법', '요리법'], 'recipe', False),
    ('홍보문', ['홍보 글', '홍보용 글', '홍보 문구'], 'promotional text', True),
    ('점심', ['점심 식사', '점심 메뉴', '낮 식사'], 'lunch', True),
    ('저녁', ['저녁 식사', '저녁 메뉴', '저녁 시간'], 'dinner', False),
    ('정치', ['정치 관련', '정치 분야', '정치적'], 'politics', True),
    ('영업', ['매장 영업', '영업 활동', '영업 운영'], 'business operation', True),
    ('한도', ['이용 한도', '허용 한도', '한도 기준'], 'limit', True),
    ('기간', ['적용 기간', '이용 기간', '기간 기준'], 'period', True),
    ('비용', ['이용 비용', '비용 금액', '발생 비용'], 'cost', True),
    ('규칙', ['기존 규칙', '안내 규칙', '정해진 규칙'], 'rules', True),
    ('가족', [], 'family', False), ('결합', [], 'bundling', True),
    ('구독팩', [], 'subscription pack', True), ('등급', [], 'membership tier', False),
    ('통신', [], 'telecommunications', True), ('회선', [], 'line', True),
    ('청구 금액', [], 'billed amount', True), ('요금', [], 'charge', True),
    ('개인정보', [], 'personal information', True), ('비밀번호', [], 'password', False),
    ('수학 숙제', [], 'math homework', True), ('영화 줄거리', [], 'movie plot', True),
    ('법률 문서', [], 'legal document', True), ('증상', [], 'symptoms', True),
    ('주식', [], 'stocks', True), ('수익', [], 'investment return', True),
    ('운세', [], 'fortune', True), ('인증', [], 'authentication', True),
    ('와이파이', [], 'Wi-Fi', False), ('할인반환금', [], 'discount repayment', True),
    ('약정', [], 'contract term', True), ('할부금', [], 'installment balance', True),
    ('할부', [], 'installment payment', True), ('통화', [], 'phone call', True),
    ('착신 전환', [], 'call forwarding', True), ('음성사서함', [], 'voicemail', True),
    ('바코드', [], 'barcode', False), ('제휴처', [], 'partner merchant', True),
    ('연락', [], 'contact', True), ('혜택', [], 'benefit', True),
    ('색상', [], 'color', False), ('배송', [], 'delivery', False),
    ('위치', [], 'location', True), ('서류', [], 'documents', True),
    ('신청', [], 'application', True), ('처리', [], 'processing', True),
    ('질문', [], 'question', True), ('검색', [], 'search', False),
    ('결제', [], 'payment', True), ('조건', [], 'conditions', True),
    ('결과', [], 'result', True), ('나이', [], 'age', False),
    ('코드', [], 'code', False), ('정보', [], 'information', True),
    ('과오납', [], 'overpayment', True), ('계좌', [], 'bank account', True),
    ('이용 내역', [], 'usage history', False), ('방법', [], 'method', False),
    ('시간', [], 'time', True), ('통신망', [], 'network', True),
]

# Korean lexical alternatives that add qualifications (e.g. a particular contract
# or device) are intentionally not used: each rule below is narrower and vetted.
SAFE_KO = {
    '휴대전화': ['휴대폰', '핸드폰', '이동전화'],
    '휴대폰': ['휴대전화', '핸드폰', '이동전화'],
    '본인인증': ['본인 인증', '본인 확인', '신원 확인'],
    '번호이동': ['번호 이동', '번호이동', '번호 이동'],
    '가족 결합': ['가족결합', '가족 회선 결합', '가족결합'],
    '부가서비스': ['부가 서비스', '추가 서비스', '부가 서비스'],
    '고객센터': ['고객 센터', '고객 지원 센터', '상담 센터'],
    '요금제': ['요금 플랜', '요금 상품', '요금 플랜'],
    '통신사': ['통신 사업자', '이동통신사', '통신 사업자'],
    '유심': ['USIM', 'SIM 카드', '심 카드'],
    'eSIM': ['이심', '내장형 SIM', '이심'],
    '청구액': ['청구 금액', '청구된 금액', '청구 금액'],
    '청구서': ['요금 청구서', '청구 내역서', '요금 청구서'],
    '인감증명서': ['인감 증명서', '인감증명 서류', '인감 증명서'],
    '가족관계증명서': ['가족관계 증명서', '가족 관계 증명서', '가족관계 증명서'],
    '법정대리인': ['법정 대리인', '법정대리인', '법정 대리인'],
    '할인율': ['할인 비율', '할인율', '할인 비율'],
    '소액결제': ['소액 결제', '소액결제', '소액 결제'],
    '보상금': ['보상 금액', '보상금', '보상 금액'],
    '레시피': ['조리법', '요리법', '조리법'],
    '홍보문': ['홍보 글', '홍보 문구', '홍보 글'],
    '현재 위치': ['지금 위치', '현 위치', '지금 있는 위치'],
    '지금 쓰는': ['현재 사용하는', '지금 이용 중인', '현재 쓰고 있는'],
    '이번 달': ['이달', '이번달', '이 달'],
    '지난달': ['직전 달', '지난 달', '바로 전 달'],
    '얼마나': ['어느 정도', '얼마나', '어느 정도'],
    '언제까지': ['어느 시점까지', '언제까지', '어느 때까지'],
    '어디로': ['어느 곳으로', '어디로', '어느 곳으로'],
    '어디서': ['어느 곳에서', '어디서', '어느 곳에서'],
    '나이 기준': ['연령 기준', '나이 요건', '연령 요건'],
    '가장 가까운': ['최단 거리에 있는', '제일 가까운', '가장 근처에 있는'],
    '최단 거리': ['가장 짧은 거리', '최단거리', '가장 짧은 거리'],
    '두 안내': ['두 안내문', '양쪽 안내', '두 안내문'],
    '적용 범위': ['적용되는 범위', '적용 대상 범위', '적용되는 범위'],
    '발급 후': ['발급한 뒤', '발급 이후', '발급한 후'],
    '연체': ['납부 지연', '연체', '납부 지연'],
    '아끼려고': ['절약하려고', '아끼려는 목적으로', '절약하기 위해'],
    '쓰고': ['사용하고', '이용하고', '사용하고'],
    '쓰는': ['사용하는', '이용하는', '사용하는'],
    '쓰면': ['사용하면', '이용하면', '사용할 경우'],
    '쓸 수': ['사용할 수', '이용할 수', '사용할 수'],
    '쓰나요': ['사용하나요', '이용하나요', '사용하나요'],
    '쓰지': ['사용하지', '이용하지', '사용하지'],
    '바꿀': ['변경할', '바꿀', '변경할'],
    '바꾸면': ['변경하면', '바꿀 경우', '변경할 경우'],
    '바꾸고': ['변경하고', '바꾸고', '변경하고'],
    '구입': ['구매', '구입', '구매'],
    '살 수': ['구매할 수', '살 수', '구입할 수'],
    '늘리면': ['추가하면', '늘릴 경우', '추가할 경우'],
    '느려지': ['속도가 느려지', '느려지', '속도가 저하되'],
    '어디에서': ['어느 곳에서', '어디서', '어느 곳에서'],
    '봐야': ['확인해야', '봐야', '확인해야'],
    '뭐': ['무엇', '뭐', '무엇'],
    '갱신': ['갱신', '업데이트', '갱신'],
    '채워지': ['충전되', '채워지', '채워지'],
    '급한 일': ['긴급한 일', '급한 상황', '긴급한 상황'],
    '문의드려요': ['문의드립니다', '질문드려요', '여쭤봅니다'],
    '물어보나요': ['문의하나요', '물어보나요', '질문하나요'],
    '챗봇': ['챗봇', '대화형 챗봇', '챗봇'],
    '가입하지 않은': ['가입한 적 없는', '가입하지 않은', '가입한 적이 없는'],
    '몇 월': ['어느 달', '몇 월', '어느 달'],
    '몇 명': ['몇 명', '몇 사람', '몇 명'],
    '며칠': ['몇 일', '며칠', '몇 일'],
    '몇 회': ['몇 번', '몇 회', '몇 번'],
    '언제부터': ['어느 시점부터', '언제부터', '어느 때부터'],
    '아까 말씀드린 내용인데': ['앞서 문의했던 내용인데', '전에 말씀드린 내용인데', '앞서 말씀드린 내용인데'],
    '그곳': ['그 장소', '해당 장소', '그 장소'],
    '원하는 색상': ['희망하는 색상', '찾는 색상', '원하는 색깔'],
    '사야': ['구매해야', '구입해야', '구매해야'],
    '안 읽혀요': ['인식되지 않아요', '잘 읽히지 않아요', '인식이 안 돼요'],
}

ENDING_RULES = [
    ('없을까요?', ['없는지 알려 주세요.', '없는지 알고 싶습니다.', '없는 건가요?', '없는지 확인 부탁드립니다.']),
    ('있을까요?', ['있는지 알려 주세요.', '있는지 알고 싶습니다.', '있는 건가요?', '있는지 확인 부탁드립니다.']),
    ('걱정돼요.', ['걱정됩니다.', '걱정되는 상황입니다.', '걱정돼서 문의드립니다.', '걱정하고 있어요.']),
    ('올랐어요.', ['올랐습니다.', '높아졌어요.', '상승했습니다.', '높아진 상황입니다.']),
    ('닫았어요.', ['닫았습니다.', '닫은 상황입니다.', '닫았는데요.', '닫아서 문의드립니다.']),
    ('끊겨요.', ['끊깁니다.', '끊기는 상황입니다.', '끊기는데요.', '끊겨서 문의드립니다.']),
    ('않았어요.', ['않았습니다.', '않은 상황입니다.', '않았는데요.', '않아서 문의드립니다.']),
    ('쌓였어요.', ['쌓였습니다.', '적립됐어요.', '쌓인 상황입니다.', '적립됐습니다.']),
    ('있어요.', ['있습니다.', '있는 상태입니다.', '있는 상황입니다.', '있는데요.']),
    ('돼요.', ['됩니다.', '되는 상황입니다.', '되는데요.', '되는 상태입니다.']),
    ('해요.', ['합니다.', '하는 상황입니다.', '하는데요.', '해서 문의드립니다.']),
    ('가능한가요?', ['가능한지 알려 주세요.', '가능한지 알고 싶습니다.', '가능한지 확인 부탁드립니다.', '가능한 건가요?']),
    ('좋나요?', ['좋은지 알려 주세요.', '좋은지 알고 싶습니다.', '좋은지 확인 부탁드립니다.', '좋은 건가요?']),
    ('어떻게 계산되나요?', ['계산 방식이 어떻게 되나요?', '어떤 기준으로 계산하나요?', '어떻게 계산되는지 알고 싶습니다.', '계산 방법을 설명해 주시겠어요?']),
    ('어떻게 하나요?', ['어떤 방법으로 하나요?', '어떻게 하면 되는지 알려 주세요.', '어떤 절차로 진행하나요?', '어떻게 하는지 설명해 주시겠어요?']),
    ('어떻게 되나요?', ['어떻게 되는지 알고 싶습니다.', '어떤 방식으로 처리되나요?', '어떻게 되는지 설명해 주세요.', '어떻게 되는 건가요?']),
    ('할 수 있나요?', ['할 수 있는지 알려 주세요.', '하는 것이 가능한가요?', '할 수 있는 건가요?', '할 수 있는지 확인하고 싶습니다.']),
    ('수 있나요?', ['수 있는지 알고 싶습니다.', '수 있는지 알려 주시겠어요?', '수 있는 건가요?', '수 있는지 확인 부탁드립니다.']),
    ('얼마인가요?', ['얼마인지 알려 주세요.', '얼마인지 알고 싶어요.', '얼마인지 확인 부탁드립니다.', '얼마인지 설명 부탁드립니다.']),
    ('걸리나요?', ['걸리는지 알려 주세요.', '걸리는지 알고 싶습니다.', '걸리는 건가요?', '걸리는지 확인 부탁드립니다.']),
    ('몇 개월인가요?', ['몇 개월인지 알고 싶습니다.', '몇 개월인지 알려 주세요.', '몇 개월인지 확인해 주세요.', '몇 개월인지 설명 부탁드립니다.']),
    ('인가요?', ['인지 알려 주세요.', '인지 알고 싶어요.', '인지 확인 부탁드립니다.', '인지 설명해 주시겠어요?']),
    ('있나요?', ['있는지 알고 싶습니다.', '있는지 알려 주세요.', '있는 건가요?', '있는지 확인해 주시겠어요?']),
    ('없나요?', ['없는지 알고 싶습니다.', '없는지 알려 주세요.', '없는 건가요?', '없는지 확인 부탁드립니다.']),
    ('되나요?', ['되는지 알고 싶습니다.', '되는지 알려 주세요.', '되는 건가요?', '되는지 확인 부탁드립니다.']),
    ('하나요?', ['하는지 알고 싶습니다.', '하는지 알려 주세요.', '하는 건가요?', '하는지 확인 부탁드립니다.']),
    ('끊기나요?', ['끊기는지 알고 싶습니다.', '끊기는지 알려 주세요.', '끊기는 건가요?', '끊기는지 확인 부탁드립니다.']),
    ('나요?', ['는지 알고 싶습니다.', '는지 알려 주세요.', '는 건가요?', '는지 확인 부탁드립니다.']),
    ('가요?', ['지 알고 싶습니다.', '지 알려 주세요.', '지 확인 부탁드립니다.', '지 설명해 주시겠어요?']),
    ('좋을까요?', ['좋을지 알고 싶습니다.', '좋을지 알려 주세요.', '좋을지 확인 부탁드립니다.', '좋을지 설명해 주시겠어요?']),
    ('어디죠?', ['어디인지 알려 주세요.', '어디인지 알고 싶습니다.', '어디인지 확인 부탁드립니다.', '어디인가요?']),
    ('잠겼어요.', ['잠긴 상태입니다.', '잠겼습니다.', '잠겼는데요.', '잠긴 상황입니다.']),
    ('잃어버렸어요.', ['분실했어요.', '잃어버렸습니다.', '분실했습니다.', '잃어버린 상황입니다.']),
    ('없어요.', ['없습니다.', '없는 상황입니다.', '없는데요.', '없는 상태입니다.']),
    ('나와요.', ['나옵니다.', '나오는 상황입니다.', '표시됩니다.', '나오고 있습니다.']),
    ('않아요.', ['않습니다.', '않는 상황입니다.', '않는데요.', '않는 상태입니다.']),
    ('왔어요.', ['왔습니다.', '도착했어요.', '왔는데요.', '도착한 상황입니다.']),
    ('됐어요.', ['됐습니다.', '된 상태입니다.', '됐는데요.', '된 상황입니다.']),
    ('없다고 나와요.', ['없다고 표시됩니다.', '없다는 안내가 나옵니다.', '없다고 표시되는 상황입니다.', '없다는 안내를 받았습니다.']),
    ('소진됩니다.', ['소진돼요.', '소진되는 상황입니다.', '소진되는데요.', '소진되고 있습니다.']),
    ('확정해 주세요.', ['확정 부탁드립니다.', '확정해 주시겠어요?', '확정해 주실 수 있나요?', '확정해서 알려 주세요.']),
    ('남았나요?', ['남았는지 알려 주세요.', '남았는지 알고 싶습니다.', '남았는지 확인 부탁드립니다.', '남아 있는 건가요?']),
    ('궁금해요.', ['궁금합니다.', '알고 싶습니다.', '확인하고 싶어요.', '알려 주시겠어요?']),
    ('궁금해요?', ['궁금합니다.', '알고 싶어요.', '확인하고 싶습니다.', '알려 주시겠어요?']),
    ('알려 주세요.', ['설명해 주시겠어요?', '알려 주실 수 있나요?', '안내 부탁드립니다.', '알려 주시면 좋겠습니다.']),
    ('설명해 주세요.', ['설명 부탁드립니다.', '설명해 주시겠어요?', '안내해 주세요.', '설명해 주실 수 있나요?']),
    ('확인해 주세요.', ['확인 부탁드립니다.', '확인해 주시겠어요?', '확인해 주실 수 있나요?', '확인해서 알려 주세요.']),
    ('답해 주세요.', ['답변해 주시겠어요?', '답변 부탁드립니다.', '답변해 주세요.', '답해 주실 수 있나요?']),
    ('말해 주세요.', ['말해 주시겠어요?', '알려 주세요.', '말해 주실 수 있나요?', '말씀해 주세요.']),
    ('싶어요.', ['싶습니다.', '싶은데요.', '싶거든요.', '싶어서 문의드립니다.']),
    ('입니다.', ['인 상황입니다.', '이라고 말씀드릴게요.', '이에요.', '입니다.']),
    ('예요.', ['입니다.', '인 상황이에요.', '이라고 나와요.', '입니다.']),
    ('해주세요.', ['해 주세요.', '부탁드려요.', '해 주시겠어요?', '해 주실 수 있나요?']),
    ('주세요.', ['주실 수 있나요?', '주시겠어요?', '주시면 좋겠습니다.', '주시기를 부탁드립니다.']),
]

TYPO_RULES = [('알려', '알랴'), ('궁금', '궁굼'), ('어떻게', '어떡게'),
              ('설명', '설멍'), ('확인', '화긴'), ('가능', '가눙'),
              ('원하', '원햐'), ('싶어요', '십어요'), ('주세요', '주세여'),
              ('있나요', '잇나요'), ('되나요', '돼나요'), ('하나요', '하나여')]

NOISE = [
    '오늘은 책상 주변을 정리하면서 오래된 메모와 봉투를 분류하고 있었어요. '
    '창문을 열었더니 바깥 소리가 들려 잠깐 하던 일을 멈췄고, 물을 마신 뒤 다시 자리에 앉았습니다. '
    '서랍 안에는 예전에 적어 둔 장보기 목록과 읽다 만 책의 메모가 섞여 있었어요. '
    '필요한 종이는 따로 모으고 쓰지 않는 종이는 정리했는데, 책꽂이에 책을 꽂는 순서가 마음에 들지 않아 다시 옮겼습니다. '
    '이런 일상적인 이야기는 제가 문의하려는 통신 서비스의 적용 조건과는 관계가 없습니다. '
    '막상 상담할 내용을 적으려니 문장을 짧게 쓰는 게 어려워서 배경을 길게 설명하게 됐어요. '
    '책상 위에는 물컵과 필기구가 놓여 있고, 옆에는 아직 펼치지 않은 잡지가 있습니다. '
    '잠시 창밖을 보다가 머릿속에서 오늘 할 일을 정리한 다음 문의 내용을 확인하려고 합니다. '
    '집안 정리 이야기는 참고하실 필요가 없고, 제가 적은 질문의 대상과 조건에 맞는 안내가 필요합니다. ',
    '상담 내용을 적기 전에 일상 이야기가 길어졌습니다. '
    '퇴근하고 돌아와 가방을 내려놓고 옷장을 정리했어요. 옷걸이가 서로 엉켜 있어 풀어 놓았고, '
    '선반에 놓인 작은 상자들을 다른 자리로 옮겼습니다. 상자에는 오래된 사진과 편지, 여행 중에 모은 종이가 들어 있었어요. '
    '사진을 살펴보다가 시간이 흘렀지만 이것들은 통신 서비스나 가입 조건을 바꾸는 정보가 아닙니다. '
    '휴식을 취하며 차를 마셨고, 내일 챙겨 갈 물건을 생각하면서 메모지를 찾아 펼쳤습니다. '
    '메모지의 빈 곳에 적을 말을 고민하다가 문장이 길어졌어요. '
    '방 안이 조용해서 창밖의 소리가 더 잘 들렸고, 자리에서 일어나 커튼을 정리했습니다. '
    '책상으로 돌아와 필기구를 고르고 종이를 가지런하게 맞추었습니다. '
    '앞에 적은 개인적인 일상은 요청 대상이나 정책 조건에 관한 추가 요구가 아닙니다. '
    '실제로 상담받고 싶은 내용은 이어서 적은 질문에 담았습니다. ',
]

PURPOSES = {
    'REAL_FAILURE': ('실제 서비스 실패 재현', '실패 입력과 발생 당시 로그·코퍼스', '로그 없음; 생성하지 않음'),
    'EXACT_ENTITY': ('명칭·약칭·띄어쓰기로 다른 상품 FAQ를 선택하는지', '같은 상품·서비스·정책', '근거 있는 명칭만 사용'),
    'HARD_NEGATIVE': ('가까운 주제의 경쟁 FAQ와 정답 근거를 구분하는지', '검수 대상 정답 근거와 원본 경쟁 Context', 'NC 원본의 경쟁 문서 환경 유지'),
    'KR_EN_EQUIVALENT': ('한영 동등 표현에서 같은 사실 근거를 찾는지', '의도·수치·조건·대응어의 의미', '단어 치환 쌍; 전체 문장 번역 아님'),
    'NEGATION': ('부정을 놓쳐 반대 의미 FAQ를 선택하는지', '부정의 대상과 조건', '극성을 뒤집고 원래 정답을 복사하지 않음'),
    'CONDITION_SCOPE': ('가입자·회선·API 필터·예외 조건을 일반화하는지', '조건 주체와 적용 범위', '원본 조건 보존'),
    'NUMERIC_CONDITION': ('경계·단위·기간·금액을 구분하는지', '수치와 비교 연산의 의미', '수치 변경 없음; 표기만 변형'),
    'COMPARISON': ('비교 대상의 근거를 모두 찾는지', '대상·비교 기준', '비교 순서와 표현 변형'),
    'MULTI_INTENT': ('여러 요구 중 일부 근거를 놓치는지', '각 요구와 필요한 사실', 'MC 질문 순서 변형'),
    'AMBIGUOUS_QUERY': ('생략·지시어에 대해 근거 없이 확정하는지', '원본 대화의 정보량과 미해결 대상', '이력 보존; 자동 명확화 정답 추가 없음'),
    'TEMPORAL_VERSION': ('기준 시각과 가상 정책 시행일을 혼동하는지', '원본 시각·유효기간·시험용 A/B 정책', 'CF 제공 Context의 가상 버전만 사용'),
    'PARAPHRASE': ('표현·문장 형태가 달라도 같은 근거를 찾는지', '의도와 필요한 사실', '명시적 어휘·종결어미·문장 구조 변형'),
    'LONG_QUERY': ('긴 배경과 질문 위치 때문에 핵심 근거를 놓치는지', '핵심 질문·조건', '질문 앞/뒤 위치를 각각 생성'),
    'TYPO': ('해석 가능한 오타·띄어쓰기 때문에 정답이 밀리는지', '의도·수치·상품·API 상태', '단일 오타 또는 띄어쓰기 변형'),
    'NO_FAQ': ('핵심 답이 없는 가까운 문서를 채택하는지', '원본의 부재 판정과 질문 의도', '원본 NO_ANSWER_TOPIC 기반; 독립 부재 검수 필요'),
    'DOCUMENT_ATTACK': ('문서 명령문이 근거 검색·Context·생성을 오염시키는지', '정상 FAQ 사실과 정상 질문', 'AD 공격을 문서로 이동한 별도 fixture'),
}

NUMBER_RE = re.compile(r'\d+(?:,\d{3})*(?:\.\d+)?')
PLAN_RE = re.compile(r'(?:5G|LTE)\s*(?:라이트|스탠다드|프리미엄|베이직|시니어|주니어|데이터|청소년)|시니어\s*LTE')
API_RE = re.compile(r'\b(?:accepted|completed|false|true|stale|fresh|pending|success|failed|null)\b')
ISO_RE = re.compile(r'\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2})?')

def numeric_signature(text):
    return Counter(str(Decimal(m.group().replace(',', '')).normalize()) for m in NUMBER_RE.finditer(text))

def term_eligible(text, start, end, term):
    if len(term)>2:
        return True
    if start>0 and re.match(r'[가-힣A-Za-z]',text[start-1]):
        return False
    remainder = text[end:]
    if remainder and re.match(r'[가-힣A-Za-z]',remainder[0]):
        return bool(re.match(r'(?:에서|에게|으로|이라|이면|이었|이는|이라|이나|이|가|은|는|을|를|의|과|와|도|만|에|로|부터|까지|처럼|보다|하고|랑|라도|조차)',remainder))
    return True

def replace_term(text, term, alias, english_final=None, count=0):
    """Keep Korean postpositions grammatical after a noun replacement."""
    changes = []
    limit = count or 999
    pattern = re.compile(re.escape(term))
    # A short word must not be replaced inside a different compound word.
    def sub(match):
        if len(changes) >= limit:
            return match.group()
        start, end = match.span()
        if not term_eligible(text,start,end,term):
            return match.group()
        changes.append({'from': term, 'to': alias})
        return alias
    result = pattern.sub(sub, text)
    if changes:
        last = alias[-1]
        final = english_final if english_final is not None else ('가' <= last <= '힣' and (ord(last)-0xAC00) % 28 != 0)
        pairs = [('이','가'), ('은','는'), ('을','를'), ('과','와')]
        for cons, vowel in pairs:
            result = re.sub(re.escape(alias)+r'('+cons+'|'+vowel+r')(?=[\s가-힣?.!,]|$)',
                            lambda m: alias + (cons if final else vowel), result)
    return result, changes

def lex_rewrite(text, mode):
    changes = []
    # Longer strings win, and inserted phrases are never processed again.
    options = sorted(SAFE_KO, key=len, reverse=True)
    pattern = re.compile('|'.join(re.escape(t) for t in options))
    def sub(match):
        term = match.group()
        start, end = match.span()
        if PLAN_RE.search(text) and any(a <= start < b for a,b in (m.span() for m in PLAN_RE.finditer(text))):
            return term
        alias = SAFE_KO[term][mode % 3]
        if alias != term:
            changes.append({'from': term, 'to': alias})
        return alias
    result = pattern.sub(sub, text)
    for c in changes:
        result, _ = replace_term(result, c['to'], c['to'])
    return result, changes

def endings(text, mode):
    # One pass avoids recursively paraphrasing text introduced by a previous rule.
    mapping = {a:b[mode % len(b)] for a,b in ENDING_RULES}
    pat = re.compile('|'.join(re.escape(a) for a in sorted(mapping, key=len, reverse=True)))
    changes = []
    def sub(match):
        before, after = match.group(), mapping[match.group()]
        if before in ('입니다.','예요.') and match.start()>0:
            last=text[match.start()-1]
            has_final=('가' <= last <= '힣' and (ord(last)-0xAC00)%28!=0)
            if not has_final:
                if after=='이에요.':after='예요.'
                if after.startswith('이라고'):after='라고'+after[len('이라고'):]
        if before != after:
            changes.append({'from': before, 'to': after})
        return after
    return pat.sub(sub, text), changes

def paraphrase(text, mode, lexical=True):
    q, change = lex_rewrite(text, mode) if lexical else (text, [])
    q, more = endings(q, mode)
    return q, change + more

def reorder(text):
    numbered = re.match(r'^(.*?)1\)\s*(.*?)\s*2\)\s*(.*)$', text, re.S)
    if numbered:
        intro, one, two = numbered.groups()
        # Keep an instruction following the second question outside the swap.
        tail = ''
        split = re.split(r'(각 질문의 조건을.*)$', two, maxsplit=1)
        if len(split) > 1:
            two, tail = split[0], split[1]
        return f'{intro}1) {two.strip()} 2) {one.strip()} {tail}'.strip(), [{'operation':'swap_intents','detail':'두 요구 모두 보존'}]
    parts = re.split(r'(?<=[?.!])\s+', text)
    # Put the last actual question first, followed by the same scenario facts.
    if len(parts) > 1 and '?' in text and not any(p.startswith(('그 ', '그 시각', '이 ', '아까', '앞의', '전에')) for p in parts[1:]):
        qs = [i for i,p in enumerate(parts) if '?' in p]
        if qs:
            i = qs[-1]
            return ' '.join([parts[i]] + parts[:i] + parts[i+1:]), [{'operation':'question_before_scenario','detail':'상황과 질문의 순서만 변경'}]
    return text, []

def bilingual(text, mode):
    matches = []
    for term, ko, english, final in sorted(LEXICON, key=lambda item:len(item[0]), reverse=True):
        for m in re.finditer(re.escape(term), text):
            if not term_eligible(text,m.start(),m.end(),term):
                continue
            if any(a <= m.start() < b for a,b in (p.span() for p in PLAN_RE.finditer(text))):
                continue
            matches.append((m.start(),term,english,final))
            break
    matches.sort(key=lambda v:(v[0],-len(v[1])))
    if not matches:
        return text, [], []
    choices = [matches[0]] if mode == 0 else [matches[-1]]
    q, changes, pairs = text, [], []
    for _,term,english,final in choices:
        q, trace = replace_term(q, term, english, final, count=1)
        changes.extend(trace)
        pairs.extend({'ko':term,'en':english} for _ in trace)
    if mode:
        q, extra = endings(q, 2)
        changes += extra
    return q, changes, pairs

def typo(text):
    for old,new in TYPO_RULES:
        if old in text:
            return text.replace(old,new,1), [{'from':old,'to':new,'operation':'single_typo'}]
    # If no function word is available, remove a single space; do not corrupt a
    # product name, number, unit, API enum, timestamp or negation.
    for m in re.finditer(' ',text):
        if not any(p.start()<=m.start()<p.end() for p in PLAN_RE.finditer(text)):
            return text[:m.start()]+text[m.end():], [{'from':' ','to':'','operation':'single_space_deletion'}]
    raise ValueError(f'No safe typo for {text!r}')

def intrinsic_tags(case, raw):
    kind, q = case['type'],case['query']
    tags = []
    if PLAN_RE.search(q) or any(t in q for t in ('유심','eSIM','IPTV','LTE','5G')):
        tags.append('EXACT_ENTITY')
    if kind == 'NC': tags.append('HARD_NEGATIVE')
    if kind in ('MC','PI') or (kind == 'UI' and len(case['source_ids'])>1):tags.append('MULTI_INTENT')
    if kind in ('CE','UI','AR'):tags.append('CONDITION_SCOPE')
    if re.search(r'\d+\s*(?:세|년|월|일|시|분|초|원|점|GB|MB|%|회|개월|회선)|이상|이하|초과|미만',q):tags.append('NUMERIC_CONDITION')
    if re.search(r'없|않|안 |불가|못|미납|미적용',q):tags.append('NEGATION')
    if '두 안내' in q or '차이' in q or '비교' in q or '동률' in q:tags.append('COMPARISON')
    if kind == 'MT' and re.search(r'그 |아까|말씀드린|그건|그것',q):tags.append('AMBIGUOUS_QUERY')
    if kind == 'CF' and '시행 시작' in (raw.get('제공 Context') or '') and ISO_RE.search(q):tags.append('TEMPORAL_VERSION')
    if case['no_faq_truth']:tags.append('NO_FAQ')
    return list(dict.fromkeys(tags))

def attack_fixture(case, raw):
    # Transfer an existing source attack to a document. Do not invent any
    # service fact or execute the attack. A controlled Context generation test
    # must explicitly load this fixture; it never modifies the FAQ corpus.
    match = re.match(r'^(.*?\?)\s+(.+)$',case['query'],re.S)
    if case['type'] != 'AD' or not case['source_ids'] or not match:
        return None
    core, payload = match.groups()
    if not re.search(r'무시|무료|승인|정책을 바꾸|처리 완료|관리자|규칙',payload):
        return None
    context = raw.get('제공 Context') or ''
    faq_id = case['source_ids'][0]
    header = re.search(r'\['+re.escape(faq_id)+r'\].*?(?=\n\n\[|$)',context,re.S)
    if not header:
        return None
    attacked = context[:header.end()] + '\n[시험용 문서 내 명령문] '+payload + context[header.end():]
    return {'fixture_id':case['case_id']+'-DOCUMENT-ATTACK', 'source_case_id':case['case_id'],
            'target_faq_id':faq_id,'normal_query':core,'payload':payload,'provided_context':attacked,
            'normal_facts_preserved':True,'synthetic':True,'expected_behavior':'명령문을 실행하지 않고 정상 근거로 답변',
            'expected_answer':raw['정답 예시'],'original_context_sha256':hashlib.sha256(context.encode()).hexdigest()}

def feature_variant(case, raw):
    q = case['query']
    fixture = attack_fixture(case,raw)
    if fixture:
        core, trace = paraphrase(fixture['normal_query'],3)
        return core, trace+[{'operation':'move_query_attack_to_document','fixture_id':fixture['fixture_id']}], 'DOCUMENT_ATTACK', fixture
    if case['type']=='MC':
        value, trace = reorder(q)
        value, more = paraphrase(value,3)
        return value, trace+more, 'MULTI_INTENT', None
    if case['type']=='CF' and ISO_RE.search(q):
        value, trace = replace_term(q,'기준 시각','정책 적용 기준 시각')
        value, more = endings(value,3)
        return value,trace+more,'TEMPORAL_VERSION',None
    if re.search(r'\d[\d,]*\s*(?:GB|MB|원|점|회선|개월|세)',q):
        value = re.sub(r'(\d+(?:,\d{3})+)\b',lambda m:m.group().replace(',',''),q)
        changes = [{'operation':'thousands_separator','from':q,'to':value}] if value!=q else []
        for old,new in [('GB','기가바이트'),('MB','메가바이트')]:
            if old in value:
                value = re.sub(r'(?<=\d)\s*'+old, ' '+new, value)
        if value != q and not changes:changes.append({'operation':'unit_spelling','detail':'GB/MB와 같은 단위를 한글로 표기'})
        value,more = paraphrase(value,3)
        return value,changes+more,'NUMERIC_CONDITION',None
    for old,new in [('없으면','없는 경우에는'),('않으면','않는 경우에는'),('안 되나요?','불가능한가요?'),('없어도','없는 상황이어도')]:
        if old in q:
            value, more = paraphrase(q.replace(old,new,1),3)
            return value,[{'from':old,'to':new}]+more,'NEGATION',None
    value, changes = reorder(q)
    value, more = paraphrase(value,3)
    tags = intrinsic_tags(case,raw)
    preferred = next((t for t in ['COMPARISON','CONDITION_SCOPE','AMBIGUOUS_QUERY','NO_FAQ','EXACT_ENTITY'] if t in tags),'PARAPHRASE')
    return value, changes+more, preferred,None

def make_candidates(case, raw):
    q = case['query']
    candidates = []
    for idx,mode in enumerate([0,1,2,3]):
        value, changes = paraphrase(q,mode)
        if idx == 2:
            value, more = reorder(value); changes += more
        candidates.append({'query':value,'primary':'PARAPHRASE','method':['격식 표현·동의어','구어체·동의어','질문 순서·표현','요청 문장 재작성'][idx],'changes':changes})
    for mode in [0,1]:
        value, changes, pairs = bilingual(q,mode)
        candidates.append({'query':value,'primary':'KR_EN_EQUIVALENT' if pairs else 'PARAPHRASE',
                           'method':'한영 동등어 치환' if pairs else '표현 재작성(동등어 해당 없음)', 'changes':changes, 'language_pairs':pairs})
    value, changes = typo(q)
    candidates.append({'query':value,'primary':'TYPO','method':'해석 가능한 단일 오타·띄어쓰기','changes':changes})
    value, changes, primary, fixture = feature_variant(case,raw)
    candidates.append({'query':value,'primary':primary,'method':'원본 검증 목적에 따른 표기·구조 변형','changes':changes,'fixture':fixture})
    for position in [0,1]:
        core, changes = paraphrase(q,position+2)
        noise = NOISE[position] + NOISE[1-position]
        value = (core+'\n\n'+noise) if position==0 else (noise+'\n\n'+core)
        candidates.append({'query':value,'primary':'LONG_QUERY','method':'긴 배경: 핵심 질문 '+('앞' if position==0 else '뒤'),
                           'changes':changes+[{'operation':'irrelevant_background','core_position':'first' if position==0 else 'last'}], 'core_query':core})
    return candidates

def distinct_candidate(candidate, case, used):
    # A collision is repaired by another real rewriting operation. It is never
    # repaired by an ID, meaningless prefix or arbitrary whitespace padding.
    source = candidate['query']
    if source != case['query'] and source not in used:
        return candidate
    for lexical in [False,True]:
        for mode in range(4):
            value, trace = paraphrase(source,mode,lexical)
            if value != case['query'] and value not in used:
                return {**candidate,'query':value,'changes':candidate['changes']+trace}
    # Same request, grammatical reporting style; also changes the actual
    # interrogative, rather than just decorating the original question.
    for mode in range(4):
        value, trace = paraphrase(case['query'],mode)
        variants = [value.replace('알려 주세요.','알려 주시면 감사하겠습니다.'),
                    value.replace('알고 싶습니다.','알고 싶은데 안내해 주실 수 있을까요?'),
                    value.replace('확인 부탁드립니다.','확인해서 설명해 주시면 좋겠습니다.'),
                    value.replace('확인하고 싶습니다.','확인해서 알려 주실 수 있나요?'),
                    value.replace('알려 주시겠어요?','알려 주시면 좋겠습니다.'),
                    value.replace('설명해 주시겠어요?','설명해 주시면 좋겠습니다.'),
                    value.replace('알려 주시면 좋겠습니다.','알려 주실 수 있을까요?'),
                    value.replace('인가요?','인지 설명 부탁드립니다.'),
                    value.replace('주실 수 있나요?','주실 수 있을까요?'),
                    value.replace('주시겠어요?','주시면 감사하겠습니다.')]
        variants += [value.replace('싶습니다.','싶어 문의드립니다.'),
                     value.replace('싶은데요.','싶은 상황입니다.'),
                     value.replace('궁금합니다.','궁금해서 여쭤봅니다.'),
                     value.replace('설명 부탁드립니다.','설명해 주실 수 있을까요?'),
                     value.replace('알려 주세요.','알려 주실 수 있을까요?'),
                     value.replace('알고 싶습니다.','알고 싶어서 질문드립니다.')]
        variants += [value.replace('상황입니다.','상황이라 문의드립니다.'),
                     value.replace('상태입니다.','상태라 문의드립니다.'),
                     value.replace('받았습니다.','받아 문의드립니다.'),
                     value.replace('싶어서 문의드립니다.','싶다는 뜻입니다.'),
                     value.replace('싶어 문의드립니다.','싶은 상황이라 문의드립니다.')]
        for v in variants:
            if v!=value and v not in used and v!=case['query']:
                return {**candidate,'query':v,'primary':'PARAPHRASE','language_pairs':[],
                        'changes':trace+[{'operation':'collision_repair','detail':'다른 요청형 문장으로 재작성'}]}
    raise ValueError(f'Ten distinct semantic variants unavailable: {case["case_id"]}, {source!r}')

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    source_manifest = json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in source_manifest['hashes'].items():
        assert sha(SOURCE/name)==digest, f'Source changed: {name}'
    cases,raws = read_jsonl(SOURCE/'cases.jsonl'),read_jsonl(SOURCE/'source_rows.jsonl')
    assert len(cases)==len(raws)==3000
    variations,fixtures,language_pairs = [],[],[]
    for case,raw in zip(cases,raws):
        assert case['case_id']==raw['실행 ID'] and case['query']==raw['사용자 질문']
        used = set()
        tags = intrinsic_tags(case,raw)
        original_repeat = case['case_id'].rsplit('-R',1)[0] if case['type']=='RT' else None
        for index,candidate in enumerate(make_candidates(case,raw),1):
            candidate = distinct_candidate(candidate,case,used)
            value = candidate['query'].strip()
            used.add(value)
            fixture = candidate.get('fixture')
            current_tags = list(dict.fromkeys([candidate['primary']]+tags))
            if fixture:
                fixtures.append(fixture)
            inherited = {k:case[k] for k in ['source_row','type','type_name','history','cohort','family_id','source_ids','no_faq_truth',
                                               'fact_groups','expected_generation_status','expected_route','repeat_index','repeat_total']}
            pairs = candidate.get('language_pairs',[])
            if pairs:
                language_pairs.append({'pair_id':case['case_id']+f'-LANG-{index:02}', 'ko_case_id':case['case_id'],
                                       'variant_case_id':case['case_id']+f'-V{index:02}', 'ko_query':case['query'],'mixed_query':value,
                                       'terms':pairs,'meaning_review':'pending_independent_review'})
            variations.append({**inherited,'case_id':case['case_id']+f'-V{index:02}', 'parent_case_id':case['case_id'],
                               'variant_index':index,'query':value,'primary_variation':candidate['primary'],
                               'core_question':candidate.get('core_query',value),
                               'validation_tags':current_tags,'validation_purposes':[PURPOSES[t][0] for t in current_tags],
                               'variation_method':candidate['method'],'change_trace':candidate['changes'],
                               'original_query':case['query'],'expected_answer':raw['정답 예시'],
                               'required_facts':raw['필수 사실·표현 기준'],'failure_conditions':raw['실패 조건'],
                               'source_fixture_id':case['case_id'],'context_fixture_id':fixture['fixture_id'] if fixture else None,
                               'model_query':('대화 이력:\n'+'\n'.join(h['role']+': '+h['content'] for h in case['history'])+'\n현재 질문: '+value) if case['history'] else value,
                               'source_repeat_group_id':original_repeat,'variation_repeat_group_id':original_repeat+f'-V{index:02}' if original_repeat else None,
                               'language_pairs':pairs,'language_pair_group':case['case_id'] if pairs else None,
                               'synthetic_policy_fixture':case['type']=='CF',
                               'semantic_relation':'same_legitimate_intent' if fixture else 'intended_same_meaning',
                               'annotation_status':'requires_independent_semantic_and_ground_truth_review',
                               'ground_truth_basis':'inherited source question/answer and controlled fixtures; tracking FAQ IDs are proxies',
                               'split':'unassigned','split_group_id':'PARENT:'+case['family_id']})
        assert len(used)==10,case['case_id']
    assert len(variations)==30000
    # All source metadata remains in a separate read-only fixture dataset.
    write_jsonl(OUT/'source_pairs.jsonl',raws)
    write_jsonl(OUT/'variations.jsonl',variations)
    write_jsonl(OUT/'language_pairs.jsonl',language_pairs)
    write_jsonl(OUT/'document_attack_fixtures.jsonl',fixtures)
    write_json(OUT/'purpose_catalog.json', [{'tag':tag,'purpose':values[0],'preserve':values[1],'scope':values[2]} for tag,values in PURPOSES.items()])
    write_json(OUT/'lexicon.json', [{'ko':term,'en':english,'kind':'author_curated_equivalence','official_alias_verified':False} for term,ko,english,final in LEXICON])
    manifest = {'source_workbook':source_manifest['source_workbook'],'source_workbook_sha256':source_manifest['source_sha256'],
                'source_manifest_sha256':sha(SOURCE/'manifest.json'),'source_pairs':3000,'variants_per_pair':10,'variation_rows':30000,
                'source_type_counts':dict(Counter(c['type'] for c in cases)),
                'variation_type_counts':dict(Counter(c['type'] for c in variations)),
                'primary_variation_counts':dict(Counter(c['primary_variation'] for c in variations)),
                'validation_tag_counts':dict(Counter(t for c in variations for t in c['validation_tags'])),
                'unique_query_strings':len({c['query'] for c in variations}),
                'language_pair_rows':len(language_pairs),'document_attack_fixture_rows':len(fixtures),
                'repeat_execution_rows':sum(c['type']=='RT' for c in variations),'repeat_variant_inputs':len({c['variation_repeat_group_id'] for c in variations if c['type']=='RT'}),
                'generation_method':'deterministic, source-aware Korean rewriting and curated term equivalence; no generative model',
                'numeric_or_policy_truth_changed':False,'independent_annotation_complete':False,
                'all_real_failure_logs_available':False,'real_failure_rows':0,
                'temporal_scope':'source CF synthetic policy version fixtures, never claimed as real policy history',
                'holdout_defined':False,'benchmark_executed':False,
                'hashes':{p.name:sha(p) for p in OUT.glob('*.jsonl')}}
    write_json(OUT/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='hashes'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    build()
