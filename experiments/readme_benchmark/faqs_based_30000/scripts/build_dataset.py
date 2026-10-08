"""Build fresh silver test cases from the supplied FAQ snapshot only.

No old evaluation data, model scores, thresholds, or generated answers are read.
Source text is untrusted data. It is never executed or used as instructions.
"""
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / 'data', ROOT / 'outputs'
SOURCE_SHA = '76d867e4b4e183e6b52c8edfb31ec7da4e74f8495b4a2a2ce10fb2472ecf8a2d'
VERSION = 'faqs-source-rules-v1'


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def norm(value):
    return re.sub(r'\s+', ' ', value or '').strip()


def dump(name, value, directory=DATA):
    (directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def lines(name, rows, directory=DATA):
    with (directory / name).open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


# Generic terms and established spellings only. These are NOT interchangeable
# products, eligibility classes, SIM types, policy values, or official plan aliases.
# Product labels inside brackets/quotes are protected from term replacement.
LEXICON = [
    ('휴대전화', 'mobile phone'), ('휴대폰', 'mobile phone'), ('스마트폰', 'smartphone'),
    ('본인인증', 'identity verification'), ('본인 인증', 'identity verification'),
    ('번호이동', 'mobile number portability'), ('번호 이동', 'mobile number portability'),
    ('부가서비스', 'add-on service'), ('부가 서비스', 'add-on service'),
    ('고객센터', 'customer service center'), ('고객 센터', 'customer service center'),
    ('요금제', 'rate plan'), ('신분증', 'ID card'), ('유심', 'USIM'), ('이심', 'eSIM'),
    ('와이파이', 'Wi-Fi'), ('테더링', 'tethering'), ('데이터 쉐어링', 'data sharing'),
    ('데이터 셰어링', 'data sharing'), ('청구서', 'bill'), ('비밀번호', 'password'),
    ('위약금', 'termination penalty'), ('약정기간', 'contract period'),
    ('할인반환금', 'discount repayment'), ('일시정지', 'temporary suspension'),
    ('데이터 로밍', 'data roaming'), ('로밍', 'roaming'), ('인터넷', 'internet'),
    ('문자메시지', 'text message'), ('문자 메시지', 'text message'),
    ('음성사서함', 'voicemail'), ('착신전환', 'call forwarding'),
    ('착신 전환', 'call forwarding'), ('배터리', 'battery'),
    ('애플리케이션', 'application'), ('앱', 'app'), ('아이폰', 'iPhone'),
    ('미성년자', 'minor'), ('법정대리인', 'legal guardian'),
    ('법정 대리인', 'legal guardian'), ('여권', 'passport'), ('쿠폰', 'coupon'),
    ('호텔', 'hotel'), ('공항', 'airport'), ('교통카드', 'transportation card'),
    ('택시', 'taxi'),
]
PROTECTED = re.compile(r'\[[^\]]+\]|〈[^〉]+〉|<[^>]+>|「[^」]+」|“[^”]+”|"[^"]+"|\b(?:[A-Za-z]*\d[A-Za-z0-9+/-]*)\b')
POLARITY = re.compile(r'없|않|아니|불가|불가능|미적용|미납|안\s*(?:되|돼|해|했|됐)|이상|이하|초과|미만|이내|이후|이전|\b(?:not|never|without|cannot)\b', re.I)
NUMERIC = re.compile(r'\d[\d,.]*\s*(?:원|만원|개월|년|월|일|시간|분|초|회|개|대|명|세|GB|MB|KB|kbps|Mbps|%)', re.I)
CONDITION = re.compile(r'경우|하면|이면|때|신규|기존|미성년|외국인|법인|국내|해외|전용|제외|제한|만\s*\d')
COMPARISON = re.compile(r'차이|비교|다른 점|difference|compare|which.+(?:SIM|product)', re.I)

# Conservative changes to question phrasing. Never flip polarity or comparator.
KO_RULES = [
    (r'어떻게 해야\s*하나요[?？.]?$', ['어떤 절차로 처리해야 하나요?', '어떻게 처리하면 돼요?', '처리 방법을 알려주세요.']),
    (r'무엇인가요[?？.]?$', ['어떤 것인가요?', '뭔가요?', '무엇인지 설명해 주세요.']),
    (r'뭔가요[?？.]?$', ['무엇인가요?', '어떤 건지 알려줘요.', '무엇인지 궁금합니다.']),
    (r'가능한가요[?？.]?$', ['가능한지 궁금합니다.', '가능해요?', '가능한지 알려주세요.']),
    (r'가능하나요[?？.]?$', ['가능한지 궁금합니다.', '가능해요?', '가능한지 알려주세요.']),
    (r'할 수 있나요[?？.]?$', ['할 수 있는지 궁금합니다.', '할 수 있어요?', '할 수 있는지 알려주세요.']),
    (r'받을 수 있나요[?？.]?$', ['받을 수 있는지 궁금합니다.', '받을 수 있어요?', '받을 수 있는지 알려주세요.']),
    (r'쓸 수 있나요[?？.]?$', ['쓸 수 있는지 궁금합니다.', '쓸 수 있어요?', '쓸 수 있는지 알려주세요.']),
    (r'없나요[?？.]?$', ['없는지 궁금합니다.', '없어요?', '없는지 확인해 주세요.']),
    (r'있나요[?？.]?$', ['있는지 궁금합니다.', '있어요?', '있는지 알려주세요.']),
    (r'되나요[?？.]?$', ['되는지 궁금합니다.', '돼요?', '되는지 알려주세요.']),
    (r'얼마인가요[?？.]?$', ['얼마인지 궁금합니다.', '얼마예요?', '얼마인지 알려주세요.']),
    (r'어떻게 하나요[?？.]?$', ['어떤 방식으로 하나요?', '어떻게 하면 돼요?', '진행 방법을 알려주세요.']),
    (r'알고 싶어요[?？.]?$', ['알고 싶습니다.', '설명해 줄 수 있어요?', '안내를 부탁드립니다.']),
    (r'알려\s*주세요[?？.]?$', ['안내해 주세요.', '알려줘요.', '설명해 주세요.']),
    (r'궁금해요[?？.]?$', ['궁금합니다.', '설명해 줄래요?', '안내를 부탁드립니다.']),
    (r'다른가요[?？.]?$', ['다른지 궁금합니다.', '달라요?', '다른지 설명해 주세요.']),
    (r'발생하나요[?？.]?$', ['발생하는지 궁금합니다.', '발생해요?', '발생하는지 알려주세요.']),
    (r'달라지나요[?？.]?$', ['달라지는지 궁금합니다.', '달라져요?', '달라지는지 알려주세요.']),
    (r'바뀌나요[?？.]?$', ['바뀌는지 궁금합니다.', '바뀌어요?', '바뀌는지 알려주세요.']),
    (r'해야\s*하나요[?？.]?$', ['해야 하는지 궁금합니다.', '해야 해요?', '해야 하는지 알려주세요.']),
    (r'되는 건가요[?？.]?$', ['되는 것인지 궁금합니다.', '되는 거예요?', '되는 것인지 알려주세요.']),
    (r'그런 건가요[?？.]?$', ['그런 것인지 궁금합니다.', '그런 거예요?', '그런 것인지 알려주세요.']),
]
EN_RULES = [
    (r'^How can I ', ['What is the way to ', 'How do I ', 'Could you explain how I can ']),
    (r'^Can I ', ['Is it possible for me to ', 'Am I able to ', 'Could you tell me whether I can ']),
    (r'^How do I ', ['What is the way to ', 'Could you explain how I ', 'Please explain how I ']),
    (r'^Where can I ', ['Where is it possible for me to ', 'Could you tell me where I can ', 'Please explain where I can ']),
]


def rewrite(q, language, mode):
    rules = EN_RULES if language == 'en' else KO_RULES
    for pattern, replacements in rules:
        m = re.search(pattern, q, re.I if language == 'en' else 0)
        if m:
            replacement = replacements[mode]
            return q[:m.start()] + replacement + q[m.end():], {'operation': 'grammar_rewrite', 'from': m.group(), 'to': replacement}
    # A fallback is explicitly a surface/request-frame change, not deep paraphrase.
    frames = (['Please explain: ', 'I have a question: ', 'Could you help me understand: '] if language == 'en'
              else ['다음 내용에 대해 안내를 부탁드립니다. ', '궁금한 게 있어요. ', '이 내용을 설명해 주세요. '])
    return frames[mode] + q, {'operation': 'request_frame_only', 'deep_paraphrase': False}


def exchange(q, reverse=False, many=False):
    spans = [(m.start(), m.end()) for m in PROTECTED.finditer(q)]
    candidates = []
    seen = set()
    for ko, en in sorted(LEXICON, key=lambda p: -len(p[1] if reverse else p[0])):
        a, b = (en, ko) if reverse else (ko, en)
        key = a.casefold()
        if key in seen:
            continue
        seen.add(key)
        # English aliases are matched on word boundaries; eSIM and USIM stay
        # distinct. Adjacent Korean particles are allowed for mixed queries.
        pattern = re.escape(a) if not reverse else r'(?<![A-Za-z])' + re.escape(a) + r'(?![A-Za-z])'
        for m in re.finditer(pattern, q, re.I if reverse else 0):
            if not reverse and a == '로밍' and q[m.end():].startswith(('패스', '쿠폰')):
                continue  # Preserve unquoted product names too.
            if not any(m.start() < end and m.end() > start for start, end in spans):
                if not any(m.start() < end and m.end() > start for start, end, *_ in candidates):
                    candidates.append((m.start(), m.end(), m.group(), b, ko, en))
    candidates.sort()
    if not many:
        candidates = candidates[:1]
    result = q
    for start, end, _, to, _, _ in reversed(candidates):
        result = result[:start] + to + result[end:]
    trace = [{'operation': 'equivalent_term', 'from': a, 'to': b, 'ko': ko, 'en': en} for _, _, a, b, ko, en in candidates]
    return result, trace


def move_topic(q, language):
    m = re.match(r'^(\[[^\]]+\])\s*(.+)$', q)
    if m:
        topic, body = m.groups()
        return body + (' Topic: ' if language == 'en' else ' 문의 대상: ') + topic, {'operation': 'move_explicit_topic', 'from': topic, 'to': topic}
    body, trace = rewrite(q, language, 2)
    return body, trace


def typo(q, language):
    spans = [(m.start(), m.end()) for m in PROTECTED.finditer(q)]
    pattern = r'(?<=[가-힣])\s+(?=[가-힣])' if language != 'en' else r'\b(?:can|the|please|with)\b'
    for m in re.finditer(pattern, q, re.I):
        if any(m.start() < end and m.end() > start for start, end in spans):
            continue
        replacement = '' if language != 'en' else {'can': 'cna', 'the': 'teh', 'please': 'pleaes', 'with': 'wiht'}[m.group().lower()]
        return q[:m.start()] + replacement + q[m.end():], {'operation': 'readable_typo', 'from': m.group(), 'to': replacement, 'position': m.start()}
    match = re.search(r'요(?=[?.!]?$)', q) if language != 'en' else None
    if match:
        return q[:match.start()] + '여' + q[match.end():], {'operation': 'readable_typo', 'from': '요', 'to': '여', 'position': match.start()}
    ending = ' Please explian.' if language == 'en' else ' 알려주세여.'
    return q + ending, {'operation': 'request_suffix_typo', 'detail': ending, 'deep_typo': False}


def numeric_or_scope(q, language):
    m = re.search(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?', q)
    if m:
        changed = q[:m.start()] + m.group().replace(',', '') + q[m.end():]
        return changed, {'operation': 'thousands_separator', 'from': m.group(), 'to': m.group().replace(',', '')}, 'NUMERIC_CONDITION'
    m = re.search(r'(\d)(원|개월|일|시간|회|세|GB|MB|kbps|%)', q, re.I)
    if m:
        # Exclude generation/network identifiers: 3G/4G/5G are not units.
        changed = q[:m.start()] + m.group(1) + ' ' + m.group(2) + q[m.end():]
        return changed, {'operation': 'numeric_unit_spacing', 'from': m.group(), 'to': m.group(1) + ' ' + m.group(2)}, 'NUMERIC_CONDITION'
    if CONDITION.search(q) or POLARITY.search(q):
        frame = 'Please keep the stated conditions when answering: ' if language == 'en' else '질문에 적힌 적용 조건을 기준으로 확인하고 싶습니다. '
        return frame + q, {'operation': 'condition_emphasis_only', 'contrast_pair': False}, 'CONDITION_SCOPE'
    frame = 'My specific question is the following: ' if language == 'en' else '확인하고 싶은 내용은 다음과 같습니다. '
    return frame + q, {'operation': 'request_frame_only', 'deep_paraphrase': False}, 'PARAPHRASE'


KO_BACKGROUND = [
    '질문과 무관한 배경을 덧붙입니다. 출퇴근길에 읽던 책의 순서를 정리하고, 주말에 들를 카페와 산책 코스를 생각하고 있습니다. 책을 어디에 둘지와 커피를 따뜻하게 마실지에 관한 이야기는 이번 문의의 조건이 아닙니다. 그 일들은 따로 결정할 예정이라 여기서 답을 요청하지 않습니다. ',
    '주변 사람들과 저녁 메뉴 이야기를 하다가 생각이 다른 방향으로 흘렀습니다. 책상 정리나 날씨에 관한 이야기도 있었지만, 모두 이번 문의와 관계없는 배경입니다. 그 내용을 서비스 조건이나 별도의 요청으로 받아들이지 말아 주세요. 서비스에 관한 핵심 질문은 따로 명시한 내용입니다. ',
    '이동 중에 남기는 문의라 설명이 길어졌습니다. 가방 안의 물건을 정리하고 메모의 순서를 바꾸면서 여러 일상이 떠올랐습니다. 걷던 길과 음식에 관한 취향은 이번 문의의 조건이 아닙니다. 이러한 주변 이야기에 대해서는 답을 구하지 않고, 명시한 서비스 질문만 확인하고 싶습니다. ',
]
EN_BACKGROUND = [
    'Here is unrelated background. I have been arranging books, considering a walk, and thinking about coffee. These details do not describe my service eligibility or any additional request. I will decide those matters separately. The only question I want answered is the service question explicitly stated here. ',
    'This message became long because I am writing while travelling. I was also talking about dinner, the weather, and organising a desk. Those topics are background only; they are not conditions of the service question and I am not requesting answers about them. Please focus on the explicitly stated question. ',
]


def quotas(groups, n):
    total = sum(map(len, groups.values()))
    # Reserve every eligible brand/category/language stratum, then distribute
    # remaining slots by bounded largest remainder of proportional deficits.
    quota = {k: 1 for k in groups}
    while sum(quota.values()) < n:
        k = max((k for k in groups if quota[k] < len(groups[k])),
                key=lambda k: (n * len(groups[k]) / total - quota[k], str(k)))
        quota[k] += 1
    return quota


def prepare_source():
    raw_bytes = (ROOT / 'source' / 'faqs.jsonl').read_bytes()
    assert hashlib.sha256(raw_bytes).hexdigest() == SOURCE_SHA, 'Source snapshot changed'
    raw = [json.loads(s) for s in raw_bytes.decode('utf-8-sig').splitlines() if s.strip()]
    assert len(raw) == 3365 and len({r['faq_id'] for r in raw}) == len(raw)
    captured = [r for r in raw if norm(r.get('question')) and norm(r.get('answer'))]
    by_question = defaultdict(set)
    by_pair = defaultdict(list)
    for r in captured:
        by_question[(r['brand'], norm(r['question']))].add(norm(r['answer']))
        by_pair[(r['brand'], norm(r['question']), norm(r['answer']))].append(r)
    canonical, review = [], []
    for key, members in sorted(by_pair.items()):
        primary = sorted(members, key=lambda r: r['faq_id'])[0]
        flags = []
        if any(r.get('question_status') == 'INFERRED_FROM_ANSWER' for r in members):
            flags.append('question_inferred_from_answer')
        if len(by_question[key[:2]]) > 1:
            flags.append('same_brand_question_multiple_answers')
        faq = {
            'faq_id': 'FAQ-' + digest('\x1f'.join(key))[:20],
            'brand': primary['brand'], 'category': primary.get('category') or '',
            'language': primary.get('language') or 'ko',
            'question': norm(primary['question']), 'answer': primary['answer'],
            'primary_source_faq_id': primary['faq_id'],
            'source_faq_ids': sorted(r['faq_id'] for r in members),
            'source_urls': sorted({r.get('source_url') or '' for r in members}),
            'source_ids': sorted({r.get('source_id') or '' for r in members}),
            'source_categories': sorted({r.get('category') or '' for r in members}),
            'retrieved_on': sorted({r.get('retrieved_on') or '' for r in members}),
            'published_at': sorted({r.get('published_at') or '' for r in members} - {''}),
            'updated_at': sorted({r.get('updated_at') or '' for r in members} - {''}),
            'source_issue_flags': flags, 'seed_eligible': not flags,
            'corpus_eligible': 'question_inferred_from_answer' not in flags,
        }
        canonical.append(faq)
        for member in members:
            if flags:
                review.append({'source_faq_id': member['faq_id'], 'canonical_faq_id': faq['faq_id'], 'reasons': flags, 'seed_eligible': False, 'corpus_eligible': faq['corpus_eligible']})
    for r in raw:
        if not norm(r.get('answer')) or not norm(r.get('question')):
            review.append({'source_faq_id': r['faq_id'], 'canonical_faq_id': None, 'reasons': ['missing_text_answer', r.get('content_status', '')], 'seed_eligible': False, 'corpus_eligible': False})
    lines('canonical_source.jsonl', canonical)
    lines('source_review.jsonl', review)
    corpus = [r for r in canonical if r['corpus_eligible']]
    lines('corpus.jsonl', corpus)
    return raw, canonical, corpus, review


def pair_sources(selected):
    # Disjoint groups of two (last odd group has three) prevent multi-intent
    # links from connecting an entire brand into one unusable split component.
    groups = defaultdict(list)
    for r in selected:
        groups[(r['brand'], r['language'])].append(r)
    partners, split_groups = {}, {}
    for key, rs in sorted(groups.items()):
        rs.sort(key=lambda r: (r['category'], digest(VERSION + r['faq_id'])))
        assert len(rs) >= 2, ('Cannot create distinct same-brand/language partner', key)
        i = 0
        while i < len(rs):
            size = 3 if len(rs) - i == 3 else 2
            members = rs[i:i + size]
            group_id = 'SPLIT-' + digest('|'.join(sorted(r['faq_id'] for r in members)))[:16]
            for j, r in enumerate(members):
                partners[r['faq_id']] = members[(j + 1) % size]
                split_groups[r['faq_id']] = group_id
            i += size
    return partners, split_groups


def tags_for(q):
    tags = []
    if re.search(r'\[[^\]]+\]|〈[^〉]+〉|eSIM|USIM|TOUR\s*PASS|CHECK\s*iN|baro|Netflix', q, re.I):
        tags.append('EXACT_ENTITY')
    if POLARITY.search(q):
        tags.append('NEGATION' if re.search(r'없|않|불가|아니|안\s*(?:되|돼)|\b(?:not|never|cannot|without)\b', q, re.I) else 'CONDITION_SCOPE')
    if NUMERIC.search(q):
        tags.append('NUMERIC_CONDITION')
    if CONDITION.search(q):
        tags.append('CONDITION_SCOPE')
    if COMPARISON.search(q):
        tags.append('COMPARISON')
    return tags


def scope_for(faq):
    needed = len(faq['question']) < 16 or bool(re.search(r'그것|그거|이 상품|이 서비스|이 요금제|\b(?:the card|the product|your service)\b', faq['question'], re.I))
    return faq['category'] if needed else ''


def variants(faq, partner, index):
    q, language = faq['question'], faq['language']
    formal, ft = rewrite(q, language, 0)
    spoken, st = rewrite(q, language, 1)
    topic, tt = move_topic(q, language)
    single, single_trace = exchange(q, reverse=language == 'en')
    mixed, many_trace = exchange(q, reverse=language == 'en', many=True)
    # Korean FAQs often contain English originals too (eSIM, Wi-Fi, app).
    if not single_trace and language != 'en':
        single, single_trace = exchange(q, reverse=True)
        mixed, many_trace = exchange(q, reverse=True, many=True)
    if single_trace:
        v4 = single
        v5 = mixed
        if mixed == single:
            v5, changed = rewrite(mixed, language, 0)
            many_trace = many_trace + [changed]
        t4 = t5 = 'KR_EN_EQUIVALENT'
    else:
        v4 = ('Regarding this question: ' if language == 'en' else '이 문의를 확인하고 싶습니다. ') + topic
        v5 = ('Please answer this point: ' if language == 'en' else '이 부분에 대한 답을 부탁드립니다. ') + spoken
        single_trace = many_trace = [{'operation': 'no_applicable_equivalent_term', 'replacement_slot': 'PARAPHRASE', 'deep_paraphrase': False}]
        t4 = t5 = 'PARAPHRASE'
    misspelled, mt = typo(q, language)
    scoped, ct, scope_tag = numeric_or_scope(q, language)
    background = (EN_BACKGROUND[index % len(EN_BACKGROUND)] if language == 'en' else KO_BACKGROUND[index % len(KO_BACKGROUND)]) * 2
    core_first = q + (' This is my only service question. ' if language == 'en' else ' 이 내용이 핵심 문의입니다. ') + background
    core_last = background + (' My only service question is: ' if language == 'en' else ' 마지막으로 핵심 문의를 남깁니다. ') + q
    companion, pt = rewrite(partner['question'], language, 0)
    companion_scope = scope_for(partner)
    if companion_scope:
        companion = companion_scope + ': ' + companion
    multi = (q + ' Separately, I also have this question: ' + companion if language == 'en'
             else q + ' 별개의 문의도 함께 드립니다. ' + companion)
    return [
        ('V01', formal, 'PARAPHRASE', [ft], [faq]),
        ('V02', spoken, 'PARAPHRASE', [st], [faq]),
        ('V03', topic, 'EXACT_ENTITY' if tt['operation'] == 'move_explicit_topic' else 'PARAPHRASE', [tt], [faq]),
        ('V04', v4, t4, single_trace, [faq]),
        ('V05', v5, t5, many_trace, [faq]),
        ('V06', misspelled, 'TYPO', [mt], [faq]),
        ('V07', scoped, scope_tag, [ct], [faq]),
        ('V08', core_first, 'LONG_QUERY', [{'operation': 'irrelevant_background', 'core_position': 'first', 'background': background}], [faq]),
        ('V09', core_last, 'LONG_QUERY', [{'operation': 'irrelevant_background', 'core_position': 'last', 'background': background}], [faq]),
        ('V10', multi, 'MULTI_INTENT', [{'operation': 'combine_two_source_questions', 'companion_rewrite': pt, 'companion_user_scope': companion_scope}], [faq, partner]),
    ]


PURPOSES = [
    ('REAL_FAILURE', '실제 장애가 구조 변경으로 해결되는지', '실패 로그·실패 당시 코퍼스가 없어 생성하지 않음'),
    ('EXACT_ENTITY', '명시된 서비스·상품을 유사 상품과 구분하는지', '원문 명칭 보존·대괄호 주제 위치 변경. 검수 없는 상품 별칭은 생성하지 않음'),
    ('HARD_NEGATIVE', '가까운 경쟁 FAQ보다 정답을 우선하는지', '별도 어휘 유사 후보 목록만 제공. 진짜 오답 여부는 검수 전'),
    ('KR_EN_EQUIVALENT', '동일 의미의 한영 일반명·음역이 같은 FAQ를 찾는지', '대응어가 실제 존재하는 질문에만 적용. 전 문장 번역 아님'),
    ('NEGATION', '부정 표현을 무시하고 반대 정책을 찾는지', '원문의 부정 보존. 긍정↔부정 대조 질문은 별도 GT 검수가 필요'),
    ('CONDITION_SCOPE', '가입자·지역·상황·예외 조건을 놓치는지', '원문의 조건을 유지하거나 조건 확인을 강조. 범위 반전 대조는 미생성'),
    ('NUMERIC_CONDITION', '값·단위·기간·경계 조건을 혼동하는지', '숫자 및 연산자 보존·숫자/단위 공백·천단위 쉼표 변형'),
    ('COMPARISON', '원문 비교 대상과 기준을 유지하는지', '실제 비교 질문에서만 태깅. 새 우위·차이 사실은 추정하지 않음'),
    ('MULTI_INTENT', '두 FAQ 요구 중 하나만 찾는지', '각각 실제 질문 두 개를 결합. 두 FAQ를 모두 필요한 근거로 지정'),
    ('AMBIGUOUS_QUERY', '정보 부족 상태에서 정답을 과도하게 확정하는지', '추가 검토 대상. 모든 주 질문은 브랜드를 명시'),
    ('TEMPORAL_VERSION', '정책 버전을 혼동하는지', '단일 스냅샷·업데이트 시각만으로 버전 GT를 만들지 않음'),
    ('PARAPHRASE', '말투·문장·질문 표현 차이로 정답을 놓치는지', '보수적 문법 변형. 요청 문구만 바뀐 행은 change_trace로 구분'),
    ('LONG_QUERY', '배경과 질문 위치 때문에 핵심 요구를 놓치는지', '핵심 앞/뒤 두 슬롯. 배경은 서비스 조건·별도 요청이 아님을 명시'),
    ('TYPO', '해석 가능한 표기 오류로 정답을 놓치는지', '조사 사이 띄어쓰기·작은 오타. 숫자·비교 연산자·상품 라벨 보존'),
    ('NO_FAQ', '근거 없는 요청을 잘못 채택하는지', '코퍼스 부재 검수 없이는 자동 라벨링하지 않음. 임계값 fit 전 별도 음성셋 필요'),
    ('DOCUMENT_ATTACK', '문서 속 지시가 Context/생성에 영향을 주는지', '실행 시 별도 공격 문서 fixture 필요. 본 30,000행에는 공격 지시를 넣지 않음'),
]


def main():
    raw, canonical, corpus, review = prepare_source()
    eligible = [r for r in corpus if r['seed_eligible']]
    strata = defaultdict(list)
    for r in eligible:
        strata[(r['brand'], r['category'], r['language'])].append(r)
    quota = quotas(strata, 3000)
    selected = []
    distribution = []
    for key, rs in sorted(strata.items()):
        ranked = sorted(rs, key=lambda r: digest(VERSION + r['faq_id']))
        selected.extend(ranked[:quota[key]])
        distribution.append({'brand': key[0], 'category': key[1], 'language': key[2], 'eligible_pairs': len(rs), 'selected_pairs': quota[key], 'variation_rows': quota[key] * 10, 'proportional_target': 3000 * len(rs) / len(eligible), 'rounding_error_pairs': quota[key] - 3000 * len(rs) / len(eligible)})
    selected.sort(key=lambda r: (r['brand'], r['category'], r['language'], r['faq_id']))
    partners, splits = pair_sources(selected)
    rows, baselines, source_pairs, model_inputs = [], [], [], []
    purpose_counts, primary_counts, operations = Counter(), Counter(), Counter()
    for i, faq in enumerate(selected):
        parent = f'P{i + 1:04d}'
        # Brand is legitimate user scope, held constant across all variants.
        # No FAQ ID, expected answer, or source URL enters the retrieval query.
        added_scope = scope_for(faq)
        prefix = faq['brand'] + (' / ' + added_scope if added_scope else '') + ': '
        baseline_id = parent + '-BASE'
        baselines.append({'case_id': baseline_id, 'parent_case_id': parent, 'query': prefix + faq['question'], 'faq_id': faq['faq_id'], 'purpose': 'paired_reference_only', 'count_in_30000': False, 'added_user_scope': added_scope})
        source_pairs.append({**faq, 'parent_case_id': parent, 'split_group_id': splits[faq['faq_id']], 'split': 'unassigned'})
        seen = set()
        for slot, body, primary, trace, supports in variants(faq, partners[faq['faq_id']], i):
            # Surface collisions are possible when a short title has no grammar
            # rewrite. Resolve with an honest request frame, never an opaque ID.
            if norm(body) in seen:
                body = ('Please give an explanation of this question: ' if faq['language'] == 'en' else '이 질문에 대한 설명을 부탁드립니다. ') + body
                trace.append({'operation': 'request_frame_collision_resolution', 'deep_paraphrase': False})
            assert norm(body) not in seen
            seen.add(norm(body))
            query = prefix + body
            tags = sorted(set(tags_for(faq['question']) + [primary]))
            if slot == 'V10':
                tags = sorted(set(tags + tags_for(supports[1]['question'])))
            terms = [t for t in trace if t['operation'] == 'equivalent_term']
            requirements = [{'fact_id': 'QA-' + f['faq_id'], 'fact_scope': 'source_question_answer_bundle', 'request': f['question'], 'acceptable_faq_ids': [f['faq_id']], 'acceptable_raw_faq_ids': f['source_faq_ids'], 'evidence_answer_sha256': digest(f['answer'])} for f in supports]
            row = {
                'case_id': parent + '-' + slot, 'parent_case_id': parent, 'variant_slot': slot,
                'brand': faq['brand'], 'category': faq['category'], 'source_language': faq['language'],
                'query': query, 'query_body': body, 'primary_variation': primary,
                'scope_prefix': prefix, 'added_user_scope': added_scope,
                'evaluation_family': ('multi_intent' if slot == 'V10' else 'condition' if primary in {'EXACT_ENTITY', 'KR_EN_EQUIVALENT', 'TYPO', 'NUMERIC_CONDITION', 'CONDITION_SCOPE'} else 'core'),
                'validation_tags': tags, 'change_trace': trace,
                'language_pairs': [{'ko': t['ko'], 'en': t['en']} for t in terms],
                'language_pair_reference': baseline_id if terms else None,
                'retrieval_status': 'FAQ_EXISTS', 'generation_status': 'NOT_EVALUATED',
                'required_facts': requirements, 'gt_granularity': 'FAQ_SUPPORT_BUNDLE',
                'atomic_fact_review_status': 'NOT_ANNOTATED',
                'source_faq_ids': [f['faq_id'] for f in supports],
                'seed_faq_id': faq['faq_id'], 'split_group_id': splits[faq['faq_id']], 'split': 'unassigned',
                'review_status': 'NEEDS_INDEPENDENT_SEMANTIC_AND_GT_REVIEW',
                'authoring_method': VERSION, 'deep_paraphrase': any(t['operation'] == 'grammar_rewrite' for t in trace),
            }
            rows.append(row)
            model_inputs.append({'case_id': row['case_id'], 'query': query, 'brand': faq['brand'], 'evaluation_family': row['evaluation_family']})
            purpose_counts.update(tags)
            primary_counts.update([primary])
            operations.update(t['operation'] for t in trace)
    assert len(selected) == 3000 and len(rows) == 30000
    lines('source_pairs.jsonl', source_pairs)
    lines('variations.jsonl', rows)
    lines('model_inputs.jsonl', model_inputs)
    lines('paired_baselines.jsonl', baselines)
    dump('strata_distribution.json', distribution)
    dump('term_lexicon.json', [{'ko': ko, 'en': en, 'scope': 'generic term/spelling; exclude quoted product labels', 'review_status': 'curated_rules_not_independently_reviewed'} for ko, en in LEXICON])
    catalog = [{'tag': tag, 'validation_purpose': purpose, 'scope_and_limit': limit, 'primary_rows': primary_counts[tag], 'tagged_rows': purpose_counts[tag]} for tag, purpose, limit in PURPOSES]
    dump('purpose_catalog.json', catalog)
    brand_dist = []
    for brand in sorted({r['brand'] for r in raw}):
        brand_dist.append({'brand': brand, 'raw_records': sum(r['brand'] == brand for r in raw), 'eligible_pairs': sum(r['brand'] == brand for r in eligible), 'selected_pairs': sum(r['brand'] == brand for r in selected), 'variation_rows': sum(r['brand'] == brand for r in selected) * 10})
    dump('brand_distribution.json', brand_dist)
    manifest = {
        'dataset_version': VERSION, 'source_path': 'source/faqs.jsonl', 'source_sha256': SOURCE_SHA,
        'source_records': len(raw), 'answer_captured_records': sum(bool(norm(r.get('answer'))) for r in raw),
        'canonical_pairs': len(canonical), 'duplicate_records_collapsed': sum(bool(norm(r.get('answer'))) for r in raw) - len(canonical),
        'corpus_faqs': len(corpus), 'eligible_seed_pairs': len(eligible), 'source_review_records': len(review),
        'selected_pairs': len(selected), 'variations_per_pair': 10, 'variation_rows': len(rows),
        'unique_query_strings': len({r['query'] for r in rows}), 'eligible_strata': len(strata),
        'represented_strata': len({(r['brand'], r['category'], r['language']) for r in selected}),
        'selection_method': 'all-strata minimum one + bounded proportional largest remainder + SHA256 within stratum',
        'selection_seed': VERSION, 'source_languages': dict(Counter(r['language'] for r in selected)),
        'brand_distribution': brand_dist, 'primary_variation_counts': dict(primary_counts),
        'validation_tag_counts': dict(purpose_counts), 'change_operation_counts': dict(operations),
        'kr_en_parents': len({r['parent_case_id'] for r in rows if r['language_pairs']}),
        'multi_intent_rows': sum(r['variant_slot'] == 'V10' for r in rows),
        'split_groups': len(set(splits.values())), 'split_assignment': 'unassigned',
        'calibration_readiness': {'answerable_retrieval_comparison': 'silver_labels_require_review', 'no_faq_threshold_fit': False, 'atomic_fact_recall': False, 'holdout': False},
        'old_data_imported': False, 'serving_benchmark_executed': False, 'calibration_resumed': False,
        'notes': [
            'All 30000 cases are source-backed answerable candidates. NO_FAQ absence is not inferred automatically.',
            'The ten slots are variants, not ten independent observations or ten repeated executions.',
            'Same-brand identical Q+A aliases are alternative acceptable IDs, not multiple required facts.',
            'Different answers to the same brand question remain flagged corpus competitors but are excluded as seeds.',
            'FAQ_SUPPORT_BUNDLE supports FAQ/source coverage metrics, not audited atomic FactRecall.',
            'Holdout must be split by split_group_id because the multi-intent slot connects paired parents.',
            'This is a frozen source snapshot; timestamps do not establish current policies or versioned policy GT.',
        ],
    }
    dump('manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ['selected_pairs', 'variation_rows', 'unique_query_strings', 'corpus_faqs', 'eligible_seed_pairs', 'eligible_strata', 'kr_en_parents', 'primary_variation_counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
