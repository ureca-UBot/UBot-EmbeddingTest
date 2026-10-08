"""New source-grounded synthetic queries; never import old evaluation questions."""
import random
import re
from collections import Counter, defaultdict
from common import ROOT, DATA, OUT, read, lines, save, save_lines, sha

SOURCE = ROOT.parent/'readme_purpose_questions'
PARA = [('어떻게 하나요','어떤 방법으로 진행하나요'),('어떻게 해야 하나요','어떤 조치를 하면 되나요'),
        ('어디에서','어느 곳에서'),('할 수 있나요','하는 것이 가능한가요'),('무엇인가요','어떤 의미인가요'),
        ('얼마인가요','어느 정도 금액인가요'),('확인하나요','조회하나요'),('휴대폰','핸드폰'),
        ('변경하려면','바꾸려면'),('변경하면','바꾸면'),('변경할','바꿀'),
        ('고객센터','고객 상담 센터'),('재발급','다시 발급'),('문자','문자 메시지')]
# Terminology substitutions only: numbers, conditions and predicate stay unchanged.
LANG = [('eSIM','이심'),('유심','USIM'),('와이파이','Wi-Fi'),('테더링','tethering'),
        ('로밍','roaming'),('데이터','data'),('블루투스','Bluetooth'),('SIM 카드','심 카드'),
        ('핫스팟','hotspot'),('인터넷','internet'),('고객센터','customer service center')]
TYPO = [('어떻게','어떡게'),('휴대폰','휴대퐁'),('통신사','통신싸'),('신청','신쳥'),
        ('사용','사욤'),('이용','이욤'),('요금','요굼'),('확인','화인')]
BACKGROUND = ('오늘은 책상 위를 정리하면서 오래된 영수증과 사진을 날짜순으로 나누어 두었습니다. '
              '창밖 날씨를 보다가 주말 일정도 적었고 책장에 있는 소설의 위치를 바꿨습니다. '
              '메모에는 할 일이 여러 가지 적혀 있었지만 지금 상담할 내용은 다음 질문 하나입니다. ')

def build():
    docs = lines(SOURCE/'data/corpus.jsonl')
    assert len(docs) == 1024
    save_lines(DATA/'corpus.jsonl', docs)
    catalog = {p['tag']:p for p in read(SOURCE/'data/purpose_catalog.json')}
    # Source selection is reproducible; no old item/brand/category quotas.
    chosen = random.Random(20261007).sample(docs,500)
    by_answer = defaultdict(list)
    for d in docs: by_answer[re.sub(r'\s+','',d['question']+'\n'+d['answer'])].append(d['faq_id'])
    cases = []
    for doc in sorted(chosen,key=lambda d:d['source_row']):
        q = doc['question'].strip()
        candidates = []
        def add(text,tag,change,pair=None):
            if text != q and text not in {x[0] for x in candidates}:
                candidates.append((text,tag,change,pair))
        # Changes are eligible by actual source wording, not a fixed V01-V10 grid.
        for old,new in LANG:
            if old in q:
                add(q.replace(old,new),'KR_EN_EQUIVALENT',f'{old}→{new}',q)
        for old,new in TYPO:
            if old in q:
                add(q.replace(old,new,1),'TYPO',f'one small spelling error: {old}→{new}')
                break
        if ' ' in q:
            add(q.replace(' ','',1),'TYPO','remove one space')
        for old,new in PARA:
            if old in q: add(q.replace(old,new),'PARAPHRASE',f'{old}→{new}')
        # Explicit polite/direct request forms preserve the complete original question.
        for ending,new in [('나요?','는지 알려주세요.'),('인가요?','인지 설명해 주세요.'),
                           ('됩니까?','되는지 알려주세요.'),('어요.','습니다. 해결 방법을 안내해 주세요.')]:
            if q.endswith(ending):
                stem=q[:-len(ending)]
                # -하나요 cannot mechanically become -하는지 without conjugation.
                if ending=='나요?' and stem.endswith('하'): text=stem+'는지 알려주세요.'
                elif ending=='나요?' and stem.endswith('되'): text=stem+'는지 알려주세요.'
                elif ending=='나요?': continue
                else: text=stem+new
                add(text,'PARAPHRASE','question to an explicit request')
        add(q+' '+BACKGROUND.replace('다음 질문','앞에서 한 질문'),'LONG_QUERY','irrelevant background after key question')
        add(BACKGROUND+q,'LONG_QUERY','irrelevant background before key question')
        # Additional speech registers; real query content is preserved, no answer inserted.
        speech = q.replace('하나요?','하죠?').replace('있나요?','있어요?').replace('무엇인가요?','뭐예요?').replace('얼마인가요?','얼마예요?')
        if speech != q: add(speech,'PARAPHRASE','polite conversational question')
        if q.endswith('?'):
            add('궁금한 점은 다음과 같아요: '+q,'PARAPHRASE','quoted direct question')
            add('다음 내용을 상담받고 싶습니다. '+q,'PARAPHRASE','consultation request formulation')
            add('제 질문을 확인해 주세요. '+q,'PARAPHRASE','explicit question formulation')
            add(q[:-1]+'? 관련 절차나 조건을 설명해 주세요.','PARAPHRASE','request source-supported explanation')
        else:
            add('이 문제는 어떻게 해결하죠? '+q,'PARAPHRASE','situation to solution request')
            add(q+' 원인과 확인 방법이 궁금해요.','PARAPHRASE','situation to diagnostic request')
            add('도움이 필요합니다. '+q,'PARAPHRASE','situation as a consultation request')
            add(q+' 어떻게 조치해야 할지 알려주세요.','PARAPHRASE','situation to action request')
        if len(candidates)<10:
            add(q.replace(' ','') if ' ' in q else q+' ','TYPO','spacing variation')
        if len(candidates)<10:
            add(q+' '+BACKGROUND[:80],'LONG_QUERY','shorter irrelevant trailing background')
        if len(candidates)<10:
            add('현재 상황을 말씀드릴게요. '+q+' 필요한 확인 절차를 알려주세요.','PARAPHRASE','explicit situation-based consultation')
        assert len(candidates)>=10, (doc['faq_id'],q,len(candidates))
        # Keep terminology and typo cases if eligible, then sample remaining purposes.
        prioritized=[x for x in candidates if x[1]=='KR_EN_EQUIVALENT'][:1]
        for tag in ['TYPO','LONG_QUERY']:
            eligible=[x for x in candidates if x[1]==tag]
            prioritized.extend(eligible[:2])
        rest=[x for x in candidates if x not in prioritized]
        rng=random.Random('purpose:'+doc['faq_id'])
        rng.shuffle(rest)
        selected=(prioritized+rest)[:10]
        assert len({x[0] for x in selected})==10
        fact={'fact_id':doc['faq_id']+'-SOURCE-ANSWER', 'statement':doc['answer'],
              'acceptable_faq_ids':by_answer[re.sub(r'\s+','',doc['question']+'\n'+doc['answer'])],
              'evidence':[{'faq_id':doc['faq_id'],'quote':doc['answer'],
                           'source_sheet':doc['source_sheet'],'source_cell':f"D{doc['source_row']}"}],
              'annotation_granularity':'full source answer support; not independently decomposed atomic facts'}
        for slot,(text,tag,changed,pair) in enumerate(selected,1):
            # Numeric strings are never introduced by this generator.
            assert re.findall(r'\d+(?:[.,]\d+)*',text)==re.findall(r'\d+(?:[.,]\d+)*',q)
            cases.append({'case_id':f"NEW-{doc['faq_id']}-{slot:02}",'parent_case_id':doc['faq_id'],
                'query':text,'model_query':text,'evaluation_role':'main','execution_group':'core',
                'primary_validation_purpose':tag,'validation_purpose':catalog[tag]['validation_purpose'],
                'changed_elements':[changed],'preserved_elements':{'original_intent_query':q,
                    'source_faq_id':doc['faq_id'],'numbers_and_policy_conditions':'unchanged'},
                'required_facts':[fact], 'retrieval_status':'FAQ_EXISTS','generation_expected_state':'ANSWER',
                'failure_criteria':catalog[tag]['validation_purpose'],
                'language_pair_original_query':pair,
                'evidence_review_status':'source-grounded deterministic transformation; independent semantic review pending',
                'benchmark_ready':False})
    controls=lines(SOURCE/'data/authoring_examples.jsonl')
    for c in controls:
        if c['execution_group']=='document_attack': continue
        c=dict(c)
        c.update(parent_case_id=c['case_id'],evaluation_role='purpose_control')
        c['model_query']=c.get('fixed_rewritten_query',c['query'])
        cases.append(c)
    save_lines(DATA/'cases.jsonl',cases)
    # Term-equivalence baselines are actual queries, not GT text/answers.
    pairs=[{'case_id':c['case_id'],'query_ko':c['language_pair_original_query'],'variant_query':c['model_query'],
            'required_facts':c['required_facts']} for c in cases if c.get('language_pair_original_query')]
    save_lines(DATA/'language_pair_controls.jsonl',pairs)
    save(DATA/'document_attack_fixtures.json',read(SOURCE/'data/document_attack_fixtures.json'))
    save(DATA/'purpose_catalog.json',list(catalog.values()))
    save(DATA/'manifest.json',{'main_rows':5000,'parent_faqs':500,'variants_per_parent':10,
        'purpose_control_rows':len(cases)-5000,'inference_rows':len(cases),'corpus_faqs':len(docs),
        'purpose_counts':dict(Counter(c['primary_validation_purpose'] for c in cases)),
        'execution_group_counts':dict(Counter(c['execution_group'] for c in cases)),
        'source_corpus_sha256':sha(SOURCE/'data/corpus.jsonl'),
        'source_workbook':read(SOURCE/'configs/question_set_policy.json')['knowledge_source'],
        'standard':'original requested README validation purposes plus user overrides',
        'old_test_queries_imported':0,'category_quotas':False,'fixed_ten_types_per_parent':False,
        'gt_review':'provisional synthetic serving/retrieval diagnostic; not a final accuracy benchmark',
        'unsupported_purposes':{'REAL_FAILURE':'actual service log unavailable','TEMPORAL_VERSION':'versioned policies unavailable'},
        'protocol':{'A':'raw FAQ question only','B':'question + two newlines + full answer',
                    'C':'union of A Top20 and B Top20, deduplicated, never re-pruned',
                    'D':'both rerankers × full question/full question+answer; raw logits; no truncation or windowing',
                    'threshold':'do not fit on independently unreviewed NO_FAQ controls'},
        'hashes':{name:sha(DATA/name) for name in ['corpus.jsonl','cases.jsonl','language_pair_controls.jsonl','document_attack_fixtures.json']}})
    print({'main':5000,'controls':len(cases)-5000,'language_pairs':len(pairs),
           'purpose_counts':dict(Counter(c['primary_validation_purpose'] for c in cases))})

if __name__=='__main__': build()
