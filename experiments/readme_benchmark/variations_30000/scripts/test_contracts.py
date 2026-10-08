import unittest
from build_variations import numeric_signature,replace_term,paraphrase,endings,bilingual,read_jsonl,SOURCE,make_candidates,distinct_candidate

class AuthoringContracts(unittest.TestCase):
    def test_amount_is_not_assumed_to_be_money(self):
        q,_=endings('남은 데이터는 얼마인가요?',2)
        self.assertNotIn('금액',q)

    def test_noun_postposition(self):
        q,_=replace_term('휴대폰이 없으면 휴대폰을 대신 인증할 수 있나요?','휴대폰','휴대전화')
        self.assertIn('휴대전화가',q)
        self.assertIn('휴대전화를',q)

    def test_vowel_copula(self):
        self.assertEqual(endings('명의자 만 60세입니다.',2)[0],'명의자 만 60세예요.')

    def test_api_enums_and_temporal_anchor(self):
        original='2026-10-01T00:00:00+09:00 기준이며 API 결과는 accepted이고 completed는 false예요. 완료되나요?'
        q,_=paraphrase(original,2)
        for value in ['2026-10-01T00:00:00+09:00','accepted','completed','false']:self.assertIn(value,q)
        self.assertEqual(numeric_signature(q),numeric_signature(original))

    def test_different_meaning_compounds_are_not_partially_translated(self):
        q,changes,pairs=bilingual('전화번호가 바뀌면 인증은 어떻게 하나요?',0)
        self.assertNotIn('phone call번호',q)

    def test_every_repeat_input_has_identical_variations(self):
        cases=read_jsonl(SOURCE/'cases.jsonl');raw=read_jsonl(SOURCE/'source_rows.jsonl')
        values=[]
        for c,r in zip(cases,raw):
            if c['case_id'].startswith('RT-0001-'):
                used=set();variants=[]
                for candidate in make_candidates(c,r):
                    v=distinct_candidate(candidate,c,used);variants.append(v['query']);used.add(v['query'])
                values.append(variants)
        self.assertEqual(len(values),10)
        self.assertTrue(all(v==values[0] for v in values))

if __name__=='__main__':unittest.main()
