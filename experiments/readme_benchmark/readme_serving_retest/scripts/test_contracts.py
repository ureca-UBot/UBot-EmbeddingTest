import tempfile
import unittest
from collections import Counter
from pathlib import Path
import numpy as np
import common
from common import doc_text, pools, metrics, inputs, FIELDS, export_log, read, save, lines

class ContractTests(unittest.TestCase):
    def test_union_keeps_disjoint_candidates(self):
        a=np.asarray([[4.,3.,2.,1.]])
        b=np.asarray([[1.,2.,3.,4.]])
        self.assertEqual(pools(a,b,2),[[0,3,1,2]])
        self.assertEqual(len(pools(a,b,2)[0]),4)

    def test_one_faq_can_satisfy_two_facts(self):
        facts=[{'acceptable_faq_ids':['FAQ-A','FAQ-C']},{'acceptable_faq_ids':['FAQ-B','FAQ-C']}]
        self.assertEqual(metrics(['FAQ-C'],facts)['fact_recall'],1)
        self.assertEqual(metrics(['FAQ-C'],facts)['all_facts_hit'],1)
        self.assertEqual(metrics(['FAQ-A'],facts)['fact_recall'],.5)
        self.assertIsNone(metrics(['FAQ-A'],[]))

    def test_only_requested_document_view(self):
        d={'question':'질문 전문','answer':'답변 전문','category':'절대 넣지 않는 메타데이터'}
        self.assertEqual(doc_text(d,'question'),'질문 전문')
        self.assertEqual(doc_text(d,'question_answer'),'질문 전문\n\n답변 전문')

    def test_actual_export_entire_corpus_ranks_and_raw_logits(self):
        old=common.OUT
        try:
            with tempfile.TemporaryDirectory() as tmp:
                common.OUT=Path(tmp)
                cache=common.OUT/'engines/mock/cache';cache.mkdir(parents=True)
                np.save(cache/'A-scores.npy',np.asarray([[.9,.8,.7,.6]],dtype=np.float32))
                np.save(cache/'B-scores.npy',np.asarray([[.1,.2,.3,.4]],dtype=np.float32))
                # C order [0,3,1,2], raw logits put 3 first and 1 before equal-scored 2.
                scores=np.asarray([[-4.,-2.,-2.,-1.]],dtype=np.float32)
                docs=[{'faq_id':str(j),'question':f'Q{j}','answer':f'A{j}'} for j in range(4)]
                cases=[{'case_id':'x','query':'raw','model_query':'raw','primary_validation_purpose':'PARAPHRASE',
                        'execution_group':'core','required_facts':[{'acceptable_faq_ids':['3']}]}]
                export_log('mock','D-test','question_answer',cases,docs,[[0,3,1,2]],scores,{})
                records=lines(common.OUT/'engines/mock/D-test/candidate_log.jsonl')
                record=next(r for r in records if r['faq_id']=='3')
                self.assertEqual((record['retrieval_rank'],record['rerank_rank'],record['rank_delta']),(2,1,1))
                self.assertEqual((record['retrieval_a_rank'],record['retrieval_b_rank']),(4,1))
                self.assertEqual((record['window_count'],record['winning_window_index']),(1,0))
                self.assertEqual(record['winning_window_text'],'Q3\n\nA3')
                self.assertEqual(record['winning_window_score'],-1.)
                self.assertEqual(record['expected_faq_id'],['3'])
                self.assertTrue(all(field in record for field in FIELDS))
                self.assertEqual(next(r for r in records if r['faq_id']=='1')['rerank_rank'],2)
        finally: common.OUT=old

    def test_frozen_new_input_and_group_separation(self):
        cases,docs=inputs()
        main=[c for c in cases if c['evaluation_role']=='main']
        parents=Counter(c['parent_case_id'] for c in main)
        self.assertEqual(len(main),5000)
        self.assertEqual(len(parents),500)
        self.assertEqual(set(parents.values()),{10})
        self.assertEqual(len({c['case_id'] for c in cases}),len(cases))
        self.assertTrue(all(c['execution_group']=='no_faq' for c in cases if c['retrieval_status']=='NO_FAQ'))
        self.assertTrue(any(c['execution_group']=='context' and c['query']!=c['model_query'] for c in cases))
        self.assertTrue(any(len(c['required_facts'])==2 for c in cases))

if __name__=='__main__': unittest.main()
