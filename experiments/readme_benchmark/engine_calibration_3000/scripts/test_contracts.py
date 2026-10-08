import unittest
import numpy as np
from common import metrics,pools,calibrate_policy,inputs

class Contracts(unittest.TestCase):
    def test_partial_source_retrieval_is_not_complete(self):
        result=metrics(['FAQ-001','FAQ-999'],['FAQ-001','FAQ-002'],3)
        self.assertEqual(result['hit'],1);self.assertEqual(result['coverage'],.5);self.assertEqual(result['all_sources'],0)
        self.assertIsNone(metrics(['FAQ-001'],[],3)['hit'])

    def test_rank_boundary_excludes_fourth_source(self):
        result=metrics(['A','B','C','D'],['D'],3)
        self.assertEqual(result['hit'],0)

    def test_union_keeps_budget_and_removes_duplicates(self):
        a=np.array([[.9,.8,.7,.1]]);b=np.array([[.85,.2,.75,.6]])
        self.assertEqual(pools(a,b,2),[[0,1]])

    def test_api_and_partial_do_not_become_no_faq_negatives(self):
        cases=[{'cohort':'general','no_faq_truth':False},{'cohort':'no_faq','no_faq_truth':True},
               {'cohort':'api','no_faq_truth':False},{'cohort':'partial','no_faq_truth':False}]
        rows=[{'top_scores':[5.,4.]},{'top_scores':[-5.,-6.]},{'top_scores':[-99.,-100.]},{'top_scores':[-99.,-100.]}]
        policy=calibrate_policy(rows,cases)
        self.assertEqual(policy['rows'],2);self.assertEqual(policy['no_faq_f1'],1.)

    def test_selected_source_is_preserved_and_repeat_inputs_are_equal(self):
        cases,_=inputs();groups={}
        for c in cases:
            if c['type']=='RT':groups.setdefault(c['case_id'].rsplit('-R',1)[0],[]).append(c)
        self.assertEqual(len(groups),20)
        for rows in groups.values():
            self.assertEqual(len(rows),10)
            self.assertEqual(len({c['model_query'] for c in rows}),1)

if __name__=='__main__':unittest.main()
