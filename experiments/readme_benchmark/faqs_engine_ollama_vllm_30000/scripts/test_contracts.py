import unittest
from collections import Counter
import numpy as np
from common import metrics,pools,calibrate_policy,inputs,MAIN_N,N
from run_experiment import merge_window_scores,rerank_documents

class Contracts(unittest.TestCase):
    def test_compound_query_requires_both_sources(self):
        self.assertEqual(metrics(['A','X'],['A','B'],3)['coverage'],.5)
        self.assertEqual(metrics(['A','X'],['A','B'],3)['all_sources'],0)
        self.assertEqual(metrics(['A','B'],['A','B'],3)['all_sources'],1)
    def test_rank_boundary(self):
        self.assertEqual(metrics(['A','B','C','D'],['D'],3)['hit'],0)
    def test_dual_budget_and_stable_ties(self):
        self.assertEqual(pools(np.array([[.9,.8,.7,.1]]),np.array([[.85,.2,.75,.6]]),2),[[0,1]])
    def test_max_window_aggregation_keeps_largest_even_with_duplicate_batch_indices(self):
        values=np.full((1,2),-1e20,dtype=np.float32)
        merge_window_scores(values,np.array([0,0,0]),np.array([0,0,1]),np.array([2.,3.,-4.],dtype=np.float32))
        np.testing.assert_array_equal(values,np.array([[3.,-4.]],dtype=np.float32))
    def test_positives_only_never_yield_a_fitted_rejection_threshold(self):
        cases=[{'evaluation_role':'main','no_faq_truth':False}]
        policy=calibrate_policy([{'top_scores':[.8,.7]}],cases)
        self.assertIsNone(policy['threshold']);self.assertIsNone(policy['no_faq_f1'])
    def test_winning_window_follows_maximum_and_stable_ties_across_batches(self):
        scores=np.full((1,2),-1e20,dtype=np.float32)
        winners=np.full((1,2),np.iinfo(np.int16).max,dtype=np.int16)
        merge_window_scores(scores,np.array([0,0,0]),np.array([0,0,1]),np.array([2.,3.,-4.],dtype=np.float32),winners,np.array([4,5,7]))
        np.testing.assert_array_equal(winners,[[5,7]])
        merge_window_scores(scores,np.array([0,0,0]),np.array([0,0,1]),np.array([3.,1.,-3.],dtype=np.float32),winners,np.array([3,2,8]))
        np.testing.assert_array_equal(winners,[[3,8]])
        np.testing.assert_array_equal(scores,[[3.,-3.]])
    def test_frozen_new_inputs_and_reference_separation(self):
        cases,docs=inputs();main=[c for c in cases if c['evaluation_role']=='main']
        self.assertEqual(len(cases),N);self.assertEqual(len(main),MAIN_N);self.assertEqual(len(docs),3246)
        self.assertEqual(set(Counter(c['parent_case_id'] for c in main).values()),{10})
        self.assertTrue(all(c['model_query']==c['query'] for c in cases))
        self.assertTrue(all(not c['no_faq_truth'] for c in cases))
    def test_windows_cover_every_candidate_faq(self):
        _,docs=inputs();texts,mapping=rerank_documents(docs,'question_answer')
        self.assertEqual(set(mapping),set(range(len(docs))))
        self.assertTrue(all(mapping[j] for j in range(len(docs))))
        self.assertEqual(sum(map(len,mapping.values())),len(texts))
if __name__=='__main__':unittest.main()
