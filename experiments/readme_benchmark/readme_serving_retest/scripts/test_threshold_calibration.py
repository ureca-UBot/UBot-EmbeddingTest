import unittest
from fractions import Fraction
import numpy as np
from calibrate_thresholds import accept,classification,fit,split
from common import inputs
from control_gt import revised_cases

class ThresholdTests(unittest.TestCase):
    def test_exact_search_matches_exhaustive_all_sample_boundaries(self):
        rng=np.random.default_rng(20261007)
        for _ in range(40):
            scores=rng.integers(-3,4,12).astype(float)/4
            margins=rng.integers(0,5,12).astype(float)/10
            labels=np.ones(12,dtype=bool);labels[rng.choice(12,3,replace=False)]=False
            result=fit(scores,margins,labels)
            actual=result['metrics']['macro_f1']
            exhaustive=max(classification(labels,accept(scores,margins,t,m))['macro_f1']
                for t in [None]+[np.nextafter(x,np.inf) for x in np.unique(scores)]
                for m in [None]+[np.nextafter(x,np.inf) for x in np.unique(margins)])
            self.assertAlmostEqual(actual,exhaustive,places=14)

    def test_joint_gate_can_beat_score_and_margin_only(self):
        scores=np.asarray([.7,.8,.6,.9]);margins=np.asarray([.3,.2,.4,.1]);labels=np.asarray([1,1,0,0],dtype=bool)
        result=fit(scores,margins,labels)
        self.assertEqual(result['metrics']['macro_f1'],1.)
        self.assertLess(fit(scores,margins,labels,'score_only')['metrics']['macro_f1'],1.)
        self.assertLess(fit(scores,margins,labels,'margin_only')['metrics']['macro_f1'],1.)

    def test_float32_boundary_is_not_rounded_back(self):
        values=np.asarray([.7,.8],dtype=np.float32)
        cutoff=float(np.nextafter(float(values[0]),np.inf))
        np.testing.assert_array_equal(accept(values,[0.,0.],cutoff,None),[False,True])
        np.testing.assert_array_equal(accept(values,[0.,0.],float(values[0]),None),[True,True])

    def test_confusion_counts_and_macro_f1(self):
        result=classification([1,1,1,0,0],[1,1,0,0,1])
        self.assertEqual((result['no_faq_tp'],result['no_faq_fn'],result['faq_false_rejections'],result['faq_accepted']),(1,1,1,2))
        self.assertAlmostEqual(result['macro_f1'],float((Fraction(1,2)+Fraction(2,3))/2))

    def test_grouped_split_keeps_every_parent_and_alias_family_together(self):
        cases,docs=inputs();cases=revised_cases(cases,docs,write_outputs=False)
        folds,assignments,report=split(cases)
        self.assertEqual(sum(map(len,folds)),5002)
        self.assertEqual(len(set(folds[0])&set(folds[1])),0)
        parent={}
        for row in assignments:
            if row['fold'] is None:continue
            if row['parent_case_id'] in parent:self.assertEqual(parent[row['parent_case_id']],row['fold'])
            parent[row['parent_case_id']]=row['fold']
        self.assertEqual(report['overlapping_acceptable_FAQ_families'],0)
        self.assertEqual(len(report['fold_no_faq_ids'][0]),1)
        self.assertEqual(len(report['fold_no_faq_ids'][1]),1)

if __name__=='__main__':unittest.main()
