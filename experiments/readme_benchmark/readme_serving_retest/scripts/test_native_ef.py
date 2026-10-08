import unittest
import numpy as np
from native_ef import sparse_matrix, inverse_ranks, rrf, maxsim_batch

class NativeContracts(unittest.TestCase):
    def test_sparse_is_token_id_weight_dot_product(self):
        q=sparse_matrix([{'2':2.,'4':3.}],8)
        d=sparse_matrix([{'2':5.,'3':11.},{'4':7.}],8)
        np.testing.assert_array_equal((q@d.T).toarray(),[[10.,21.]])

    def test_rrf_uses_ranks_not_raw_score_scale(self):
        a=np.asarray([[.9,.4,.1]]);b=np.asarray([[2.,5.,3.]])
        np.testing.assert_allclose(rrf(a,b),rrf(a*100,b*1000))
        np.testing.assert_array_equal(inverse_ranks(np.asarray([[1.,1.,.5]])),[[1,2,3]])

    def test_padded_negative_tokens_and_query_mean_are_correct(self):
        # Padding must not become an extra zero-score document token; padded query tokens must not affect the mean.
        q=[np.asarray([[1.,0.]],dtype=np.float32),np.asarray([[1.,0.],[0.,1.]],dtype=np.float32)]
        d=[np.asarray([[-1.,-1.]],dtype=np.float32),np.asarray([[.5,0.],[0.,1.]],dtype=np.float32)]
        np.testing.assert_allclose(maxsim_batch(q,d,'cpu'),[[-1.,.5],[-1.,.75]])
        # Query→document MaxSim is directional.
        forward=maxsim_batch(q[0:1],d[1:2],'cpu')[0,0]
        backward=maxsim_batch(d[1:2],q[0:1],'cpu')[0,0]
        self.assertNotEqual(forward,backward)

if __name__=='__main__': unittest.main()
