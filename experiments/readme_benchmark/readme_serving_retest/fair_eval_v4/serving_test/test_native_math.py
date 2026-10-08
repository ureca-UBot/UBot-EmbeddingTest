import unittest
import numpy as np
from run_native import maxsim_batch,sparse_matrix

class NativeMathTests(unittest.TestCase):
 def test_maxsim_padding_does_not_win_negative_scores(self):
  q=[np.array([[1.,0.],[0.,1.]],np.float32),np.array([[1.,1.]],np.float32)]
  d=[np.array([[-1.,-1.]],np.float32),np.array([[-2.,-3.],[-4.,-1.]],np.float32)]
  result=maxsim_batch(q,d,device='cpu')
  expected=np.array([[(a@b.T).max(1).mean() for b in d] for a in q])
  np.testing.assert_allclose(result,expected,atol=1e-6)
 def test_sparse_native_weights_use_dot_product(self):
  q=sparse_matrix([{'1':2.,'4':3.}],6);d=sparse_matrix([{'1':4.,'4':2.},{'2':7.}],6)
  np.testing.assert_allclose((q@d.T).toarray(),[[14.,0.]])
 def test_installed_reranker_pair_templates_and_padding(self):
  from transformers import AutoTokenizer
  from common_v4 import model_path
  from run_rerank import pair_ids,padded_batch
  for name in ['BAAI/bge-reranker-v2-m3','jinaai/jina-reranker-v2-base-multilingual']:
   tok=AutoTokenizer.from_pretrained(model_path(name));texts=[['한국어 질문','상세한 답변입니다.'],['Q','A']]
   expected=tok(texts,padding=True,truncation=False,return_tensors='pt',return_token_type_ids=False)
   pairs=[pair_ids(tok,tok.encode(q,add_special_tokens=False),tok.encode(d,add_special_tokens=False)) for q,d in texts]
   actual=padded_batch(tok,pairs)
   for k in actual:np.testing.assert_array_equal(actual[k].numpy(),expected[k].numpy())

if __name__=='__main__':unittest.main(verbosity=2)
