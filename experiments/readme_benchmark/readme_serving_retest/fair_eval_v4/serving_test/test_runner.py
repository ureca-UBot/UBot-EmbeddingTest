import tempfile,unittest
from pathlib import Path
import numpy as np
import common_v4 as c
from run_rerank import spans
from run_v4 import request

class Tests(unittest.TestCase):
 def test_union_and_budget_are_distinct(self):
  a=np.array([[.9,.8,.2,.1]],np.float32);b=np.array([[.1,.2,.9,.8]],np.float32)
  u=c.pool(a,b,2)[0];self.assertEqual(len(u),4);self.assertEqual(len(u[:2]),2)
 def test_windows_cover_every_token(self):
  for n in (0,75,980,981,3548):
   windows=spans(n,980);covered=set()
   for i,(start,end) in enumerate(windows):
    self.assertLessEqual(end-start,980);covered.update(range(start,end))
    if i:self.assertEqual(windows[i-1][1]-start,128)
   self.assertEqual(covered,set(range(n)))
 def test_http_normalization_and_payload(self):
  class Result:
   def raise_for_status(self):pass
   def json(self):return {'embeddings':[[2.]+[0.]*1023]}
  class Session:
   def post(self,url,json,timeout):self.body=json;return Result()
  s=Session();v,_,_=request(s,'ollama','http://test',['hello']);self.assertEqual(float(np.linalg.norm(v[0])),1)
  self.assertEqual(s.body['input'],['hello']);self.assertFalse(s.body['truncate'])
  request(s,'ollama','http://test',['long document'],max_tokens=3548);self.assertEqual(s.body['options']['num_batch'],4096)
 def test_rank_delta_and_multiple_gold_log(self):
  old=c.OUT
  with tempfile.TemporaryDirectory() as tmp:
   c.OUT=Path(tmp)
   qs=[dict(question_id='q',split='calibration',query='hello',model_query='hello',answer_faq_ids=['a','b'],required_facts=[dict(fact_id='f',acceptable_faq_ids=['a','b'])])]
   docs=[{'faq_id':'a'},{'faq_id':'b'},{'faq_id':'c'}]
   a=np.array([[.5,.9,.8]],np.float32);b=np.array([[.6,.95,.7]],np.float32);rr=np.array([[3.,1.,2.]],np.float32)
   win={(0,j):dict(count=1,index=0,text='document') for j in range(3)}
   c.export_log('kt','ollama','D-test',qs,docs,[[1,2,0]],a,b,rr,win)
   rows=c.lines(c.base('kt','ollama')/'D-test/candidate_log.jsonl.gz')
   self.assertEqual([r['rank_delta'] for r in rows],[-2,0,2])
   self.assertEqual(rows[0]['expected_faq_id'],['a','b']);self.assertEqual(rows[2]['retrieval_a_rank'],3)
  c.OUT=old
 def test_wrong_carrier_rejected_before_inference(self):
  d,qs,docs=c.load('kt');d.corpus.append(dict(dataset_id='skt',faq_id='bad',brand='SKT'));calls=[]
  with self.assertRaises(ValueError):d.embed_corpus(lambda t:calls.append(t),engine='test',model='bge',representation='dense',view='question',settings={})
  self.assertEqual(calls,[])

if __name__=='__main__':unittest.main(verbosity=2)
