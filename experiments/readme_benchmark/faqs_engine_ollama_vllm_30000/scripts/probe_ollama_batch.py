"""Diagnose full-document physical batch capacity without changing benchmark caches."""
import json
import sys
import urllib.request
import urllib.error
import numpy as np
from common import DATA,OUT,doc_text,lines,save

docs=lines(DATA/'corpus.jsonl')
cases=lines(DATA/'cases.jsonl')
text=doc_text(docs[2332],'question_answer')
url='http://127.0.0.1:11435/api/embed'
records=[]
for batch in (2048,4096):
    payload={'model':'bge-m3','input':[text],'truncate':False,'keep_alive':'30m','options':{'num_ctx':8192,'num_batch':batch}}
    req=urllib.request.Request(url,json.dumps(payload).encode(),{'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=300) as r:body=json.load(r)
        v=np.asarray(body['embeddings'],np.float32)
        assert v.shape==(1,1024) and np.isfinite(v).all() and np.linalg.norm(v)>0
        record={'num_batch':batch,'status':200,'processed_tokens':body['prompt_eval_count'],'vector_dimensions':1024,'finite':True}
        assert body['prompt_eval_count']==3570
    except urllib.error.HTTPError as exc:
        record={'num_batch':batch,'status':exc.code,'error':exc.read().decode()}
        assert batch==2048
    records.append(record)
    print(json.dumps(record),flush=True)
assert records[0]['status']==400 and records[1]['status']==200
save(OUT/'engines/ollama/physical_batch_diagnostic.json',{'faq_id':docs[2332]['faq_id'],'full_input_tokens':3570,'truncation':False,'requests':records,'official_source':'https://github.com/ollama/ollama/blob/v0.35.0/server/routes.go'})
