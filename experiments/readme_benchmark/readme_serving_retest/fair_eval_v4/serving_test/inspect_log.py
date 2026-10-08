"""Print a single query's gzip candidate log without loading the whole file."""
import argparse,gzip,json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--dataset',required=True,choices=['existing','kt','skt','lgu'])
p.add_argument('--engine',required=True,choices=['ollama','vllm'])
p.add_argument('--structure',default='D-bge-question_answer')
p.add_argument('--question',required=True)
a=p.parse_args()
if not all(c.isalnum() or c in '-_' for c in a.structure):p.error('Invalid structure name')
path=Path(__file__).resolve().parent/'outputs/run-v4'/a.dataset/a.engine/a.structure/'candidate_log.jsonl.gz'
found=[]
with gzip.open(path,'rt',encoding='utf-8') as stream:
 for line in stream:
  row=json.loads(line)
  if row['question_id']==a.question:found.append(row)
if not found:p.error('Question not found in this dataset/structure')
print(json.dumps(sorted(found,key=lambda x:x['rerank_rank'] or x['retrieval_rank']),ensure_ascii=False,indent=2))
