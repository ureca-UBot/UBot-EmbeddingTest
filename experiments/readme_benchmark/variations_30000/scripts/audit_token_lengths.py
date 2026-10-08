"""Measure real BGE-M3 input lengths without running retrieval or changing data."""
import json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from build_variations import read_jsonl,write_json,write_jsonl

ROOT=Path(__file__).resolve().parents[1]
rows=read_jsonl(ROOT/'data'/'variations.jsonl')
tokenizer=AutoTokenizer.from_pretrained('/models/hf/serving-bge-m3-512',local_files_only=True)
lengths=[]
for start in range(0,len(rows),128):
    chunk=rows[start:start+128]
    encoded=tokenizer([r['model_query'] for r in chunk],truncation=False,add_special_tokens=True,return_offsets_mapping=True)
    for row,ids,offsets in zip(chunk,encoded['input_ids'],encoded['offset_mapping']):
        core_start=row['model_query'].rfind(row['core_question'])
        core_end=core_start+len(row['core_question'])
        kept_end=max((end for begin,end in offsets[:511]),default=0)
        lengths.append({'case_id':row['case_id'],'primary':row['primary_variation'],'tokens':len(ids),
                        'over_512':len(ids)>512,'core_complete_in_first_512':core_start>=0 and core_end<=kept_end})
    if start%5120==0:print(json.dumps({'measured':min(start+128,len(rows)),'total':len(rows)}),flush=True)
write_jsonl(ROOT/'outputs'/'token_lengths.jsonl',lengths)
distribution={}
for primary in dict.fromkeys(r['primary'] for r in lengths):
    group=[r for r in lengths if r['primary']==primary]
    distribution[primary]={'rows':len(group),'max_tokens':max(r['tokens'] for r in group),
                           'over_512':sum(r['over_512'] for r in group),'core_missing_at_512':sum(not r['core_complete_in_first_512'] for r in group)}
summary={'tokenizer':'BAAI/bge-m3 local frozen tokenizer','max_length_under_test':512,
         'rows':len(lengths),'over_512_rows':sum(r['over_512'] for r in lengths),
         'max_tokens':max(r['tokens'] for r in lengths),'by_primary':distribution,
         'actual_serving_truncation_test_executed':False}
write_json(ROOT/'outputs'/'token_length_summary.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
