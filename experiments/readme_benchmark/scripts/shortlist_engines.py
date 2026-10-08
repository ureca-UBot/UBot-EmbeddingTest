"""Shortlist measured engines using parity, errors and matched workloads."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/run-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
rows=[];workloads=set();sample_hashes=set()
for engine in ['ollama','tei','vllm']:
    p=OUT/'serving'/engine;parity=read(p/'parity.json');cells=read(p/'load_results.json');stop=read(p/'stop_verification.json')
    manifest=read(p/'manifest.json');workloads.add(manifest['workload_hash']);sample_hashes.add(tuple(parity['sample_query_hashes']))
    if len(cells)!=15 or not stop['verified_stopped']:raise ValueError('Incomplete engine run')
    eligible=parity['status']=='passed' and all(c['failures']==0 for c in cells)
    rows.append({'engine':engine,'eligible':eligible,'parity':parity['status'],'failed_attempts':sum(c['failures'] for c in cells),
                 'cells':len(cells),'single_request_p95_ms':next(c['p95_ms'] for c in cells if c['concurrency']==1 and c['texts_per_request']==1),
                 'best_texts_per_second':max(c['texts_per_sec'] for c in cells),'cold_start_seconds':read(p/'readiness.json')['cold_start_to_ready_seconds']})
if len(workloads)!=1 or len(sample_hashes)!=1:raise ValueError('Engine workloads or parity samples differ')
eligible=sorted([r for r in rows if r['eligible']],key=lambda r:(r['single_request_p95_ms'],-r['best_texts_per_second'],r['engine']))
result={'status':'PROVISIONAL_COMPONENT_SHORTLIST','first_candidate':eligible[0]['engine'] if eligible else None,
        'rule':'HF float32 dense cosine parity >=0.9999 and zero failed attempts; then batch1 concurrency1 p95 latency',
        'engines':rows,'profile':'one exploratory 10-second/at-least-100-attempt cell; no endurance claim',
        'matched_workload_hash':next(iter(workloads)),'same_parity_sample_verified':True,
        'full_retrieval_end_to_end_engine_comparison':False,'hardware_stage_may_start':True}
(OUT/'serving/shortlist.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
