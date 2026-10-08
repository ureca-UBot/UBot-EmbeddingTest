import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/run-v1'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
cpu=read(OUT/'hardware/cpu.json');gpu=read(OUT/'hardware/cuda.json')
if cpu['query_hash']!=gpu['query_hash']:raise ValueError('Hardware workload mismatch')
c=np.load(OUT/'hardware/cpu-vectors.npy');g=np.load(OUT/'hardware/cuda-vectors.npy')
cosine=np.sum(c*g,axis=1)/(np.linalg.norm(c,axis=1)*np.linalg.norm(g,axis=1))
result={'scope':'Same SentenceTransformer BGE-M3 float32 embedding component after fresh engine comparison and parity audit',
        'cpu_gpu_engine_end_to_end_comparison':False,'query_hash':cpu['query_hash'],'cosine_min':float(cosine.min()),
        'max_absolute_difference':float(np.max(np.abs(c-g))),'parity_passed':bool(cosine.min()>=.9999),
        'results':cpu['results']+gpu['results']}
(OUT/'hardware/cpu-gpu.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if not result['parity_passed']:raise SystemExit(1)
