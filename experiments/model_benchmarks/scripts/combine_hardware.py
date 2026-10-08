"""Compare separate CPU/GPU dense probes without starting either model."""
import hashlib
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'outputs/retrieval/run-20261006-v1'

def main():
    root = OUT / 'hardware'
    cpu = json.loads((root / 'cpu.json').read_text(encoding='utf-8-sig'))
    gpu = json.loads((root / 'cuda.json').read_text(encoding='utf-8-sig'))
    assert cpu['query_hash'] == gpu['query_hash'], 'Hardware workloads differ'
    a = np.load(root / 'cpu-vectors.npy')
    b = np.load(root / 'cuda-vectors.npy')
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    cosine = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    result = {'query_count': len(a), 'query_hash': cpu['query_hash'],
              'parity_min_cosine': float(cosine.min()), 'parity_mean_cosine': float(cosine.mean()),
              'max_absolute_error': float(np.max(np.abs(a - b))),
              'scope': 'separate frozen FP32 SentenceTransformer dense component probes',
              'cpu_gpu_engine_end_to_end_comparison': False,
              'results': {'cpu': cpu['results'], 'cuda': gpu['results']},
              'input_file_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [root / 'cpu.json', root / 'cuda.json']}}
    (root / 'cpu-gpu.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
