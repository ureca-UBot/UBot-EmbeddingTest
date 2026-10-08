"""Audit saved results and record identities without another model evaluation."""
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/run-v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def memory_bytes(value):
    match = re.fullmatch(r'([\d.]+)\s*([A-Za-z]+)', value.strip())
    if not match:
        raise ValueError(f'Unrecognized docker memory sample: {value}')
    scales = {'B': 1, 'kB': 1000, 'MB': 1000**2, 'GB': 1000**3,
              'KiB': 1024, 'MiB': 1024**2, 'GiB': 1024**3}
    return float(match[1]) * scales[match[2]]


def main():
    manifest = read(ROOT / 'data/split_manifest.json')
    for name, expected in manifest['hashes'].items():
        assert digest(ROOT / 'data' / name) == expected, f'Frozen input changed: {name}'
    holdout = read(OUT / 'holdout/policy_results.json')
    assert holdout['dataset_hash'] == digest(ROOT / 'data/split_manifest.json')
    assert holdout['selection_hash'] == digest(OUT / 'selection.json')
    assert holdout['holdout_passes'] == 1
    shortlist = read(OUT / 'serving/shortlist.json')
    assert shortlist['same_parity_sample_verified']
    assert read(OUT / 'hardware/cpu-gpu.json')['parity_passed']
    assert read(OUT / 'final_shutdown.json')['verified_stopped']

    resources = {}
    total_attempts = total_failures = 0
    corpus = [json.loads(line) for line in (ROOT / 'data/corpus.jsonl').read_text(encoding='utf-8').splitlines()]
    for engine in ['ollama', 'tei', 'vllm']:
        folder = OUT / 'serving' / engine
        loads = read(folder / 'load_results.json')
        assert {(r['concurrency'], r['texts_per_request']) for r in loads} == {
            (c, b) for c in [1, 4, 8, 16, 32] for b in [1, 8, 32]}
        assert len(loads) == 15
        for row in loads:
            assert row['attempts'] >= 100 and row['wall_seconds'] >= 10
            assert row['attempts'] == row['successes'] + row['failures']
            total_attempts += row['attempts']
            total_failures += row['failures']
        assert read(folder / 'stop_verification.json')['verified_stopped']
        ram = []
        host_vram = []
        for line in (folder / 'memory_samples.jsonl').read_text(encoding='utf-8-sig').splitlines():
            sample = json.loads(line)
            if sample.get('docker_stats'):
                stats = json.loads(sample['docker_stats'])
                ram.append(memory_bytes(stats['MemUsage'].split('/')[0]))
            if sample.get('gpu_memory_and_utilization'):
                host_vram.append(float(sample['gpu_memory_and_utilization'].split(',')[0]))
        assert ram and host_vram
        parity = read(folder / 'parity.json')
        ref = np.load(OUT / f"cache/dense-{parity['document_view']}-corpus.npy")
        vectors = np.load(folder / 'corpus_vectors.npy')
        normalized = vectors / np.linalg.norm(vectors, axis=1)[:, None]
        cos = np.sum(normalized * ref, axis=1)
        worst = np.argsort(cos)[:5]
        resources[engine] = {
            'container_ram_sampled_peak_bytes': max(ram),
            'host_total_vram_sampled_peak_mib': max(host_vram),
            'ram_samples': len(ram), 'vram_samples': len(host_vram),
            'memory_scope': 'container Docker RAM; host total VRAM includes desktop and other processes',
            'cold_start_to_ready_seconds': read(folder / 'readiness.json')['cold_start_to_ready_seconds'],
            'corpus_vectors_below_0_9999': int(np.sum(cos < .9999)),
            'worst_corpus_cosines': [{'faq_id': corpus[int(i)]['faq_id'], 'cosine': float(cos[i])} for i in worst],
            'top1_agreement': parity['top1_agreement'], 'top10_overlap': parity['top10_overlap'],
            'measured_attempts': sum(r['attempts'] for r in loads),
            'failed_attempts': sum(r['failures'] for r in loads),
        }
    save(OUT / 'serving/resource_summary.json', resources)

    scripts = {}
    for path in sorted((ROOT / 'scripts').glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        scripts[str(path.relative_to(ROOT)).replace('\\', '/')] = digest(path)
    save(OUT / 'execution_code_identity.json', {
        'data_hashes_still_frozen': True, 'legacy_imports': False,
        'fresh_compose_mount_scope': 'readme_benchmark only + outputs RW + cached model volumes',
        'scripts_sha256': scripts, 'compose_sha256': digest(ROOT / 'compose.yaml'),
        'powershell_sha256': {p.name: digest(p) for p in sorted(ROOT.glob('*.ps1'))},
        'note': 'Final source snapshot; inference scripts are unchanged except hardware scope wording after engine audit.',
    })
    save(OUT / 'completion_audit.json', {
        'status': 'COMPLETED_PROVISIONAL_RETRIEVAL_AND_COMPONENT_EXPERIMENT',
        'frozen_inputs_verified': True, 'holdout_passes': 1,
        'serving_cells': 45, 'serving_attempts': total_attempts, 'serving_failed_attempts': total_failures,
        'first_engine_candidate': shortlist['first_candidate'], 'all_engines_stopped': True,
        'hardware_cells': 6, 'hardware_parity_passed': True,
        'independent_annotation_review_complete': False,
        'generation_and_api_tests_executed': False,
        'not_executed': ['independent lexical-miss review / conditional BM25',
                         'generation guardrail behavior', 'personal-data API execution',
                         'clarification execution policy', 'real failure-log regression',
                         'policy version validity tests', 'long-running repeated engine load',
                         'complete retrieval-plus-reranker engine end-to-end comparison'],
    })
    print(json.dumps({'serving_attempts': total_attempts, 'serving_failed_attempts': total_failures,
                      'frozen_inputs_verified': True, 'all_engines_stopped': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
