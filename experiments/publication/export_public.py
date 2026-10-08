"""Export allowlisted aggregate fields and source identifiers; never FAQ text."""
import argparse
import json
import math
import re
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / 'experiments/readme_benchmark/readme_serving_retest/fair_eval_v4'
OUT = ROOT / 'experiments/public_results'


def numeric_tree(value):
    if value is None or type(value) in (int, bool):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not re.fullmatch(r'[A-Za-z0-9_@:+./<>=-]+', key):
                raise ValueError('Unexpected metric key')
            numeric_tree(child)
        return
    raise ValueError('Metric contains non-numeric payload')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def export(v4=V4, destination=OUT):
    raw = json.loads((v4 / 'serving_test/outputs/run-v4/results.json').read_text(encoding='utf-8'))
    metrics = {}
    for dataset in ('existing', 'kt', 'skt', 'lgu'):
        metrics[dataset] = {}
        for engine in ('ollama', 'vllm'):
            metrics[dataset][engine] = {}
            for structure, result in raw[dataset][engine].items():
                if not re.fullmatch(r'[A-Za-z0-9_-]+', structure):
                    raise ValueError('Unexpected structure ID')
                groups = {}
                for group, values in result['groups'].items():
                    if not re.fullmatch(r'[A-Za-z0-9_:/-]+', group):
                        raise ValueError('Unexpected reporting group')
                    groups[group] = {key: values[key] for key in ('rows', 'scored_rows', 'groups', 'row_mean', 'leakage_group_macro')}
                    numeric_tree(groups[group])
                metrics[dataset][engine][structure] = {'inference_rows': result['inference_rows'], 'groups': groups}
    numeric_tree(metrics)

    raw_load = json.loads((v4 / 'serving_test/outputs/run-v4/load_results.json').read_text(encoding='utf-8'))
    load = {}
    fields = ('concurrency', 'requests', 'successes', 'failures', 'seconds', 'requests_per_second', 'latency_seconds', 'top1_consistency', 'min_embedding_cosine')
    for dataset in metrics:
        load[dataset] = {}
        for engine in metrics[dataset]:
            runs = [{key: row[key] for key in fields} for row in raw_load[dataset][engine]['runs']]
            for row in runs:
                numeric_tree(row)
            load[dataset][engine] = runs

    sources = []
    allowed_hosts = {'ermsweb.kt.com', 'globalshop.kt.com', 'www.tworld.co.kr', 'www.lguplus.com'}
    for brand in ('kt', 'skt', 'lgu'):
        path = v4 / 'datasets' / brand / 'faq_pairs.jsonl'
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            urls = row['source_urls']
            if not urls or any(urlsplit(url).scheme != 'https' or urlsplit(url).hostname not in allowed_hosts for url in urls):
                raise ValueError('Unexpected source URL')
            sources.append({'faq_id': row['faq_id'], 'source_urls': list(dict.fromkeys(urls))})
    if len(sources) != len({row['faq_id'] for row in sources}):
        raise ValueError('Duplicate FAQ identifier')
    write(destination / 'aggregate_metrics.json', metrics)
    write(destination / 'load_metrics.json', load)
    write(destination / 'faq_sources.json', sources)
    print(json.dumps({'structures': sum(len(s) for d in metrics.values() for s in d.values()), 'source_ids': len(sources), 'faq_body_exported': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--v4-root', type=Path, default=V4)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    export(args.v4_root, args.output)
