"""Offline token audit, traceable reranker windows, and review-only competitors."""
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
import transformers

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / 'data', ROOT / 'outputs'


def read(name):
    return [json.loads(s) for s in (DATA / name).read_text(encoding='utf-8').splitlines()]


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_rows(name, rows):
    with (DATA / name).open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def load_tokenizer(cache_name):
    candidates = sorted((Path('/models/hf/hub') / cache_name / 'snapshots').iterdir())
    candidates = [p for p in candidates if (p / 'tokenizer_config.json').exists()]
    assert len(candidates) == 1, (cache_name, candidates)
    return AutoTokenizer.from_pretrained(str(candidates[0]), local_files_only=True, use_fast=True, trust_remote_code=False), candidates[0]


def stats(values):
    return {'min': min(values), 'p50': float(np.percentile(values, 50)), 'p95': float(np.percentile(values, 95)), 'max': max(values), 'over_512': sum(v > 512 for v in values), 'over_1024': sum(v > 1024 for v in values)}


def token_audit(rows, corpus):
    tokenizers = {}
    snapshots = {}
    for label, cache_name in [
        ('bge_m3', 'models--BAAI--bge-m3'),
        ('bge_reranker', 'models--BAAI--bge-reranker-v2-m3'),
        ('jina_reranker', 'models--jinaai--jina-reranker-v2-base-multilingual'),
    ]:
        tokenizer, snapshot = load_tokenizer(cache_name)
        assert tokenizer.is_fast
        tokenizers[label] = tokenizer
        snapshots[label] = {'snapshot': snapshot.name, 'tokenizer_class': type(tokenizer).__name__, 'tokenizer_config_sha256': hashlib.sha256((snapshot / 'tokenizer_config.json').read_bytes()).hexdigest()}
    bge = tokenizers['bge_m3']
    token_metadata = []
    query_lengths = {}
    for label, tok in tokenizers.items():
        sizes = []
        for start in range(0, len(rows), 256):
            encoded = tok([r['query'] for r in rows[start:start + 256]], add_special_tokens=True, truncation=False)
            sizes.extend(map(len, encoded['input_ids']))
        query_lengths[label] = sizes
    for i, r in enumerate(rows):
        token_metadata.append({'case_id': r['case_id'], **{label: values[i] for label, values in query_lengths.items()}})
    write_rows('query_token_lengths.jsonl', token_metadata)
    docs = []
    for f in corpus:
        header = f"{f['brand']} / {f['category']}\nQuestion: {f['question']}\nAnswer: "
        docs.append({'faq_id': f['faq_id'], 'question_text': f"{f['brand']} / {f['category']}\nQuestion: {f['question']}", 'content_text': header + f['answer']})
    write_rows('corpus_views.jsonl', docs)
    chunks = []
    coverage = []
    for faq_index, (f, view) in enumerate(zip(corpus, docs)):
        header = view['content_text'][:-len(f['answer'])]
        head_length = max(len(tok(header, add_special_tokens=True)['input_ids']) for tok in tokenizers.values())
        budget = 256 - head_length - 8
        assert budget >= 16, ('Question/header requires a different chunk policy', f['faq_id'], head_length)
        offsets = bge(f['answer'], add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        assert offsets, f['faq_id']
        start, ordinal = 0, 0
        intervals = []
        while start < len(offsets):
            end = min(len(offsets), start + budget)
            a = 0 if start == 0 else offsets[start][0]
            z = len(f['answer']) if end == len(offsets) else offsets[end - 1][1]
            text = header + f['answer'][a:z]
            counts = {label: len(tok(text, add_special_tokens=True)['input_ids']) for label, tok in tokenizers.items()}
            while max(counts.values()) > 256 and end > start + 1:
                end -= 1
                z = offsets[end - 1][1]
                text = header + f['answer'][a:z]
                counts = {label: len(tok(text, add_special_tokens=True)['input_ids']) for label, tok in tokenizers.items()}
            assert max(counts.values()) <= 256
            ordinal += 1
            chunks.append({'chunk_id': f"{f['faq_id']}-A{ordinal:03d}", 'faq_id': f['faq_id'], 'text': text, 'answer_start_char': a, 'answer_end_char': z, 'answer_slice_sha256': hashlib.sha256(f['answer'][a:z].encode()).hexdigest(), 'token_lengths': counts, 'usage': 'optional_reranker_window_only; source-level ranks must merge by faq_id'})
            intervals.append((a, z))
            if end == len(offsets):
                break
            overlap = min(32, (end - start) // 4)
            start = end - overlap
        assert intervals[0][0] == 0 and intervals[-1][1] == len(f['answer'])
        assert all(right[0] <= left[1] for left, right in zip(intervals, intervals[1:])), f['faq_id']
        coverage.append({'faq_id': f['faq_id'], 'answer_chars': len(f['answer']), 'chunks': ordinal, 'fully_covered': True})
        if faq_index % 1000 == 0:
            print(json.dumps({'stage': 'answer_windows', 'faqs': faq_index, 'chunks': len(chunks)}), flush=True)
    write_rows('reranker_chunks.jsonl', chunks)
    write_rows('answer_chunk_coverage.jsonl', coverage)
    unchunked = {}
    chunked = {}
    for label, tok in tokenizers.items():
        qs = query_lengths[label]
        # Worst actual query + every document/window, not a truncated estimate.
        longest_query = rows[max(range(len(rows)), key=lambda i: qs[i])]['query']
        full_lengths = [len(tok(longest_query, v['content_text'], truncation=False)['input_ids']) for v in docs]
        pair_max = max(len(tok(longest_query, c['text'], truncation=False)['input_ids']) for c in chunks)
        unchunked[label] = stats(full_lengths)
        chunked[label] = {'worst_pair_tokens': pair_max, 'within_1024': pair_max <= 1024}
    report = {
        'runtime': {'python': platform.python_version(), 'transformers': transformers.__version__, 'image': 'ubot-retrieval-jina:run-v1', 'remote_code_executed': False, 'network_enabled': False},
        'tokenizer_snapshots': snapshots, 'query_tokens': {k: stats(v) for k, v in query_lengths.items()},
        'full_document_worst_query_pairs': unchunked, 'windowed_worst_query_pairs': chunked,
        'corpus_faqs': len(corpus), 'reranker_chunks': len(chunks),
        'document_chunk_target_tokens': 256, 'maximum_answer_overlap_tokens': 32,
        'all_answers_fully_covered': all(c['fully_covered'] for c in coverage),
        'long_queries_not_truncated': True, 'serving_executed': False,
        'policy': 'Full documents remain the embedding source. Optional answer windows are source-preserving reranker preparation; adopting/merging them is an explicit architecture decision before measurement.',
    }
    save('token_audit.json', report)
    print(json.dumps({'stage': 'token_audit_done', 'query_tokens': report['query_tokens'], 'chunks': len(chunks)}, ensure_ascii=False), flush=True)


def mine_competitors(selected, corpus):
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), min_df=2, max_features=80000, sublinear_tf=True)
    features = vectorizer.fit_transform([f['question'] for f in corpus])
    positions = {f['faq_id']: i for i, f in enumerate(corpus)}
    mined = []
    for start in range(0, len(selected), 64):
        rs = selected[start:start + 64]
        scores = (features[[positions[r['faq_id']] for r in rs]] @ features.T).toarray()
        for faq, sims in zip(rs, scores):
            same, other = [], []
            for j in np.argsort(-sims):
                target = corpus[int(j)]
                if target['faq_id'] == faq['faq_id'] or ' '.join(target['answer'].split()) == ' '.join(faq['answer'].split()):
                    continue
                item = {'faq_id': target['faq_id'], 'brand': target['brand'], 'question': target['question'], 'char_tfidf_similarity': float(sims[j]), 'source_issue_flags': target['source_issue_flags']}
                dest, cap = (same, 3) if target['brand'] == faq['brand'] else (other, 2)
                if len(dest) < cap:
                    dest.append(item)
                if len(same) == 3 and len(other) == 2:
                    break
            mined.append({'parent_case_id': faq['parent_case_id'], 'seed_faq_id': faq['faq_id'], 'same_brand_candidates': same, 'cross_brand_candidates': other, 'review_status': 'UNVERIFIED_COMPETITORS_NOT_GOLD_NEGATIVES', 'candidate_source': 'new FAQ corpus only, char TF-IDF; no prior retrieval score imported'})
    write_rows('hard_negative_candidates.jsonl', mined)
    save('hard_negative_audit.json', {'parents': len(mined), 'candidate_references': sum(len(r['same_brand_candidates']) + len(r['cross_brand_candidates']) for r in mined), 'validated_hard_negatives': 0, 'warning': 'Different answers or providers do not by themselves prove a negative; review scope and fact overlap before scoring.'})
    print(json.dumps({'stage': 'competitors_done', 'parents': len(mined)}), flush=True)


if __name__ == '__main__':
    rows, corpus, selected = read('variations.jsonl'), read('corpus.jsonl'), read('source_pairs.jsonl')
    token_audit(rows, corpus)
    mine_competitors(selected, corpus)
