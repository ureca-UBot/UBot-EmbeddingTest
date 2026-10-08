"""Check artifacts against source evidence; structural success is not semantic QA."""
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / 'data', ROOT / 'outputs'


def read(name):
    return [json.loads(s) for s in (DATA / name).read_text(encoding='utf-8').splitlines()]


def main():
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    rows, inputs, selected, corpus, baselines = map(read, ['variations.jsonl', 'model_inputs.jsonl', 'source_pairs.jsonl', 'corpus.jsonl', 'paired_baselines.jsonl'])
    raw_path = ROOT / 'source' / 'faqs.jsonl'
    raw = {r['faq_id']: r for r in [json.loads(s) for s in raw_path.read_text(encoding='utf-8-sig').splitlines()]}
    checks, failures = {}, []

    def check(name, condition, detail=None):
        checks[name] = bool(condition)
        if not condition:
            failures.append({'check': name, 'detail': detail})

    check('source_snapshot_sha256', hashlib.sha256(raw_path.read_bytes()).hexdigest() == manifest['source_sha256'])
    check('exact_sizes', len(selected) == len(baselines) == 3000 and len(rows) == len(inputs) == 30000)
    check('unique_case_ids_and_queries', len({r['case_id'] for r in rows}) == len({r['query'] for r in rows}) == 30000)
    by_parent = defaultdict(list)
    for row in rows:
        by_parent[row['parent_case_id']].append(row)
    check('ten_distinct_variants_per_parent', len(by_parent) == 3000 and all(len(rs) == 10 and len({r['query'] for r in rs}) == 10 and {r['variant_slot'] for r in rs} == {f'V{i:02d}' for i in range(1, 11)} for rs in by_parent.values()))
    source_by_parent = {f['parent_case_id']: f for f in selected}
    faqs = {f['faq_id']: f for f in corpus}
    check('all_seeds_eligible', all(f['seed_eligible'] and not f['source_issue_flags'] for f in selected))
    check('all_eligible_strata_retained', manifest['eligible_strata'] == manifest['represented_strata'] == len({(f['brand'], f['category'], f['language']) for f in selected}))
    check('source_provenance_and_aliases', all(all(sid in raw and raw[sid]['brand'] == f['brand'] and re.sub(r'\s+', ' ', raw[sid]['question']).strip() == f['question'] and re.sub(r'\s+', ' ', raw[sid]['answer']).strip() == re.sub(r'\s+', ' ', f['answer']).strip() for sid in f['source_faq_ids']) for f in corpus))
    check('exact_model_input_projection', all(set(i) == {'case_id', 'query', 'brand', 'evaluation_family'} and i['case_id'] == r['case_id'] and i['query'] == r['query'] and i['brand'] == r['brand'] for i, r in zip(inputs, rows)))
    check('no_ids_urls_answers_in_query_fields', all(all(f['faq_id'] not in r['query'] and all(sid not in r['query'] for sid in f['source_faq_ids']) for f in (faqs[sid] for sid in r['source_faq_ids'])) for r in rows))
    check('brand_scope_explicit', all(r['query'] == r['scope_prefix'] + r['query_body'] and r['scope_prefix'].startswith(r['brand']) for r in rows))
    check('correct_gt_cardinality', all(len(r['required_facts']) == (2 if r['variant_slot'] == 'V10' else 1) and len(set(r['source_faq_ids'])) == len(r['required_facts']) for r in rows))
    check('gt_points_to_source_answers', all(all(fact['acceptable_faq_ids'] == [sid] and fact['acceptable_raw_faq_ids'] == faqs[sid]['source_faq_ids'] and fact['request'] == faqs[sid]['question'] and fact['evidence_answer_sha256'] == hashlib.sha256(faqs[sid]['answer'].encode()).hexdigest() for sid, fact in zip(r['source_faq_ids'], r['required_facts'])) for r in rows))
    check('same_brand_compound_questions', all(all(faqs[sid]['brand'] == r['brand'] for sid in r['source_faq_ids']) for r in rows))
    # Split graph must include every multi-intent edge, including reciprocal ones.
    faq_split = {f['faq_id']: f['split_group_id'] for f in selected}
    check('all_supports_in_same_split_component', all(all(faq_split[sid] == r['split_group_id'] for sid in r['source_faq_ids']) for r in rows))
    check('no_split_or_holdout_assigned', all(r['split'] == 'unassigned' for r in rows))
    check('no_false_gold_or_threshold_claim', all(r['review_status'] == 'NEEDS_INDEPENDENT_SEMANTIC_AND_GT_REVIEW' and r['gt_granularity'] == 'FAQ_SUPPORT_BUNDLE' and r['generation_status'] == 'NOT_EVALUATED' for r in rows) and not manifest['calibration_readiness']['no_faq_threshold_fit'])
    lexicon = {(p['ko'], p['en']) for p in json.loads((DATA / 'term_lexicon.json').read_text(encoding='utf-8'))}
    check('bilingual_tags_have_real_term_substitution', all(('KR_EN_EQUIVALENT' in r['validation_tags']) == bool(r['language_pairs']) and all((p['ko'], p['en']) in lexicon for p in r['language_pairs']) for r in rows))
    preserved_number_errors, protected_errors = [], []
    for r in rows:
        source_questions = [faqs[sid]['question'] for sid in r['source_faq_ids']]
        source_questions += [t['companion_user_scope'] for t in r['change_trace'] if t['operation'] == 'combine_two_source_questions' and t.get('companion_user_scope')]
        expected = Counter(re.findall(r'\d+(?:[.,]\d+)*', ' '.join(source_questions)))
        actual = Counter(re.findall(r'\d+(?:[.,]\d+)*', r['query_body']))
        # Numeric format variation may remove thousands separators.
        exp = Counter()
        got = Counter()
        for v, n in expected.items(): exp[v.replace(',', '')] += n
        for v, n in actual.items(): got[v.replace(',', '')] += n
        if exp != got:
            preserved_number_errors.append(r['case_id'])
        # Restore the numeric whitespace variation before exact-label checking.
        restore = r['query_body']
        for t in reversed(r['change_trace']):
            if t['operation'] in {'numeric_unit_spacing', 'thousands_separator'}:
                restore = restore.replace(t['to'], t['from'], 1)
        for q in source_questions:
            labels = re.findall(r'\[[^\]]+\]|〈[^〉]+〉|\b(?:[A-Za-z]*\d[A-Za-z0-9+/-]*)\b', q)
            if any(label not in restore for label in labels): protected_errors.append(r['case_id'])
    check('all_numeric_values_preserved', not preserved_number_errors, preserved_number_errors[:20])
    check('explicit_product_labels_preserved', not protected_errors, protected_errors[:20])
    polarity_errors = []
    logic_pattern = r'없|않|불가능|불가|미적용|미납|아니|이상|이하|초과|미만|이내|이후|이전|\b(?:not|never|cannot|without)\b'
    for r in rows:
        expected = Counter(re.findall(logic_pattern, ' '.join(faqs[sid]['question'] for sid in r['source_faq_ids']), re.I))
        actual = Counter(re.findall(logic_pattern, r['query_body'], re.I))
        if any(actual[word] < n for word, n in expected.items()): polarity_errors.append(r['case_id'])
    check('original_negation_and_comparators_retained', not polarity_errors, polarity_errors[:20])
    audit = json.loads((OUT / 'token_audit.json').read_text(encoding='utf-8'))
    coverage = read('answer_chunk_coverage.jsonl')
    check('all_source_answers_covered_by_windows', len(coverage) == len(corpus) and all(r['fully_covered'] for r in coverage) and audit['all_answers_fully_covered'])
    check('windowed_reranker_inputs_within_1024', all(v['within_1024'] for v in audit['windowed_worst_query_pairs'].values()))
    check('bge_full_queries_within_native_8192', audit['query_tokens']['bge_m3']['max'] <= 8192)
    mined = read('hard_negative_candidates.jsonl')
    check('competitors_not_mislabeled_gold', len(mined) == 3000 and all(r['review_status'] == 'UNVERIFIED_COMPETITORS_NOT_GOLD_NEGATIVES' for r in mined))
    report = {'passed': all(checks.values()), 'checks': checks, 'failures': failures, 'independent_semantic_review_completed': False, 'atomic_fact_review_completed': False, 'no_faq_threshold_fit_ready': False, 'counts': {'parents': len(selected), 'variants': len(rows), 'unique_queries': len({r['query'] for r in rows}), 'corpus_faqs': len(corpus)}, 'limitations': manifest['notes']}
    (OUT / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'checks': len(checks), 'failures': failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
