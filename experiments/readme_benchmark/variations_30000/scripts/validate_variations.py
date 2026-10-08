"""Dataset-wide structural, provenance, invariance and repeat checks."""
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from build_variations import read_jsonl, numeric_signature, ISO_RE, API_RE, PLAN_RE, sha, write_json, write_jsonl

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
OUT=ROOT/'outputs'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=read_jsonl(DATA/'source_pairs.jsonl')
    original=read_jsonl(ROOT.parent/'engine_calibration_3000'/'data'/'source_rows.jsonl')
    variants=read_jsonl(DATA/'variations.jsonl')
    fixture_rows=read_jsonl(DATA/'document_attack_fixtures.jsonl')
    fixtures={r['fixture_id']:r for r in fixture_rows}
    docs=read_jsonl(ROOT.parent/'engine_calibration_3000'/'data'/'corpus.jsonl')
    corpus_ids={d['faq_id'] for d in docs}
    assert source==original, 'Source fixtures changed'
    assert len(source)==3000 and len(variants)==30000
    assert len({r['case_id'] for r in variants})==30000
    parents={r['실행 ID']:r for r in source}
    groups=defaultdict(list)
    repeats=defaultdict(list)
    failures=[]
    for row in variants:
        parent=parents[row['parent_case_id']]
        groups[row['parent_case_id']].append(row)
        query=row['query']; original_query=parent['사용자 질문']
        target=row['context_fixture_id']
        semantic_original=fixtures[target]['normal_query'] if target else original_query
        if target:
            assert target in fixtures and row['primary_variation']=='DOCUMENT_ATTACK'
            assert fixtures[target]['expected_answer']==parent['정답 예시']
        checks={
            'nonempty_and_changed':bool(query.strip()) and query!=original_query,
            'source_answer_preserved':row['expected_answer']==parent['정답 예시'],
            'source_type_preserved':row['type']==parent['항목 코드'],
            'source_route_preserved':row['expected_route']==parent['기대 처리 경로'],
            'source_status_preserved':row['expected_generation_status']==parent['기대 상태'],
            'numeric_values_preserved':numeric_signature(query)==numeric_signature(semantic_original),
            'timestamps_preserved':Counter(ISO_RE.findall(query))==Counter(ISO_RE.findall(semantic_original)),
            'api_enums_preserved':Counter(API_RE.findall(query))==Counter(API_RE.findall(semantic_original)),
            'plan_names_preserved':Counter(PLAN_RE.findall(query))==Counter(PLAN_RE.findall(semantic_original)),
            'gold_ids_exist':set(row['source_ids'])<=corpus_ids,
            'not_independently_approved':row['annotation_status']=='requires_independent_semantic_and_ground_truth_review',
            'trace_present':bool(row['change_trace']),
        }
        if not all(checks.values()):failures.append({'case_id':row['case_id'],'failed':[k for k,v in checks.items() if not v], 'original':semantic_original,'query':query})
        if row['type']=='RT':repeats[row['variation_repeat_group_id']].append(row)
    for pid,rs in groups.items():
        assert len(rs)==10 and {r['variant_index'] for r in rs}==set(range(1,11)),pid
        assert len({r['query'] for r in rs})==10,pid
    assert len(groups)==3000
    assert set(Counter(r['type'] for r in variants).values())=={2000}
    for key,rs in repeats.items():
        assert len(rs)==10,key
        # Repeat rows must differ only in execution identity, not actual input.
        assert len({r['model_query'] for r in rs})==1,key
        fixture_values=[]
        for r in rs:
            p=parents[r['parent_case_id']]
            fixture_values.append(json.dumps([p['제공 Context'],p['대화 이력 (JSON)'],p['사용자 정보·API (JSON)'],p['추가 페르소나 지시']],ensure_ascii=False,sort_keys=True))
        assert len(set(fixture_values))==1,key
    assert len(repeats)==200
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['hashes'].items():assert sha(DATA/name)==digest,name
    type_samples=[]
    for kind in dict.fromkeys(r['type'] for r in variants):
        parent_ids=[p for p,rs in groups.items() if rs[0]['type']==kind]
        for pid in [parent_ids[0],parent_ids[137]]:type_samples.extend(groups[pid])
    write_jsonl(OUT/'review_samples.jsonl',type_samples)
    write_json(OUT/'validation_failures.json',failures)
    report={'passed':not failures,'source_pairs':len(source),'variation_rows':len(variants),'variants_per_source':10,
            'within_source_distinct_questions':True,'source_original_preserved':True,'repeat_input_groups':len(repeats),
            'repeat_rows':sum(len(v) for v in repeats.values()),'failures':len(failures),
            'checks_scope':'mechanical invariants and provenance; not independent semantic/ground-truth approval',
            'review_sample_rows':len(type_samples),'annotation_complete':False,'benchmark_executed':False}
    write_json(OUT/'validation.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if failures:
        for failure in failures[:8]:print(json.dumps(failure,ensure_ascii=False))
        raise SystemExit(1)

if __name__=='__main__':main()
