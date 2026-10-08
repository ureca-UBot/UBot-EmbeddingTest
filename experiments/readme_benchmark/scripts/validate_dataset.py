"""Validate the new folder's inputs, labels and split contracts, no models."""
from __future__ import annotations
import hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path
from jsonschema import Draft202012Validator
from retrieval_common import semantic_contract_errors

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
def lines(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def catalog():
    base={d['faq_id']:d for d in lines(DATA/'corpus.jsonl')}
    manifest=json.loads((DATA/'split_manifest.json').read_text(encoding='utf-8'))
    result={manifest['corpus_version']:base}
    for d in lines(DATA/'attack_fixtures.jsonl'):
        result[d['corpus_version']]={**{k:v for k,v in base.items() if k!=d['base_faq_id']},d['faq_id']:d}
    return result
def validate():
    manifest=json.loads((DATA/'split_manifest.json').read_text(encoding='utf-8'))
    cases=lines(DATA/'cases.jsonl');errors=[]
    for name,value in manifest['hashes'].items():
        if sha(DATA/name)!=value:errors.append('Frozen input hash changed: '+name)
    schema=json.loads((ROOT/'schemas/retrieval_case.schema.json').read_text(encoding='utf-8'))
    Draft202012Validator.check_schema(schema)
    validator=Draft202012Validator(schema)
    for c in cases:
        errors.extend(f"{c['case_id']}: {e.message}" for e in validator.iter_errors(c))
        if c['source']['origin']!='authored' or c['source']['source_key']!='README_ONLY_V1':errors.append('Non-authored source: '+c['case_id'])
        if c['source']['row'] is not None or c['source']['original_question_id'] is not None:errors.append('Source test label leak: '+c['case_id'])
    if not errors:errors.extend(semantic_contract_errors(cases,catalog()))
    # Additional split guard on acceptable evidence, not just query families.
    fact_splits=defaultdict(set);repeat=defaultdict(list);pairs=defaultdict(list)
    for c in cases:
        if c['track']=='core':
            for g in c['fact_groups']:
                for i in g['primary_ids']+g['acceptable_ids']:fact_splits[i].add(c['split'])
        if c.get('repeat'):repeat[c['repeat']['group_id']].append(c)
        if c.get('semantic_pair_id'):pairs[c['semantic_pair_id']].append(c)
    errors.extend('Evidence crosses splits: '+i for i,ss in fact_splits.items() if len(ss)>1)
    for key,group in repeat.items():
        if sorted(c['repeat']['index'] for c in group)!=list(range(1,group[0]['repeat']['total']+1)):errors.append('Incomplete repeat: '+key)
        if len({(c['query'],json.dumps(c['history'])) for c in group})!=1:errors.append('Repeated input changed: '+key)
    # Term pairs are literal substitutions. Full translations retain their
    # declared conditions; proof text is reviewed separately, not auto-approved.
    for key,group in pairs.items():
        if key.startswith('TERM-'):
            ko=next(c for c in group if c['language_variant']=='ko_anchor')
            en=next(c for c in group if c['language_variant']=='en_term')
            if re.findall(r'\d+(?:\.\d+)?',ko['query'])!=re.findall(r'\d+(?:\.\d+)?',en['query']):errors.append('Term replacement changed numbers: '+key)
    result={'valid':not errors,'errors':errors,'case_rows':len(cases),'corpus_documents':1024,
            'cohorts':dict(Counter(c['reporting_cohort'] for c in cases)),
            'splits':dict(Counter(c['split'] for c in cases)),
            'pending_annotations':sum(c['annotation']['status']=='pending' for c in cases),
            'approved_annotations':sum(c['annotation']['status']=='approved' for c in cases),
            'formal_acceptance_set':False,'provisional_benchmark_allowed':not errors,
            'old_test_questions_imported':0,'models_executed':False}
    out=ROOT/'outputs/run-v1';out.mkdir(parents=True,exist_ok=True)
    (out/'dataset_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2));return result
if __name__=='__main__':raise SystemExit(0 if validate()['valid'] else 1)
