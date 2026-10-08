import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs/run-v1'
KS = [1, 3, 5, 10, 20]
FIELDS = ['query','expected_faq_id','faq_id','retrieval_rank','retrieval_cosine',
          'rerank_rank','rerank_score','window_count','winning_window_index',
          'winning_window_text','winning_window_score','rank_delta',
          'retrieval_a_rank','retrieval_a_cosine','retrieval_b_rank','retrieval_b_cosine']

def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def lines(path): return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(tmp, path)
def save_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        for row in rows: stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    os.replace(tmp, path)
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()
def inputs():
    manifest = read(DATA/'manifest.json')
    for file, digest in manifest['hashes'].items(): assert sha(DATA/file) == digest, file
    cases, docs = lines(DATA/'cases.jsonl'), lines(DATA/'corpus.jsonl')
    assert len(cases) == manifest['inference_rows'] and len(docs) == manifest['corpus_faqs']
    return cases, docs
def model_path(name): return read(ROOT/'configs/resolved_model_revisions.json')[name]['snapshot_path']
def doc_text(doc, view):
    return doc['question'] if view == 'question' else doc['question']+'\n\n'+doc['answer']
def rankings(scores): return np.argsort(-scores, axis=1, kind='stable')
def pools(a, b, k=20):
    ar, br = rankings(a), rankings(b)
    merged = np.maximum(a, b)
    return [sorted(set(map(int, aa[:k])) | set(map(int, bb[:k])),
                   key=lambda j: (-float(merged[i,j]), j)) for i,(aa,bb) in enumerate(zip(ar,br))]
def expected_ids(case):
    return list(dict.fromkeys(fid for fact in case['required_facts'] for fid in fact['acceptable_faq_ids']))
def metrics(ids, facts):
    if not facts: return None
    positions = {fid:i+1 for i,fid in enumerate(ids)}
    hit_facts = [any(fid in positions for fid in f['acceptable_faq_ids']) for f in facts]
    ranks = [positions[fid] for f in facts for fid in f['acceptable_faq_ids'] if fid in positions]
    return {'hit':int(any(hit_facts)), 'fact_recall':sum(hit_facts)/len(facts),
            'all_facts_hit':int(all(hit_facts)), 'mrr':1/min(ranks) if ranks else 0.}
def summarize(engine, name, cases, docs, ranked, scores, execution, candidate_sets=None):
    folder = OUT/'engines'/engine/name
    ids = [doc['faq_id'] for doc in docs]
    rows = []
    for i,(case,rr) in enumerate(zip(cases,ranked)):
        rr = list(map(int, rr))
        values = {}
        for k in KS:
            chosen = candidate_sets[k][i] if candidate_sets else rr[:k]
            values[str(k)] = metrics([ids[j] for j in chosen], case['required_facts'])
        rows.append({'case_id':case['case_id'], 'parent_case_id':case['parent_case_id'],
                     'evaluation_role':case['evaluation_role'], 'execution_group':case['execution_group'],
                     'purpose':case['primary_validation_purpose'], 'query':case['query'],
                     'gt_annotation_version':case.get('gt_annotation_version','INITIAL_SOURCE_GT_V1'),
                     'model_query':case['model_query'], 'expected_faq_id':expected_ids(case),
                     'required_facts':case['required_facts'], 'retrieval_status':case['retrieval_status'],
                     'candidate_count':len(rr), 'top_ids':[ids[j] for j in rr[:20]],
                     'top_scores':[float(scores[i,j]) for j in rr[:20]], 'metrics':values})
    groups = defaultdict(list)
    for row in rows:
        groups['execution_group:'+row['execution_group']].append(row)
        groups['purpose:'+row['purpose']].append(row)
        if row['evaluation_role'] == 'main' and row['execution_group'] == 'core': groups['main_core'].append(row)
    aggregates = {}
    for label,group in sorted(groups.items()):
        valid = [r for r in group if r['metrics']['3'] is not None]
        means = {}
        for k in KS:
            for key in ['hit','fact_recall','all_facts_hit','mrr']:
                means[f'{key}@{k}'] = float(np.mean([r['metrics'][str(k)][key] for r in valid])) if valid else None
        aggregates[label] = {'rows':len(group), 'scored_rows':len(valid), 'row_mean':means}
    summary = {'engine':engine, 'structure':name, 'inference_rows':len(cases),
               'main_rows':sum(c['evaluation_role']=='main' for c in cases),
               'groups':aggregates, 'execution':execution,
               'gt_status':'source-grounded synthetic diagnostic; independent semantic equivalence/alternative FAQ review pending',
               'candidate_recall_k_semantics':'per-view K, complete deduplicated union' if candidate_sets else 'ranked prefix K'}
    save_lines(folder/'per_query.jsonl', rows)
    save(folder/'summary.json', summary)
    print(json.dumps({'complete':name,'engine':engine,'main_core':aggregates.get('main_core')},ensure_ascii=False), flush=True)
    return rows

def export_log(engine, name, view, cases, docs, candidates, rerank_scores, execution):
    base = OUT/'engines'/engine
    a, b = np.load(base/'cache/A-scores.npy'), np.load(base/'cache/B-scores.npy')
    ar, br = rankings(a), rankings(b)
    ai, bi = np.empty_like(ar), np.empty_like(br)
    indices = np.arange(len(docs))+1
    np.put_along_axis(ai, ar, np.broadcast_to(indices, ar.shape), axis=1)
    np.put_along_axis(bi, br, np.broadcast_to(indices, br.shape), axis=1)
    count = 0
    outcomes = defaultdict(int)
    transitions = []
    def rows():
        nonlocal count
        for i,(case,ids) in enumerate(zip(cases,candidates)):
            reordered = sorted(ids,key=lambda j:(-float(rerank_scores[i,j]),j))
            inverse = {j:p+1 for p,j in enumerate(reordered)}
            accepted = set(expected_ids(case))
            before = min((p+1 for p,j in enumerate(ids) if docs[j]['faq_id'] in accepted),default=None)
            after = min((p+1 for p,j in enumerate(reordered) if docs[j]['faq_id'] in accepted),default=None)
            outcome = ('UNSCORED' if not accepted else 'STAGE1_MISS' if before is None else
                       'IMPROVED' if after < before else 'WORSENED' if after > before else 'UNCHANGED')
            outcomes[outcome] += 1
            transitions.append({'case_id':case['case_id'],'query':case['query'],'expected_faq_id':list(accepted),
                                'execution_group':case['execution_group'],'before_rank':before,'after_rank':after,
                                'rank_delta':before-after if before else None,'outcome':outcome})
            for p,j in enumerate(ids):
                score = float(rerank_scores[i,j])
                row = dict(zip(FIELDS,[case['query'],expected_ids(case),docs[j]['faq_id'],p+1,
                    float(max(a[i,j],b[i,j])),inverse[j],score,1,0,doc_text(docs[j],view),score,
                    p+1-inverse[j],int(ai[i,j]),float(a[i,j]),int(bi[i,j]),float(b[i,j])]))
                row.update(case_id=case['case_id'], model_query=case['model_query'], engine=engine,
                           structure=name, execution_group=case['execution_group'], purpose=case['primary_validation_purpose'],
                           required_facts=[{'fact_id':f['fact_id'],'acceptable_faq_ids':f['acceptable_faq_ids']} for f in case['required_facts']],
                           gt_annotation_version=case.get('gt_annotation_version','INITIAL_SOURCE_GT_V1'),
                           candidate_count=len(ids))
                assert row['rank_delta'] == row['retrieval_rank'] - row['rerank_rank']
                assert np.isfinite([row[x] for x in ['retrieval_cosine','rerank_score','retrieval_a_cosine','retrieval_b_cosine']]).all()
                count += 1
                yield row
    path = base/name/'candidate_log.jsonl'
    save_lines(path, rows())
    save_lines(base/name/'rank_changes.jsonl', transitions)
    save(base/name/'candidate_log_schema.json', {'requested_fields':FIELDS,'row_count':count,'sha256':sha(path),
        'ranks':'1-based; A/B rank in the entire corpus; retrieval_rank within unpruned C union',
        'rerank_sort':'raw logit descending; equal logits use corpus order',
        'retrieval_sort':'max(A_cosine,B_cosine) descending for diagnostic ordering; corpus order breaks ties',
        'expected_faq_id':'union of acceptable FAQ IDs; use required_facts for all-facts scoring, not all IDs required',
        'window_contract':'one full FAQ view; index=0, count=1, winning score equals rerank_score',
        'context_query':'query is original utterance; model_query is frozen rewrite for context group',
        'rank_changes':dict(outcomes),'execution':execution})
    print(json.dumps({'candidate_log':str(path),'rows':count,'rank_changes':dict(outcomes)}),flush=True)
