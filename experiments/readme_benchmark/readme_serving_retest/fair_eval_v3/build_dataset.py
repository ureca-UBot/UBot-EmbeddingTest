"""Compile separately authored questions into four independent retrieval datasets."""
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = {"existing": None, "kt": "KT", "skt": "SKT", "lgu": "LG U+"}

def read(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def token(value):
    return hashlib.sha256(value.encode()).hexdigest()[:16]

def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in rows),encoding="utf-8",newline="\n")

def norm(s):
    return re.sub(r"[\W_]", "", s.casefold())

class Groups:
    def __init__(self): self.parent = {}
    def find(self,x):
        self.parent.setdefault(x,x)
        if self.parent[x]!=x:self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        self.parent[max(a,b)] = min(a,b)

def authored(corpus):
    queue = {r['authoring_id']:r for r in read(ROOT/'authoring/queue.jsonl')}
    aliases = read(ROOT/'authoring/aliases.jsonl') if (ROOT/'authoring/aliases.jsonl').exists() else []
    alias_map = {r['question_id']:r for r in aliases}
    cases=[]; facts=[]; seen=Counter(); used_aliases=set()
    for path in sorted((ROOT/'authoring').glob('questions_*.txt')):
        for lineno,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
            if not line.strip() or line.startswith('#'):continue
            sid,purpose,query,quote=line.split('|',3)
            d=queue[sid]; seen[sid]+=1
            qid=f"Q3-{sid}-{seen[sid]:02}"
            key=next(k for k,v in DATASETS.items() if v==d['brand'])
            # Quotes may have multiple non-contiguous pieces, all must occur.
            quotes=quote.split(' || ')
            assert all(q in d['answer'] for q in quotes),(path.name,lineno,sid,quote)
            accepted={d['faq_id']}
            support=[{'faq_id':d['faq_id'],'quotes':quotes,'review':'author_source_check'}]
            if qid in alias_map:
                a=alias_map[qid];used_aliases.add(qid)
                for s in a['supports']:
                    candidate=corpus[s['faq_id']]
                    assert candidate['brand']==d['brand']
                    assert all(x in candidate['answer'] for x in s['quotes'])
                    accepted.add(s['faq_id']);support.append(s)
            fid=f"F3-{sid}-{seen[sid]:02}"
            facts.append({'fact_id':fid,'dataset_id':key,'requested_information':query,
                'source_question_scope':d['question'],'acceptable_faq_ids':sorted(accepted),'support':support,
                'independent_support_review':'pending'})
            cases.append({'question_id':qid,'case_id':qid,'dataset_id':key,'user_question':query,'query':query,
                'model_query':query,'answer_faq_ids':sorted(accepted),'expected_faq_ids':sorted(accepted),
                'required_facts':[{'fact_id':fid,'acceptable_faq_ids':sorted(accepted)}],
                'source_faq_ids':[d['faq_id']],'source_authoring_id':sid,'category':d['category'],
                'primary_purpose':purpose,'purpose_tags':[purpose],'corpus_status':'FAQ_EXISTS',
                'origin':'new_assistant_authored_synthetic','language':'ko','dialogue_history':[],
                'review_status':'independent_review_pending','eligible_for_final_benchmark':False,
                'evaluation_track':'retrieval','binary_fit_eligible':True,
                'policy_family':key+':'+d['category']})
    assert used_aliases==set(alias_map),'Stale alternative-answer annotation'
    return cases,facts

def add_controls(cases):
    base={q['question_id']:q for q in cases};extra=[]
    path=ROOT/'authoring/controls.txt'
    if not path.exists():return cases
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):continue
        key,purpose,lang,text=line.split('|',3)
        original=base['Q3-'+key]
        q=json.loads(json.dumps(original));qid='C3-'+token(line)
        q.update(question_id=qid,case_id=qid,primary_purpose=purpose,purpose_tags=[purpose,original['primary_purpose']],
            language='ko-en' if lang=='mixed' else lang,evaluation_track='controlled',binary_fit_eligible=False,control_of=original['question_id'],
            pair_ids=['P3-'+key+'-'+purpose])
        original['pair_ids']=sorted(set(original.get('pair_ids',[])+q['pair_ids']))
        if purpose=='CONTEXT_RETRIEVAL':
            history,query=text.split(' >> ',1)
            q.update(dialogue_history=[{'role':'user','content':history}],user_question=query,query=query,
                model_query=original['model_query'],rewrite_method='fixed_author_rewrite_no_answer')
        else:q.update(user_question=text,query=text,model_query=text)
        extra.append(q)
    return cases+extra

def annotate_near_misses(cases,catalog):
    refs={d['authoring_id']:d['faq_id'] for d in read(ROOT/'authoring/queue.jsonl')}
    refs.update({f'L{i:03}':d['faq_id'] for i,d in enumerate([d for d in catalog if d['brand']=='LG U+'],1)})
    docs={d['faq_id']:d for d in catalog};base={q['question_id']:q for q in cases}
    for line in (ROOT/'authoring/hard_negatives.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):continue
        key,ref,reason=line.split('|',2);fid=refs.get(ref,ref);q=base['Q3-'+key]
        assert fid not in q['answer_faq_ids'] and docs[fid]['brand']==DATASETS[q['dataset_id']]
        q['purpose_tags'].insert(0,'HARD_NEGATIVE');q['primary_purpose']='HARD_NEGATIVE'
        q['near_miss_candidates']=[{'faq_id':fid,'reason':reason,'independent_review':'pending'}]
    return cases

def add_composites(cases):
    base={q['question_id']:q for q in cases};extra=[]
    path=ROOT/'authoring/composites.txt'
    if not path.exists():return cases
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):continue
        keys,purpose,query=line.split('|',2)
        originals=[base['Q3-'+key] for key in keys.split(',')]
        assert len({q['dataset_id'] for q in originals})==1
        q=json.loads(json.dumps(originals[0]));qid='M3-'+token(line)
        q.pop('pair_ids',None)
        q.pop('near_miss_candidates',None)
        facts={f['fact_id']:f for x in originals for f in x['required_facts']}
        q.update(question_id=qid,case_id=qid,user_question=query,query=query,model_query=query,
            primary_purpose=purpose,purpose_tags=[purpose],required_facts=list(facts.values()),
            answer_faq_ids=sorted({f for x in originals for f in x['answer_faq_ids']}),
            source_faq_ids=sorted({f for x in originals for f in x['source_faq_ids']}),
            component_question_ids=[x['question_id'] for x in originals],binary_fit_eligible=False,
            evaluation_track='multi_fact',category=' + '.join(sorted({x['category'] for x in originals})))
        q['expected_faq_ids']=q['answer_faq_ids'];extra.append(q)
    return cases+extra

def add_status_questions(cases,facts,catalog):
    base={q['question_id']:q for q in cases};extra=[];audit=[]
    path=ROOT/'authoring/status_questions.txt'
    if not path.exists():return cases,facts
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):continue
        key,status,refs,query,missing,terms=line.split('|',5)
        assert key in DATASETS and key!='existing'
        originals=[] if refs=='-' else [base['Q3-'+x] for x in refs.split(',')]
        assert all(q['dataset_id']==key for q in originals)
        assert status in {'NO_FAQ','PARTIAL','UNDERSPECIFIED'}
        qid='N3-'+token(line);required=[]
        if status=='PARTIAL':
            assert len(originals)==1
            required=json.loads(json.dumps(originals[0]['required_facts']))
        if status!='UNDERSPECIFIED':
            fid='U3-'+token(key+':'+missing)
            required.append({'fact_id':fid,'acceptable_faq_ids':[]})
            facts.append({'fact_id':fid,'dataset_id':key,'requested_information':missing,
                'acceptable_faq_ids':[],'support':[],
                'independent_support_review':'pending','absence_review':'provisional'})
        ids=sorted({fid for f in required for fid in f['acceptable_faq_ids']})
        category=originals[0]['category'] if originals else 'answer_absence_challenge'
        family=originals[0]['policy_family'] if originals else key+':absence:'+token(missing)
        q={'question_id':qid,'case_id':qid,'dataset_id':key,'user_question':query,'query':query,
            'model_query':query,'answer_faq_ids':ids,'expected_faq_ids':ids,'required_facts':required,
            'source_faq_ids':sorted({f for x in originals for f in x['source_faq_ids']}),
            'category':category,'primary_purpose':'AMBIGUOUS_QUERY' if status=='UNDERSPECIFIED' else status,
            'purpose_tags':['AMBIGUOUS_QUERY' if status=='UNDERSPECIFIED' else status],
            'corpus_status':'PARTIAL_FACTS_EXIST' if status=='PARTIAL' else status,'origin':'new_assistant_authored_synthetic','language':'ko',
            'dialogue_history':[],'review_status':'independent_review_pending',
            'eligible_for_final_benchmark':False,'evaluation_track':'abstention_challenge',
            'binary_fit_eligible':False,'policy_family':family,'unresolved_information':missing,
            'absence_review':'provisional' if status!='UNDERSPECIFIED' else 'not_applicable',
            'clarification_needed':status=='UNDERSPECIFIED',
            'interpretation_candidate_ids':sorted({f for x in originals for f in x['answer_faq_ids']}) if status=='UNDERSPECIFIED' else [],
            'component_question_ids':[x['question_id'] for x in originals],
            'leakage_anchor_ids':sorted({f for x in originals for f in x['answer_faq_ids']})}
        # Absence examples are provisional. They may NOT drive threshold fitting
        # until a reviewer establishes absence within this corpus snapshot.
        extra.append(q)
        hits=[{'faq_id':d['faq_id'],'question':d['question'],
            'matched_terms':[t for t in terms.split(',') if t.casefold() in (d['question']+' '+d['answer']).casefold()]}
            for d in catalog if d['brand']==DATASETS[key]]
        audit.append({'question_id':qid,'dataset_id':key,'status':status,
            'unresolved_information':missing,'search_terms':terms.split(','),
            'corpus_scanned':sum(d['brand']==DATASETS[key] for d in catalog),
            'term_match_candidates':[x for x in hits if x['matched_terms']],
            'absence_review':'pending_semantic_review; keyword matches/absence are not proof'})
    write(ROOT/'data/absence_audit.jsonl',audit)
    return cases+extra,facts

def assign_splits(cases):
    # Keep a whole original category together, then merge if any known
    # acceptable document joins categories. This is intentionally conservative.
    g=Groups()
    for q in cases:
        family=q['policy_family']
        for fid in q['answer_faq_ids']+q.get('leakage_anchor_ids',[]):g.union(family,fid)
        for f in q['required_facts']:g.union(family,f['fact_id'])
    grouped=defaultdict(list)
    for q in cases:grouped[g.find(q['policy_family'])].append(q)
    totals=Counter(); n=len(cases)
    # Stable deterministic group allocation, never uses retrieval scores.
    for group,rows in sorted(grouped.items(),key=lambda kv:(-len(kv[1]),token(kv[0]))):
        split=max(('development','calibration','holdout'),key=lambda s:({'development':.5,'calibration':.25,'holdout':.25}[s]*n-totals[s]))
        for q in rows:q.update(split=split,leakage_group_id='G3-'+token(group),aggregate_unit='G3-'+token(group))
        totals[split]+=len(rows)

def build():
    catalog=read(ROOT/'authoring/corpus_catalog.jsonl')
    cases,facts=authored({d['faq_id']:d for d in catalog})
    cases=annotate_near_misses(cases,catalog)
    cases=add_composites(add_controls(cases))
    cases,facts=add_status_questions(cases,facts,catalog)
    unique_facts={}
    for f in facts:
        if f['fact_id'] in unique_facts:assert unique_facts[f['fact_id']]==f
        unique_facts[f['fact_id']]=f
    facts=list(unique_facts.values())
    legacy=read(ROOT.parent/'data/corpus.jsonl')
    old_cases=read(ROOT.parent/'fair_eval_v2/data/cases.jsonl')
    old_facts=read(ROOT.parent/'fair_eval_v2/data/facts.jsonl')
    for q in old_cases:
        q.update(dataset_id='existing',question_id=q['case_id'],user_question=q['query'],answer_faq_ids=q['expected_faq_ids'])
    for f in old_facts:f['dataset_id']='existing'
    manifest={'schema_version':3,'dataset_ids':list(DATASETS),'datasets':{},
        'scope_rule':'Select one dataset BEFORE embedding. Never embed a pooled four-dataset corpus.',
        'source_snapshot_sha256':sha(ROOT/'source/faqs.jsonl'),
        'independent_review_status':'pending','serving_runs_performed':False,
        'threshold_fitting_ready':False,
        'reason':'Provisional new absence labels and alternate-answer recall require independent review.'}
    for key,brand in DATASETS.items():
        docs=legacy if key=='existing' else [d for d in catalog if d['brand']==brand]
        qs=old_cases if key=='existing' else [q for q in cases if q['dataset_id']==key]
        fs=old_facts if key=='existing' else [f for f in facts if f['dataset_id']==key]
        if key!='existing':assign_splits(qs)
        out=ROOT/'datasets'/key
        write(out/'faq_pairs.jsonl',[dict(d,dataset_id=key) for d in docs])
        write(out/'user_questions.jsonl',qs);write(out/'facts.jsonl',fs)
        for split in ('development','calibration','holdout'):
            selected=[q for q in qs if q['split']==split]
            write(out/'splits'/f'{split}.jsonl',selected)
            write(out/'inputs'/f'{split}.jsonl',[{'question_id':q['question_id'],'query':q['model_query']} for q in selected])
        manifest['datasets'][key]={'brand':brand,'faq_count':len(docs),'question_count':len(qs),
            'faq_pairs':str((out/'faq_pairs.jsonl').relative_to(ROOT)).replace('\\','/'),
            'user_questions':str((out/'user_questions.jsonl').relative_to(ROOT)).replace('\\','/'),
            'corpus_sha256':sha(out/'faq_pairs.jsonl'),'questions_sha256':sha(out/'user_questions.jsonl'),
            'split_counts':dict(Counter(q['split'] for q in qs)),
            'purpose_counts':dict(Counter(q['primary_purpose'] for q in qs)),
            'status_counts':dict(Counter(q['corpus_status'] for q in qs)),
            'language_counts':dict(Counter(q['language'] for q in qs)),
            'leakage_group_count':len({q['leakage_group_id'] for q in qs}),
            'authored_base_question_count':sum(q['question_id'].startswith('Q3-') for q in qs),
            'covered_source_faqs':len({x for q in qs for x in q['source_faq_ids']}),
            'multi_answer_questions':sum(len(q['answer_faq_ids'])>1 for q in qs)}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:{f:v[f] for f in ['faq_count','question_count','split_counts','multi_answer_questions']} for k,v in manifest['datasets'].items()},ensure_ascii=False,indent=2))

if __name__=='__main__':build()
