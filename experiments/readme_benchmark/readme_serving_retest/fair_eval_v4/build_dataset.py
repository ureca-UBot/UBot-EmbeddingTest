"""Four isolated datasets with the same README evaluation contract.

V3 is immutable input. No serving result is used for authoring or selection.
"""
import copy,hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
V3=ROOT.parent/'fair_eval_v3'
KEYS=('existing','kt','skt','lgu')
BRANDS={'existing':None,'kt':'KT','skt':'SKT','lgu':'LG U+'}
PURPOSES=('PARAPHRASE','EXACT_ENTITY','KR_EN_EQUIVALENT','TYPO','LONG_QUERY','NEGATION',
 'NUMERIC_CONDITION','CONDITION_SCOPE','HARD_NEGATIVE','COMPARISON','MULTI_INTENT',
 'NO_FAQ','PARTIAL','AMBIGUOUS_QUERY','CONTEXT_RETRIEVAL')
read=lambda p:[json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
token=lambda s:hashlib.sha256(s.encode()).hexdigest()[:16]
norm=lambda s:re.sub(r'[\W_]','',s.casefold())
def write(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8',newline='\n')
def save(p,data):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

class Groups:
 def __init__(self):self.p={}
 def find(self,x):
  self.p.setdefault(x,x)
  if self.p[x]!=x:self.p[x]=self.find(self.p[x])
  return self.p[x]
 def union(self,a,b):
  a,b=self.find(a),self.find(b);self.p[max(a,b)]=min(a,b)

def load():
 docs={k:read(V3/'datasets'/k/'faq_pairs.jsonl') for k in KEYS}
 qs={k:read(V3/'datasets'/k/'user_questions.jsonl') for k in KEYS}
 facts={k:read(V3/'datasets'/k/'facts.jsonl') for k in KEYS}
 refs={d['authoring_id']:d['faq_id'] for d in read(V3/'authoring/queue.jsonl')}
 refs.update({f'L{i:03}':d['faq_id'] for i,d in enumerate(docs['lgu'],1)})
 return docs,qs,facts,refs

def normalize(qs,facts):
 corrections=[]
 for key in KEYS:
  fm={f['fact_id']:f for f in facts[key]}
  for q in qs[key]:
   q['source_case_id']=q['question_id'];q['source_dataset_version']='v3'
   unsupported=copy.deepcopy(q.get('unsupported_facts',[]));supported=[]
   for f in q['required_facts']:
    if f['acceptable_faq_ids']:supported.append(f)
    else:unsupported.append({'missing_fact_id':f['fact_id'],
       'requested_fact':fm[f['fact_id']].get('requested_information',''),
       'full_corpus_absence_review':'pending','subtype':'UNSUPPORTED_STATIC_INFORMATION'})
   q['required_facts']=supported;q['unsupported_facts']=unsupported
   q['tool_requirements']=[]
   # These records explicitly request a personal value or live state, not FAQ
   # instructions. Do not turn absent account access into a NO_FAQ example.
   live=any(word in q.get('unresolved_information','') for word in ('실제','실시간','특정 고객','현재 잔액','현재 위치','남은 약정'))
   if key!='existing' and unsupported and live:
    before=q['corpus_status'];q['tool_requirements']=unsupported;q['unsupported_facts']=[]
    q['corpus_status']='FAQ_EXISTS' if supported else 'NOT_APPLICABLE'
    q['primary_purpose']='TOOL_LOOKUP';q['purpose_tags']=['TOOL_LOOKUP']
    q['evaluation_track']='tool_lookup';q['expected_action']='ANSWER_FAQ_AND_LOOKUP' if supported else 'LOOKUP_REQUIRED'
    corrections.append({'question_id':q['question_id'],'dataset_id':key,'before':before,
       'after':q['corpus_status'],'reason':'Personal/live lookup is separate from FAQ absence'})
   elif q['corpus_status']=='PARTIAL_FACTS_EXIST':q['expected_action']='ANSWER_SUPPORTED_PART_AND_STATE_LIMIT'
   elif q['corpus_status']=='NO_FAQ':q['expected_action']='ABSTAIN_UNSUPPORTED'
   elif q['corpus_status']=='UNDERSPECIFIED':q['expected_action']='CLARIFY'
   else:q['expected_action']='ANSWER'
   if unsupported and not live and any(not f['acceptable_faq_ids'] for f in read_case_required(key,q['question_id'])):
    corrections.append({'question_id':q['question_id'],'dataset_id':key,
       'reason':'Unsupported facts removed from retrieval recall denominator'})
   q['retrieval_scope']='faq_subtasks_only'
   q['binary_fit_eligible']=False  # Independent semantic review is still pending.
   q['eligible_for_final_benchmark']=False
   q['cue_location']='unclassified'
   q['evidence_layout']='multi_fact' if len(supported)>1 else ('single_fact' if supported else 'no_retrieval_target')
 write(ROOT/'data/label_corrections.jsonl',corrections)

_original_required={}
def read_case_required(key,qid):return _original_required[(key,qid)]

def add_cases(docs,qs,facts,refs):
 path=ROOT/'authoring/additions.jsonl'
 if not path.exists():return
 dm={d['faq_id']:d for ds in docs.values() for d in ds}
 byid={k:{q['question_id']:q for q in rows} for k,rows in qs.items()}
 for a in read(path):
  key=a['dataset_id'];qid=a['question_id'];purpose=a['primary_purpose'];query=a['query']
  bases=[byid[key][x] for x in a.get('base_ids',[])]
  required={f['fact_id']:copy.deepcopy(f) for q in bases for f in q['required_facts']}
  source_ids={x for q in bases for x in q['source_faq_ids']}
  for i,s in enumerate(a.get('facts',[]),1):
   ids=sorted({refs.get(x,x) for x in s['faq_refs']});fid=qid+f'-F{i}'
   assert all(x in {d['faq_id'] for d in docs[key]} for x in ids)
   support=[{'faq_id':x,'quotes':s.get('quotes',{}).get(x,[dm[x]['answer']]),'review':'author_source_check'} for x in ids]
   for sp in support:assert all(t in dm[sp['faq_id']]['answer'] for t in sp['quotes'])
   facts[key].append({'fact_id':fid,'dataset_id':key,'requested_information':s['requested_information'],
      'acceptable_faq_ids':ids,'support':support,'independent_support_review':'pending'})
   required[fid]={'fact_id':fid,'acceptable_faq_ids':ids};source_ids.update(ids)
  if purpose in ('AMBIGUOUS_QUERY','NO_FAQ'):required={}
  ids=sorted({x for f in required.values() for x in f['acceptable_faq_ids']})
  missing=a.get('missing_information')
  unsupported=[{'missing_fact_id':a.get('missing_fact_id',qid+'-U'),
       'requested_fact':missing,'subtype':'NEAR_DOMAIN','full_corpus_absence_review':'pending',
       'absence_rationale':a.get('absence_rationale','Needs independent corpus-wide review')}] if missing and purpose in ('NO_FAQ','PARTIAL') else []
  status={'NO_FAQ':'NO_FAQ','PARTIAL':'PARTIAL_FACTS_EXIST','AMBIGUOUS_QUERY':'UNDERSPECIFIED'}.get(purpose,'FAQ_EXISTS')
  q={'question_id':qid,'case_id':qid,'dataset_id':key,'query':query,'user_question':query,
     'model_query':query,'answer_faq_ids':ids,'expected_faq_ids':ids,'required_facts':list(required.values()),
     'unsupported_facts':unsupported,'tool_requirements':[],'source_faq_ids':sorted(source_ids),
     'category':a.get('category',bases[0]['category'] if bases else 'author_reviewed_challenge'),
     'primary_purpose':purpose,'purpose_tags':a.get('purpose_tags',[purpose]),'corpus_status':status,
     'language':a.get('language','ko'),'dialogue_history':a.get('dialogue_history',[]),
     'review_status':'independent_review_pending','eligible_for_final_benchmark':False,
     'binary_fit_eligible':False,'retrieval_scope':'faq_subtasks_only','origin':'new_assistant_authored_synthetic',
     'evaluation_track':'core' if status=='FAQ_EXISTS' else 'answerability',
     'expected_action':{'NO_FAQ':'ABSTAIN_UNSUPPORTED','PARTIAL':'ANSWER_SUPPORTED_PART_AND_STATE_LIMIT',
        'AMBIGUOUS_QUERY':'CLARIFY'}.get(purpose,'ANSWER'),
     'cue_location':'unclassified','evidence_layout':'multi_fact' if len(required)>1 else ('single_fact' if required else 'no_retrieval_target'),
     'component_question_ids':a.get('base_ids',[]),'missing_information':missing,
     'absence_search_terms':a.get('absence_search_terms',[]),
     'interpretation_candidate_ids':[refs.get(x,x) for x in a.get('interpretation_refs',[])],
     'coverage_anchor':a.get('coverage_anchor'),
     'near_miss_candidates':[{'faq_id':refs.get(n['ref'],n['ref']),'reason':n['reason'],'independent_review':'pending'} for n in a.get('near_misses',[])]}
  if purpose=='CONTEXT_RETRIEVAL':
   assert len(bases)==1 and q['dialogue_history'];q['model_query']=bases[0]['model_query'];q['evaluation_track']='context'
  q['pair_ids']=a.get('pair_ids',[])
  qs[key].append(q);byid[key][qid]=q

def aliases_and_units(docs,qs,facts):
 graphs={}
 for key in KEYS:
  g=Groups()
  for d in docs[key]:g.find(d['faq_id'])
  for f in facts[key]:
   ids=f['acceptable_faq_ids']
   for x in ids[1:]:g.union(ids[0],x)
  graphs[key]=g
  for q in qs[key]:
   roots=sorted({g.find(x) for f in q['required_facts'] for x in f['acceptable_faq_ids']})
   if q['primary_purpose']=='NO_FAQ':roots=['absence:'+u['missing_fact_id'] for u in q['unsupported_facts']]
   elif q['primary_purpose']=='AMBIGUOUS_QUERY':
    roots=['ambiguous:'+str(q.get('coverage_anchor') or q.get('missing_information') or q.get('unresolved_information') or q['question_id'])]
    # V2 uses a generic missing_information string; original pair identifies the different missing context.
    if q.get('pair_ids'):roots=['ambiguous:'+q['pair_ids'][0]]
   q['coverage_units']=[key+':'+r for r in roots]
 return graphs

def enrich(qs):
 panels=read(ROOT/'authoring/cue_panels.jsonl')
 # Explicit counterpart documents for inherited V2 contrast questions. Shared
 # acceptable documents (e.g. FAQs covering both products) are not negatives.
 existing_negatives={
  'D-sim-direction-a':('FAQ-325','실물 SIM→eSIM과 eSIM→실물 SIM의 전환 방향이 다르다.'),
  'D-sim-direction-b':('FAQ-324','전환 후 비활성화되는 SIM을 방향에 맞게 구분해야 한다.'),
  'D-points-change-subject-a':('FAQ-845','번호변경과 명의변경은 포인트 승계 정책이 다르다.'),
  'D-points-change-subject-b':('FAQ-843','같은 가입자의 번호변경 정책으로 명의변경을 답할 수 없다.'),
  'D-subscription-price-a':('FAQ-762','음악 구독팩 가격으로 영상 구독팩 가격을 답할 수 없다.'),
  'D-subscription-price-b':('FAQ-758','영상 구독팩 가격은 음악 구독팩 가격과 다르다.'),
  'D-family-size-a':('FAQ-864','3회선 결합 할인액을 2회선에 적용할 수 없다.'),
  'D-family-size-b':('FAQ-863','2회선 결합 할인액과 3회선 결합 할인액은 다르다.'),
  'D-account-change-a':('FAQ-843','같은 통신사 번호변경의 포인트 유지와 타사 번호이동은 다르다.'),
  'D-account-change-b':('FAQ-254','타사 번호이동 포인트 미승계는 같은 통신사 번호변경 조건이 아니다.'),
 }
 lookup=defaultdict(list)
 for p in panels:lookup[p['question_id']].append(p)
 for key,rows in qs.items():
  for q in rows:
   if q['question_id'] in existing_negatives:
    fid,reason=existing_negatives[q['question_id']]
    q['near_miss_candidates']=[{'faq_id':fid,'reason':reason,'independent_review':'pending','origin':'author_source_contrast'}]
   q['diagnostic_panels']=lookup[q['question_id']]
   if q['diagnostic_panels']:q['cue_location']=q['diagnostic_panels'][0]['panel']
   fs=[set(f['acceptable_faq_ids']) for f in q['required_facts']]
   q['evidence_layout']='distributed_documents' if len(fs)>1 and not set.intersection(*fs) else ('multiple_facts_one_document_possible' if len(fs)>1 else ('single_fact' if fs else 'no_retrieval_target'))
   if q['primary_purpose']=='CONTEXT_RETRIEVAL':
    assert q['dialogue_history']
    # A label-blind, frozen context packing rule. This is not an LLM rewrite.
    history='\n'.join(x['role']+': '+x['content'] for x in q['dialogue_history'])
    q['model_query']='이전 대화:\n'+history+'\n현재 질문:\n'+q['query']
    q['rewrite_method']='context_concat_v1'
    q['rewrite_input_sha256']=hashlib.sha256(json.dumps([q['dialogue_history'],q['query']],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    q['evaluation_track']='context'
   if q['primary_purpose'] in ('HARD_NEGATIVE','EXACT_ENTITY'):
    q['evaluation_track']='special_conditions'
   q['binary_answerability_target']=(1 if q['corpus_status']=='FAQ_EXISTS' and not q['tool_requirements'] else (0 if q['corpus_status']=='NO_FAQ' else None))
   q['whole_request_supported_by_corpus']=q['corpus_status']=='FAQ_EXISTS' and not q['tool_requirements'] and not q['unsupported_facts']

def assign_splits(qs):
 for key,rows in qs.items():
  g=Groups()
  for q in rows:
   qid=q['question_id'];g.find(qid)
   links=q['answer_faq_ids']+[f['fact_id'] for f in q['required_facts']]+q.get('pair_ids',[])
   links += [u['missing_fact_id'] for u in q['unsupported_facts']]
   links += q.get('component_question_ids',[])
   if q.get('control_of'):links.append(q['control_of'])
   # Preserve reviewed V2 policy families; V3 source category is not a policy.
   links += ['policy:'+x for x in q.get('policy_families',[])]
   for x in links:g.union(qid,x)
  grouped=defaultdict(list)
  for q in rows:grouped[g.find(q['question_id'])].append(q)
  totals=Counter();n=len(rows)
  for group,subset in sorted(grouped.items(),key=lambda x:(-len(x[1]),token(x[0]))):
   split=max(('development','calibration','holdout'),key=lambda s:dict(development=.5,calibration=.25,holdout=.25)[s]*n-totals[s])
   for q in subset:q.update(split=split,leakage_group_id=key+':G4-'+token(group),aggregate_unit=key+':G4-'+token(group))
   totals[split]+=len(subset)

def coverage(qs):
 result={}
 for key,rows in qs.items():
  report={}
  for purpose in PURPOSES:
   selected=[q for q in rows if purpose in q['purpose_tags']]
   units={x for q in selected for x in q['coverage_units']}
   # Reusing the same requested fact set or a translated question does not add a scenario.
   scenarios={tuple(q['coverage_units']) for q in selected if q['coverage_units']}
   report[purpose]={'rows':len(selected),'evidence_units':len(units),'distinct_scenarios':len(scenarios),
       'minimum':10,'meets_authoring_minimum':len(scenarios)>=10 and len(units)>=10,
       'statistical_independence_claimed':False}
  result[key]=report
 return result

def build():
 docs,qs,facts,refs=load()
 for key in KEYS:
  for q in qs[key]:_original_required[(key,q['question_id'])]=copy.deepcopy(q['required_facts'])
 normalize(qs,facts);add_cases(docs,qs,facts,refs);enrich(qs);aliases_and_units(docs,qs,facts);assign_splits(qs)
 for key in KEYS:
  used={f['fact_id'] for q in qs[key] for f in q['required_facts']}
  facts[key]=[f for f in facts[key] if f['fact_id'] in used and f['acceptable_faq_ids']]
 cov=coverage(qs);manifest={'schema_version':4,'dataset_ids':list(KEYS),'datasets':{},
   'scope_rule':'Choose one dataset before corpus embedding; never pool carriers.',
   'source_v3_manifest_sha256':sha(V3/'manifest.json'),'serving_runs_performed':False,
   'independent_review_status':'pending','threshold_fitting_ready':False}
 for key in KEYS:
  out=ROOT/'datasets'/key
  write(out/'faq_pairs.jsonl',docs[key]);write(out/'user_questions.jsonl',qs[key]);write(out/'facts.jsonl',facts[key])
  for split in ('development','calibration','holdout'):
   selected=[q for q in qs[key] if q['split']==split]
   write(out/'splits'/f'{split}.jsonl',selected)
   write(out/'inputs'/f'{split}.jsonl',[{'question_id':q['question_id'],'query':q['model_query']} for q in selected])
  manifest['datasets'][key]={'brand':BRANDS[key],'faq_count':len(docs[key]),'question_count':len(qs[key]),
    'corpus_sha256':sha(out/'faq_pairs.jsonl'),'questions_sha256':sha(out/'user_questions.jsonl'),
    'split_counts':dict(Counter(q['split'] for q in qs[key])),
    'purpose_counts':dict(Counter(q['primary_purpose'] for q in qs[key])),
    'status_counts':dict(Counter(q['corpus_status'] for q in qs[key])),
    'leakage_group_count':len({q['leakage_group_id'] for q in qs[key]}),
    'readme_core_coverage_complete':all(v['meets_authoring_minimum'] for v in cov[key].values())}
 save(ROOT/'manifest.json',manifest);save(ROOT/'data/coverage_matrix.json',cov)
 diagnostics={}
 for key,rows in qs.items():
  diagnostics[key]={
   'question_cue_rows':sum(q['cue_location']=='question_cue' for q in rows),
   'answer_only_source_cue_rows':sum(q['cue_location']=='answer_only_source_cue' for q in rows),
   'distributed_document_rows':sum(q['evidence_layout']=='distributed_documents' for q in rows),
   'authored_hard_negative_rows':sum(bool(q.get('near_miss_candidates')) for q in rows),
   'split_purposes':{s:{p:sum(p in q['purpose_tags'] for q in rows if q['split']==s) for p in PURPOSES} for s in ('development','calibration','holdout')},
   'source_category_counts':dict(Counter(d['category'] for d in docs[key])),
   'question_category_counts':dict(Counter(q['category'] for q in rows)),
   'source_faqs_used_as_gold':len({x for q in rows for x in q['answer_faq_ids']}),
   'largest_leakage_group_rows':max(Counter(q['leakage_group_id'] for q in rows).values()),
  }
 save(ROOT/'data/diagnostic_coverage.json',diagnostics)
 for k,v in cov.items():print(k,'GAPS',{p:c['distinct_scenarios'] for p,c in v.items() if not c['meets_authoring_minimum']})

if __name__=='__main__':build()
