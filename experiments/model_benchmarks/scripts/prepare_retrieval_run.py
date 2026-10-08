"""Freeze the source-labelled exploration set before any model scores exist."""
from __future__ import annotations
import json, hashlib, shutil, re
from collections import Counter, defaultdict
from pathlib import Path
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'faq/retrieval_run_v2'

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def save_lines(path, data):
    path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in data), encoding='utf-8')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if (OUT / 'split_manifest.json').exists():
        raise SystemExit('Frozen dataset already exists. Use its manifest; do not rebuild after scores.')
    OUT.mkdir(parents=True, exist_ok=True)
    sources = json.loads((ROOT / 'configs/retrieval_source_audit.json').read_text(encoding='utf-8'))
    paths = json.loads((ROOT / 'configs/container_source_paths.json').read_text(encoding='utf-8'))
    snapshots = OUT / 'sources'; snapshots.mkdir(exist_ok=True)
    for src in sources['sources']:
        path = Path(paths[src['key']])
        if digest(path) != src['sha256']: raise ValueError('Source hash mismatch: '+src['key'])
        shutil.copy2(path, snapshots / (src['key']+'.xlsx'))
    wb = openpyxl.load_workbook(snapshots/'current_retrieval_base.xlsx', read_only=True, data_only=True)
    rows = list(wb['FAQ 원문'].iter_rows(values_only=True))
    corpus = [{'faq_id':r[0], 'category':r[1], 'question':r[2], 'answer':r[3], 'intent':r[4]} for r in rows[1:] if r[0]]
    for d in corpus: d['body_hash']=hashlib.sha256((d['question']+'\n'+d['answer']).encode()).hexdigest()
    save_lines(OUT/'corpus.jsonl', corpus)
    faq = {d['faq_id']:d for d in corpus}
    raw = list(wb['테스트 3000건'].iter_rows(values_only=True))
    headers = raw[0]; source_rows = [dict(zip(headers,r)) for r in raw[1:] if r[0]]
    save_lines(OUT/'source_rows.jsonl', source_rows)
    fixtures = [dict(zip(['condition_id','faq_id','base_faq_id','question','answer','priority','valid_from','valid_until','scope'],r)) for r in wb['시험 문서'].iter_rows(min_row=2,values_only=True) if r[1]]
    save_lines(OUT/'fixture_documents.jsonl', fixtures)
    conditions = list(wb['실행 조건'].iter_rows(values_only=True))
    save_lines(OUT/'execution_conditions.jsonl',[dict(zip(conditions[0],r)) for r in conditions[1:] if r[0]])
    cases=[]; excluded=Counter()
    for rowno, r in enumerate(source_rows,2):
        mode=r['실행 모드']; route=r['기대 처리 경로']; category=r['항목 코드']
        gt=json.loads(r['검색 정답 FAQ 집합 (JSON)'] or '[]')
        history=json.loads(r['대화 이력 (JSON)'] or '[]')
        if mode!='실제 검색': excluded['special_or_controlled']=excluded['special_or_controlled']+1;continue
        if 'FAQ_RAG' not in str(route):excluded['non_faq_route']+=1;continue
        if not gt and category not in ['SR','AD']:excluded['unresolved_gt']+=1;continue
        if any(fid not in faq for g in gt for fid in g):raise ValueError(r['실행 ID']+' unknown FAQ')
        facts=[{'fact_id':f'f{i+1}','requirement':str(r['필수 사실·표현 기준'] or '원천의 검색 근거'), 'primary_ids':g,'acceptable_ids':[]} for i,g in enumerate(gt)]
        typ='NO_FAQ' if not gt else ('CONDITION_SCOPE' if category=='CE' else 'STANDARD')
        track='repeat' if category=='RT' else ('contextual' if history else ('no_faq' if not gt else 'core'))
        cohort={'repeat':'repeat','contextual':'contextual','no_faq':'no_faq','core':'targeted_condition' if typ=='CONDITION_SCOPE' else 'core_general'}[track]
        # Conservative grouping by complete gold signature; shared single facts stay together.
        signature=json.dumps(sorted([sorted(g) for g in gt]),ensure_ascii=False,separators=(',',':')) if gt else str(r['시나리오 그룹']).split(':',1)[-1]
        family='src-'+hashlib.sha256(signature.encode()).hexdigest()[:16]
        case={'case_id':'SOURCE-'+r['실행 ID'],'dataset_version':'retrieval-run-v1-provisional','corpus_version':'current-faq-1024-v1','family_id':family,'split_group_id':family,'split':'calibration','track':track,'primary_type':typ,'tags':[typ]+(['MULTI_FACT'] if len(gt)>1 else [])+(['CONTEXTUAL'] if history else []),'reporting_cohort':cohort,'source':{'origin':'workbook','source_key':'current_retrieval_base','sheet':'테스트 3000건','row':rowno,'execution_id':r['실행 ID'],'original_question_id':r['원본 질문 ID'],'scenario_group':r['시나리오 그룹'],'category':category,'service_log_ref':None},'query':r['사용자 질문'],'history':history,'query_mode':'frozen_contextual_rewrite' if history else 'standalone','answerability':'none' if not gt else ('partial' if r['기대 상태 (실행조건 충족 시)']=='PARTIAL' else 'full'),'retrieval_status':'NO_FAQ' if not gt else ('PARTIAL_FACTS_EXIST' if r['기대 상태 (실행조건 충족 시)']=='PARTIAL' else 'FAQ_EXISTS'),'expected_generation_status':r['기대 상태 (실행조건 충족 시)'] if r['기대 상태 (실행조건 충족 시)'] in ['ANSWER','PARTIAL','ABSTAIN','CLARIFY','ROUTE'] else 'NOT_EVALUATED','expected_routes':['FAQ_RAG'],'execution_condition_id':r['실행조건 ID'] or 'NORMAL','fact_groups':facts,'hard_negative_ids':[],'annotation':{'status':'pending','reviewer':None,'rationale':'原천 라벨과 FAQ 참조를 보존한 잠정 실험셋. 독립 도메인 검수 전 결과로 표시한다.'}}
        if category=='RT':case['repeat']={'group_id':r['원본 질문 ID'],'index':r['실행 회차'],'total':r['총 반복 횟수']}
        case['dataset_version']='retrieval-run-v2-provisional'
        routes=[x.strip() for x in route.split('+')]
        if len(routes)>1:
            case['primary_type']='MULTI_INTENT'
            case['tags'].append('MULTI_INTENT')
            case['expected_routes']=routes
            case['subrequests']=[{'request_id':f'r{i+1}','route':value,'requirement':'FAQ 하위 사실 검색' if value=='FAQ_RAG' else '제공된 개인 정보 또는 외부 조회로 처리하는 하위 요구','retrieval_fact_ids':[g['fact_id'] for g in facts] if value=='FAQ_RAG' else [],'external_action_required':value!='FAQ_RAG'} for i,value in enumerate(routes)]
        cases.append(case)
    wb.close()
    # Explicit union of source IDs, duplicate text, scenario labels and semantic groups.
    parent={c['case_id']:c['case_id'] for c in cases}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    groups={}
    for c in cases:
        keys=[('family',c['family_id']),('original',c['source']['original_question_id']),('scenario',c['source']['scenario_group']),('query',re.sub(r'\s+',' ',c['query']).strip())]
        for key in keys:
            if not key[1]:continue
            other=groups.setdefault(key,c['case_id']);a,b=find(c['case_id']),find(other)
            if a!=b:parent[max(a,b)]=min(a,b)
    for c in cases:
        component=find(c['case_id']);c['split_group_id']=component
        score=int(hashlib.sha256(('20261006:'+component).encode()).hexdigest()[:8],16)/0xffffffff
        c['split']='holdout' if score<0.25 else 'calibration'
    # Preserve established examples as a separate diagnostic slice, never in holdout.
    examples=[json.loads(x) for x in (ROOT/'faq/retrieval_design_examples.jsonl').read_text(encoding='utf-8').splitlines()]
    save_lines(OUT/'cases.jsonl',cases)
    save_lines(OUT/'diagnostic_examples.jsonl',examples)
    legacy=[]
    for key in ['legacy_60','legacy_235']:
        w=openpyxl.load_workbook(snapshots/(key+'.xlsx'),read_only=True,data_only=True)
        cr=[{'faq_id':r[0],'question':r[2],'answer':r[3]} for r in w['FAQ'].iter_rows(min_row=2,values_only=True) if r[0]]
        qs=[]
        for sn in w.sheetnames:
            if not sn.startswith('Retrieval'):continue
            rr=list(w[sn].iter_rows(values_only=True));hh=rr[0]
            for row in rr[1:]:
                if not row[0]:continue
                entry=dict(zip(hh,row));qs.append({'sheet':sn,'values':entry})
        save_json(OUT/(key+'.json'),{'corpus':cr,'queries':qs});w.close()
    save_json(OUT/'split_manifest.json',{'dataset_version':'retrieval-run-v2-provisional','label_status':'source_labels_pending_independent_review','holdout_status':'provisional_source_label_holdout_not_production_certification','split_seed':20261006,'corpus_count':len(corpus),'case_count':len(cases),'split_counts':dict(Counter(c['split'] for c in cases)),'cohort_counts':dict(Counter(c['reporting_cohort'] for c in cases)),'family_count':len(set(c['family_id'] for c in cases)),'component_count':len(set(c['split_group_id'] for c in cases)),'excluded_source_rows':dict(excluded),'frozen_before_model_results':True,'hashes':{p.name:digest(p) for p in OUT.glob('*.jsonl')}})
    print((OUT/'split_manifest.json').read_text(encoding='utf-8'),flush=True)

if __name__=='__main__':main()
