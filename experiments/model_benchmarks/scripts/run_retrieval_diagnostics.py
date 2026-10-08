"""Special source contracts, detailed meaning pairs, and actual repeat runs."""
from __future__ import annotations
import json,re,time
from collections import defaultdict,Counter
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from retrieval_common import fact_metrics,stable_hash
from run_retrieval_benchmark import OUT,DATA,lines,save,save_lines,model_path,text,rank,query,setup,sync

def main():
    setup()
    corpus=lines(DATA/'corpus.jsonl');ids=[d['faq_id'] for d in corpus];docmap={d['faq_id']:d for d in corpus}
    model=SentenceTransformer(model_path('BAAI/bge-m3'),device='cuda',model_kwargs={'dtype':torch.float32});model.max_seq_length=512
    views={v:np.load(OUT/f'cache/dense-{v}-corpus.npy') for v in ['question','question_answer']}
    # Detailed pairs use grounded substitutions and are posthoc diagnostics, not tuning/holdout.
    mapping={'로밍':'roaming','유심':'USIM','데이터':'data','테더링':'tethering','와이파이':'Wi-Fi','스팸':'spam','번호이동':'number portability'}
    pairs=[];used=set()
    for ko,en in mapping.items():
        matches=[d for d in corpus if ko in d['question'] and d.get('intent')=='FAQ_RAG'][:25]
        for d in matches:
            if d['faq_id'] in used:continue
            used.add(d['faq_id']);pid=ko+'-'+d['faq_id']
            for variant,q in [('ko_anchor',d['question']),('en_term',d['question'].replace(ko,en))]:
                pairs.append({'pair_id':pid,'faq_id':d['faq_id'],'variant':variant,'query':q})
    examples=[c for c in lines(DATA/'diagnostic_examples.jsonl') if c['corpus_version']=='current-faq-1024-v1']
    qs=[r['query'] for r in pairs]+[query(c) for c in examples]
    qv=model.encode(qs,batch_size=16,normalize_embeddings=True)
    results=[]
    for view,dv in views.items():
        ss=qv@dv.T
        for i,r in enumerate(pairs):
            ordered=[ids[j] for j in rank(ss[i],ids)]
            results.append({**r,'view':view,'rank':ordered.index(r['faq_id'])+1,'hit_at_3':int(r['faq_id'] in ordered[:3]),'top_10':ordered[:10]})
        save_lines(OUT/f'diagnostics/design-examples-{view}.jsonl',[{'case_id':c['case_id'],'primary_type':c['primary_type'],'query':query(c),'metrics':fact_metrics([ids[j] for j in rank(ss[len(pairs)+i],ids)],c['fact_groups'],3)} for i,c in enumerate(examples)])
    save_lines(OUT/'diagnostics/language-pairs.jsonl',results)
    summaries={}
    for view in views:
        grouped=defaultdict(dict)
        for r in results:
            if r['view']==view:grouped[r['pair_id']][r['variant']]=r
        summaries[view]={'pairs':len(grouped),'pair_hit_at_3':float(np.mean([int(g['ko_anchor']['hit_at_3'] and g['en_term']['hit_at_3']) for g in grouped.values()])),'anchor_hit_at_3':float(np.mean([g['ko_anchor']['hit_at_3'] for g in grouped.values()])),'en_term_hit_at_3':float(np.mean([g['en_term']['hit_at_3'] for g in grouped.values()])),'mean_rank_delta':float(np.mean([g['en_term']['rank']-g['ko_anchor']['rank'] for g in grouped.values()])),'top10_overlap':float(np.mean([len(set(g['en_term']['top_10'])&set(g['ko_anchor']['top_10']))/10 for g in grouped.values()]))}
    save(OUT/'diagnostics/language-summary.json',{'status':'posthoc_grounded_diagnostic_not_holdout','substitution_dictionary':mapping,'views':summaries})
    # Run each raw HR/EC/CF/AD/RT fixture using its own isolated execution condition.
    source=lines(DATA/'source_rows.jsonl');fixtures=lines(DATA/'fixture_documents.jsonl')
    fixture_groups=defaultdict(list)
    for d in fixtures:fixture_groups[d['condition_id']].append(d)
    special=[r for r in source if r['실행 모드']!='실제 검색']
    qstrings=list(dict.fromkeys(r['사용자 질문'] for r in special));qemb=model.encode(qstrings,batch_size=16,normalize_embeddings=True);qi={q:i for i,q in enumerate(qstrings)}
    fdv=model.encode([d['question'] for d in fixtures],batch_size=16,normalize_embeddings=True);fv={d['faq_id']:fdv[i] for i,d in enumerate(fixtures)}
    special_results=[]
    for r in special:
        condition=r['실행조건 ID'];cat=r['항목 코드'];q=r['사용자 질문'];gt=json.loads(r['검색 정답 FAQ 집합 (JSON)'] or '[]')
        if condition=='EC_EMPTY':scope=[];vv=np.empty((0,1024),dtype=np.float32)
        elif condition=='HR_UNRELATED':scope=[f'FAQ-{i:03d}' for i in range(807,828)];vv=views['question'][[ids.index(i) for i in scope]]
        elif condition in fixture_groups:
            docs=fixture_groups[condition]
            if condition.startswith('CF-'):
                base=[f'FAQ-{i:03d}' for i in range(18,28)];scope=base+[d['faq_id'] for d in docs];vv=np.stack([views['question'][ids.index(fid)] for fid in base]+[fv[d['faq_id']] for d in docs])
            else:
                replaced={d['base_faq_id'] for d in docs};base=[fid for fid in ids if fid not in replaced];scope=base+[d['faq_id'] for d in docs];vv=np.stack([views['question'][ids.index(fid)] for fid in base]+[fv[d['faq_id']] for d in docs])
        else:
            special_results.append({'execution_id':r['실행 ID'],'condition':condition,'status':'UNSUPPORTED_SOURCE_CONDITION'});continue
        scores=qemb[qi[q]]@vv.T;ordered=[scope[j] for j in rank(scores,scope)[:3]]
        retrieved=[docmap[fid] if fid in docmap else next(d for d in fixtures if d['faq_id']==fid) for fid in ordered]
        context='\n\n'.join(text(d,'question_answer') for d in retrieved)
        gold=[{'fact_id':f'f{i}','primary_ids':g,'acceptable_ids':[]} for i,g in enumerate(gt)]
        reproduced=(len(ordered)==0 if condition=='EC_EMPTY' else len(ordered)>0 if condition=='HR_UNRELATED' else all(set(g)&set(ordered) for g in gt))
        special_results.append({'execution_id':r['실행 ID'],'source_category':cat,'condition':condition,'corpus_count':len(scope),'top_ids':ordered,'metrics':fact_metrics(ordered,gold,3),'retrieval_condition_reproduced':bool(reproduced),'context_hash':stable_hash(context),'attack_exposed_in_context':any(fid.startswith('TEST-AD') for fid in ordered),'actual_LLM_input_exposure_evaluated':False,'generation_guardrail_status':'NOT_EVALUATED_NO_GENERATION_MODEL_CONFIGURED','context':context if condition.startswith('AD-') else None})
    save_lines(OUT/'diagnostics/special-conditions.jsonl',special_results)
    save(OUT/'diagnostics/special-summary.json',{'execution_rows':len(special_results),'by_category':dict(Counter(r.get('source_category','unsupported') for r in special_results)),'retrieval_condition_reproduced':sum(r.get('retrieval_condition_reproduced',False) for r in special_results),'attack_exposed_in_context':sum(r.get('attack_exposed_in_context',False) for r in special_results),'actual_LLM_input_exposure_evaluated':False,'generation_evaluated':False})
    # Actual repeated encoding, not ten reads from the same cache.
    repeats=[r for r in source if r['항목 코드']=='RT' and r['실행 모드']=='실제 검색' and 'FAQ_RAG' in str(r['기대 처리 경로'])]
    repeat_results=[]
    for r in repeats:
        history=json.loads(r['대화 이력 (JSON)'] or '[]');q=r['사용자 질문']
        if history:q='대화 이력:\n'+'\n'.join(f"{h['role']}: {h['content']}" for h in history)+'\n현재 질문: '+q
        vector=model.encode([q],normalize_embeddings=True)[0];ss=vector@views['question'].T;ordered=[ids[j] for j in rank(ss,ids)[:20]]
        ctx='\n\n'.join(text(docmap[fid],'question_answer') for fid in ordered[:3])
        repeat_results.append({'execution_id':r['실행 ID'],'repeat_group':r['원본 질문 ID'],'run':r['실행 회차'],'query_hash':stable_hash(q),'top20_hash':stable_hash(ordered),'score_hash':stable_hash([float(ss[ids.index(fid)]) for fid in ordered]),'context_hash':stable_hash(ctx),'output_hash':None})
    save_lines(OUT/'diagnostics/repeats.jsonl',repeat_results)
    groups=defaultdict(list)
    for r in repeat_results:groups[r['repeat_group']].append(r)
    save(OUT/'diagnostics/repeat-summary.json',{'groups':len(groups),'executions':len(repeat_results),'rank_unstable_groups':sum(len(set(r['top20_hash'] for r in g))>1 for g in groups.values()),'context_unstable_groups':sum(len(set(r['context_hash'] for r in g))>1 for g in groups.values()),'score_unstable_groups':sum(len(set(r['score_hash'] for r in g))>1 for g in groups.values()),'generation_evaluated':False})
    print('Diagnostics completed',json.dumps(summaries,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
