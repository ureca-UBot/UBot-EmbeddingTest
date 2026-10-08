"""One frozen holdout pass plus separate contextual, repeat and attack probes."""
from __future__ import annotations
import argparse, hashlib, json, time, pickle
from collections import defaultdict
import numpy as np
import torch
from run_retrieval_benchmark import ROOT,OUT,DATA,lines,jsread,save,save_lines,sha,text,query,rank,model_path,sync,setup
from retrieval_common import fact_metrics,merge_dense_views

class CLSEncoder:
    """BGE's frozen CLS+L2 path without SentenceTransformer packaging imports."""
    def __init__(self):
        from transformers import AutoTokenizer,AutoModel
        self.tokenizer=AutoTokenizer.from_pretrained(model_path('BAAI/bge-m3'))
        self.model=AutoModel.from_pretrained(model_path('BAAI/bge-m3'),torch_dtype=torch.float32).to('cuda').eval()
        self.max_seq_length=512
    def encode(self,texts,batch_size=16,normalize_embeddings=True):
        values=[]
        for start in range(0,len(texts),batch_size):
            batch=self.tokenizer(texts[start:start+batch_size],padding=True,truncation=True,max_length=self.max_seq_length,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                vec=self.model(**batch).last_hidden_state[:,0,:].float()
                if normalize_embeddings:vec=torch.nn.functional.normalize(vec,p=2,dim=1)
            values.append(vec.cpu().numpy())
        return np.concatenate(values)

class Retriever:
    def __init__(self,selection):
        self.name=selection['selected'];self.config=selection
        self.base=lines(DATA/'corpus.jsonl')
        self.embed=CLSEncoder()
        self.views=['question'] if self.name=='A' else ['question_answer'] if self.name=='B' else ['question','question_answer']
        self.dv={v:np.load(OUT/f'cache/dense-{v}-corpus.npy') for v in self.views}
        self.reranker=None;self.native=None
        if self.name.startswith('D-'):
            from transformers import AutoTokenizer,AutoModelForSequenceClassification
            self.which='jina' if self.name.startswith('D-jina') else 'bge'
            self.rr_view='question_answer' if '-question_answer-' in self.name else 'question'
            m='jinaai/jina-reranker-v2-base-multilingual' if self.which=='jina' else 'BAAI/bge-reranker-v2-m3'
            self.reranker=AutoModelForSequenceClassification.from_pretrained(model_path(m),trust_remote_code=self.which=='jina',
                            torch_dtype=torch.float32,**({'use_flash_attn':False} if self.which=='jina' else {})).to('cuda').eval()
            self.tokenizer=AutoTokenizer.from_pretrained(model_path(m),trust_remote_code=self.which=='jina')
        elif self.name.startswith(('E-','F-')):
            from FlagEmbedding import BGEM3FlagModel
            del self.embed;torch.cuda.empty_cache()
            self.native=BGEM3FlagModel(model_path('BAAI/bge-m3'),use_fp16=False,devices='cuda:0',batch_size=16,query_max_length=512,passage_max_length=512)
            self.native_docs=pickle.loads((OUT/'cache/native-question_answer-corpus.pkl').read_bytes())
    def retrieve(self,texts,corpus=None):
        docs=corpus or self.base;ids=[d['faq_id'] for d in docs]
        if self.native is not None:
            return self.retrieve_native(texts,docs,ids,corpus is not None)
        dv=self.dv if corpus is None else {v:self.embed.encode([text(d,v) for d in docs],batch_size=16,normalize_embeddings=True) for v in self.views}
        q=self.embed.encode(texts,batch_size=16,normalize_embeddings=True)
        scores={v:q@dv[v].T for v in self.views};rows=[]
        for i,qt in enumerate(texts):
            if len(self.views)==1:
                ss=scores[self.views[0]][i];pool=rank(ss,ids)
            else:
                ss=np.maximum(scores['question'][i],scores['question_answer'][i])
                merged=merge_dense_views(dict(zip(ids,scores['question'][i])),dict(zip(ids,scores['question_answer'][i])),20,20)
                pool=[ids.index(k) for k in merged['top_m_ids']]
            if self.reranker is not None:
                pairs=[[qt,text(docs[j],self.rr_view)] for j in pool[:20]]
                with torch.inference_mode():
                    inputs=self.tokenizer(pairs,padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
                    val=self.reranker(**inputs).logits.reshape(-1).float().cpu().numpy()
                ss=np.full(len(ids),-1e20,dtype=np.float32)
                for j,v in zip(pool[:20],val):ss[j]=v
                pool=sorted(pool[:20],key=lambda j:(-ss[j],ids[j]))
            ordered=[ids[j] for j in pool[:20]];values=[float(ss[j]) for j in pool[:20]]
            rows.append({'top_ids':ordered,'top_scores':values,'query_hash':hashlib.sha256(qt.encode()).hexdigest()})
        return rows

    def retrieve_native(self,texts,docs,ids,fixture):
        dd=self.native_docs if not fixture else self.native.encode([text(d,'question_answer') for d in docs],batch_size=16,max_length=512,return_dense=True,return_sparse=True,return_colbert_vecs=True)
        qq=self.native.encode(texts,batch_size=16,max_length=512,return_dense=True,return_sparse=True,return_colbert_vecs=True)
        dense=qq['dense_vecs']@dd['dense_vecs'].T
        if self.name.startswith('E-'):
            from run_native_retrieval import sparse_matrix
            sparse=(sparse_matrix(qq['lexical_weights'])@sparse_matrix(dd['lexical_weights']).T).toarray()
            scores=np.zeros_like(dense);pools=[]
            for i,(ds,ss) in enumerate(zip(dense,sparse)):
                dr=rank(ds,ids);sr=rank(ss,ids)
                for rr in [dr,sr]:
                    for pos,j in enumerate(rr,1):scores[i,j]+=1/(60+pos)
                pools.append(sorted(set(dr[:20])|set(sr[:20]),key=lambda j:(-scores[i,j],ids[j]))[:20])
        else:
            order=sorted(range(len(ids)),key=lambda i:len(dd['colbert_vecs'][i]));buckets=[]
            for b in range(0,len(ids),32):
                ix=order[b:b+32];length=max(len(dd['colbert_vecs'][i]) for i in ix);dim=dd['colbert_vecs'][ix[0]].shape[1]
                vals=torch.zeros((len(ix),length,dim),device='cuda');mask=torch.zeros((len(ix),length),device='cuda',dtype=torch.bool)
                for j,k in enumerate(ix):
                    v=dd['colbert_vecs'][k];vals[j,:len(v)]=torch.as_tensor(v,device='cuda');mask[j,:len(v)]=True
                buckets.append((ix,vals,mask))
            scores=np.empty_like(dense)
            with torch.inference_mode():
                for i,q in enumerate(qq['colbert_vecs']):
                    qt=torch.as_tensor(q,device='cuda')
                    for ix,dt,mask in buckets:
                        token_scores=torch.einsum('qd,btd->bqt',qt,dt).masked_fill(~mask[:,None,:],-torch.inf)
                        scores[i,ix]=token_scores.max(dim=-1).values.mean(dim=-1).cpu().numpy()
            pools=[rank(s,ids) for s in scores] if self.name=='F-full' else [sorted(rank(d,ids)[:20],key=lambda j:(-s[j],ids[j])) for s,d in zip(scores,dense)]
        return [{'top_ids':[ids[j] for j in p[:20]],'top_scores':[float(s[j]) for j in p[:20]],
                 'query_hash':hashlib.sha256(t.encode()).hexdigest()} for p,s,t in zip(pools,scores,texts)]

def summary(cases,rows):
    output=[]
    for c,r in zip(cases,rows):
        score=r['top_scores'][0];margin=score-r['top_scores'][1]
        output.append({'case_id':c['case_id'],'family_id':c['family_id'],'cohort':c['reporting_cohort'],'type':c['primary_type'],
                       **r,'metrics':{str(k):fact_metrics(r['top_ids'],c['fact_groups'],k) for k in [1,3,5,10,20]},
                       'no_faq_truth':c['track']=='no_faq','predicted_no_faq':score<CONFIG['policy']['threshold'] or margin<CONFIG['policy']['margin']})
    cohorts={}
    for cohort in sorted({r['cohort'] for r in output}):
        rr=[r for r in output if r['cohort']==cohort and r['metrics']['3']['fact_recall'] is not None]
        groups=defaultdict(list)
        for r in rr:groups[r['family_id']].append(r)
        cohorts[cohort]={'rows':len(rr),'families':len(groups),'family_mean':
            {f'{m}@{k}':float(np.mean([np.mean([r['metrics'][str(k)][m] for r in g]) for g in groups.values()])) if groups else None
             for k in [1,3,5,10,20] for m in ['semantic_hit','fact_recall','all_facts_hit','semantic_mrr']}}
    truth=np.array([r['no_faq_truth'] for r in output]);pred=np.array([r['predicted_no_faq'] for r in output])
    tp=int(np.sum(truth&pred));fp=int(np.sum(~truth&pred));fn=int(np.sum(truth&~pred));tn=int(np.sum(~truth&~pred))
    pp=tp/(tp+fp) if tp+fp else 0;rr=tp/(tp+fn) if tp+fn else 0
    nofaq={'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':pp,'recall':rr,'f1':2*pp*rr/(pp+rr) if pp+rr else 0,
            'unsupported_question_acceptance':fn/(tp+fn) if tp+fn else None,'answerable_rejection':fp/(fp+tn) if fp+tn else None}
    return output,{'cohorts':cohorts,'no_faq':nofaq,'label_status':'provisional_authored'}

def main():
    global CONFIG
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['holdout','diagnostics'],required=True)
    p.add_argument('--resume-infrastructure-failure',action='store_true');args=p.parse_args()
    torch.manual_seed(20261006);np.random.seed(20261006)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    CONFIG=jsread(OUT/'selection.json')
    assert sha(DATA/'split_manifest.json')==CONFIG['dataset_manifest_sha256'],'Dataset changed after selection'
    target=OUT/'holdout/policy_results.json' if args.mode=='holdout' else OUT/'diagnostics/summary.json'
    marker=target.with_suffix('.started.json')
    if target.exists():raise SystemExit('Frozen evaluation completed; no second holdout look')
    prior=jsread(marker) if marker.exists() else None
    if prior and not args.resume_infrastructure_failure:raise SystemExit('Evaluation started; explicit infrastructure-failure resume required')
    if prior and not (OUT/'holdout/infrastructure_failure.json').exists():raise SystemExit('Missing failed-load evidence; cannot resume')
    save(marker,{'selection_hash':sha(OUT/'selection.json'),'started_at':time.time(),
                 'prior_infrastructure_attempt':prior,'resume_infrastructure_failure':bool(prior)})
    model=Retriever(CONFIG);all_cases=lines(DATA/'cases.jsonl')
    if model.native is None:
        probes=jsread(OUT/'cache/dense-calibration-query_texts.json')[:12]
        reference=np.load(OUT/'cache/dense-calibration-queries.npy')[:12]
        current=model.embed.encode(probes,batch_size=16,normalize_embeddings=True)
        cos=np.sum(reference*current,axis=1)/(np.linalg.norm(reference,axis=1)*np.linalg.norm(current,axis=1))
        bridge={'query_count':len(probes),'reference_to_runtime_cosine_min':float(cos.min()),
                'normalized_max_absolute_difference':float(np.max(np.abs(reference-current))),
                'passed':bool(cos.min()>=.9999),'calibration_metrics_used_for_bridge':False}
        save(OUT/f'{args.mode}/model_runtime_bridge.json',bridge)
        if not bridge['passed']:raise ValueError('Embedding runtime drift: do not evaluate with mismatched calibration vectors')
    if args.mode=='holdout':
        cases=[c for c in all_cases if c['split']=='holdout' and c['track'] in ['core','no_faq']]
        rows=model.retrieve([query(c) for c in cases]);details,result=summary(cases,rows)
        save_lines(OUT/'holdout/per_query.jsonl',details)
        save(target,{'selected':CONFIG['selected'],'selection_hash':sha(OUT/'selection.json'),'dataset_hash':sha(DATA/'split_manifest.json'),
                     'case_count':len(cases),'independent_review_complete':False,'holdout_passes':1,**result})
        print(json.dumps(jsread(target),ensure_ascii=False,indent=2));return
    result={'selected':CONFIG['selected'],'ordinary_average_contains_diagnostics':False}
    context=[c for c in all_cases if c['track']=='contextual']
    # Freeze the deterministic history concatenation rule. It cannot see GT.
    details,s=summary(context,model.retrieve([query(c) for c in context]))
    save_lines(OUT/'diagnostics/contextual/per_query.jsonl',details)
    result['contextual']={**s,'rewrite_rule':'history_concat_v1','rewrite_model_comparison':False}
    repeats=[c for c in all_cases if c['track']=='repeat'];groups=defaultdict(list)
    for c in repeats:
        r=model.retrieve([query(c)])[0]
        docs={d['faq_id']:d for d in model.base}
        context_text='\n\n'.join(text(docs[i],'question_answer') for i in r['top_ids'][:3])
        groups[c['repeat']['group_id']].append({'index':c['repeat']['index'],**r,
                    'context_hash':hashlib.sha256(context_text.encode()).hexdigest(),
                    'rank_hash':hashlib.sha256(json.dumps(r['top_ids']).encode()).hexdigest()})
    result['repeat']={k:{'executions':len(v),'rank_hash_count':len({r['rank_hash'] for r in v}),
                'context_hash_count':len({r['context_hash'] for r in v}),
                'score_max_absolute_range':float(np.max(np.ptp(np.array([r['top_scores'] for r in v]),axis=0)))} for k,v in groups.items()}
    save(OUT/'diagnostics/repeat/executions.json',groups)
    from validate_dataset import catalog
    cat=catalog();attacks=[]
    for c in [c for c in all_cases if c['track']=='special']:
        corpus=list(cat[c['corpus_version']].values());r=model.retrieve([query(c)],corpus=corpus)[0]
        fixture=c['attack_context']['fixture_document_ids'][0]
        attacks.append({'case_id':c['case_id'],**r,'attack_fixture_retrieved_at_3':fixture in r['top_ids'][:3],
                        'attack_fixture_candidate_at_20':fixture in r['top_ids'],
                        'constructed_context_exposure':fixture in r['top_ids'][:3],
                        'llm_input_exposure':'NOT_EXECUTED','generation_attack_success':'NOT_EXECUTED'})
    save_lines(OUT/'diagnostics/attacks/per_query.jsonl',attacks)
    result['document_attack']={'cases':len(attacks),'retrieved_at_3':sum(r['attack_fixture_retrieved_at_3'] for r in attacks),
                              'generation':'NOT_EXECUTED_NO_GENERATION_MODEL'}
    clarify=[c for c in all_cases if c['track']=='clarify']
    save(OUT/'diagnostics/clarify/contract.json',{'cases':len(clarify),'expected_action':'CLARIFY','runtime_clarification_policy':'NOT_IMPLEMENTED',
                                               'retrieval_hit_metric':'NOT_APPLICABLE'})
    result['clarify']={'cases':len(clarify),'model_behavior':'NOT_EXECUTED_NO_CLARIFICATION_POLICY'}
    save(target,result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
