"""Apply the frozen architecture to original regressions and fresh repetitions."""
import json, re
from collections import defaultdict
import numpy as np
import torch
from run_retrieval_benchmark import OUT,DATA,lines,jsread,save,save_lines,model_path,text,rank,report,query,re_split,setup
from retrieval_common import stable_hash

class FrozenRetriever:
    def __init__(self,name):
        self.name=name
        self.native=name.startswith(('E-','F-'))
        self.view='question_answer' if 'question_answer' in name or name=='B' else 'question'
        if self.native:
            from FlagEmbedding import BGEM3FlagModel
            self.model=BGEM3FlagModel(model_path('BAAI/bge-m3'),use_fp16=False,devices='cuda:0')
        else:
            from sentence_transformers import SentenceTransformer
            self.model=SentenceTransformer(model_path('BAAI/bge-m3'),device='cuda',model_kwargs={'dtype':torch.float32})
            self.model.max_seq_length=512
        self.reranker=None
        if name.startswith('D-'):
            from transformers import AutoModelForSequenceClassification,AutoTokenizer
            jina='jina' in name
            path=model_path('jinaai/jina-reranker-v2-base-multilingual' if jina else 'BAAI/bge-reranker-v2-m3')
            self.reranker=AutoModelForSequenceClassification.from_pretrained(path,trust_remote_code=jina,torch_dtype=torch.float32,
                                **({'use_flash_attn':False} if jina else {})).to('cuda').eval()
            self.tokenizer=AutoTokenizer.from_pretrained(path,trust_remote_code=jina)

    def set_corpus(self,corpus):
        self.corpus=corpus;self.ids=[d['faq_id'] for d in corpus]
        if self.native:
            self.doc=self.model.encode([text(d,'question') for d in corpus],batch_size=16,max_length=512,
                                      return_dense=True,return_sparse=True,return_colbert_vecs=True)
            self.buckets=[]
            if self.name.startswith('F-'):
                order=sorted(range(len(corpus)),key=lambda i:len(self.doc['colbert_vecs'][i]))
                for begin in range(0,len(order),32):
                    ix=order[begin:begin+32];length=max(len(self.doc['colbert_vecs'][i]) for i in ix)
                    values=torch.zeros((len(ix),length,1024),device='cuda');mask=torch.zeros((len(ix),length),device='cuda',dtype=torch.bool)
                    for j,i in enumerate(ix):
                        v=self.doc['colbert_vecs'][i];values[j,:len(v)]=torch.as_tensor(v,device='cuda');mask[j,:len(v)]=True
                    self.buckets.append((ix,values,mask))
        else:
            self.dv=self.model.encode([text(d,'question') for d in corpus],batch_size=16,normalize_embeddings=True)
            self.cv=None
            if self.name=='B' or self.name.startswith('C-') or (self.name.startswith('D-') and 'Acontrol' not in self.name):
                self.cv=self.model.encode([text(d,'question_answer') for d in corpus],batch_size=16,normalize_embeddings=True)

    def score(self,queries):
        if self.native:
            q=self.model.encode(queries,batch_size=16,max_length=512,return_dense=True,return_sparse=True,return_colbert_vecs=True)
            scores=q['dense_vecs']@self.doc['dense_vecs'].T
            if self.name=='E-sparse' or 'RRF' in self.name:
                from run_native_retrieval import sparse_matrix
                sparse=(sparse_matrix(q['lexical_weights'])@sparse_matrix(self.doc['lexical_weights']).T).toarray()
                if self.name=='E-sparse':scores=sparse
                else:
                    rrf=np.zeros_like(scores)
                    for i,(d,s) in enumerate(zip(scores,sparse)):
                        for ranking in [rank(d,self.ids),rank(s,self.ids)]:
                            for position,index in enumerate(ranking,1):rrf[i,index]+=1/(60+position)
                    scores=rrf
            if self.name.startswith('F-'):
                dense=scores.copy();scores=np.empty_like(scores)
                with torch.inference_mode():
                    for i,v in enumerate(q['colbert_vecs']):
                        qq=torch.as_tensor(v,device='cuda')
                        for ix,dd,mask in self.buckets:
                            ss=torch.einsum('qd,btd->bqt',qq,dd).masked_fill(~mask[:,None,:],-torch.inf)
                            scores[i,ix]=ss.max(dim=-1).values.mean(dim=-1).cpu().numpy()
                if self.name.startswith('F-dense-pool'):
                    m=int(self.name.rsplit('M',1)[1])
                    for i,d in enumerate(dense):
                        pool=set(rank(d,self.ids)[:m]);scores[i,[j for j in range(len(self.ids)) if j not in pool]]=-1e20
        else:
            qv=self.model.encode(queries,batch_size=16,normalize_embeddings=True);a=qv@self.dv.T
            scores=a if self.cv is None else np.maximum(a,qv@self.cv.T)
            if self.name=='B':scores=qv@self.cv.T
        if self.reranker is not None:
            m=int(self.name.rsplit('M',1)[1]);pools=[rank(s,self.ids)[:m] for s in scores]
            pairs=[[q,text(self.corpus[j],self.view)] for q,pool in zip(queries,pools) for j in pool];values=[]
            with torch.inference_mode():
                for begin in range(0,len(pairs),24):
                    inputs=self.tokenizer(pairs[begin:begin+24],padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
                    values.extend(self.reranker(**inputs).logits.reshape(-1).float().cpu().numpy().tolist())
            scores=np.full_like(scores,-1e20)
            for i,pool in enumerate(pools):
                for j,v in zip(pool,values[i*m:(i+1)*m]):scores[i,j]=v
        return scores

def main():
    setup();name=jsread(OUT/'selection.json')['selected_with_top1_gate'];retriever=FrozenRetriever(name)
    for key in ['legacy_60','legacy_235']:
        data=jsread(DATA/(key+'.json'));cases=[]
        for r in data['queries']:
            v=r['values']
            if v.get('처리 의도')!='FAQ_RAG':continue
            cases.append({'case_id':key+'-'+str(v['ID']),'family_id':key+'-'+str(v['ID']),
                          'reporting_cohort':'legacy_regression','source':{'category':r['sheet']},'answerability':'full',
                          'query':v['사용자 질문'],'query_mode':'standalone',
                          'fact_groups':[{'fact_id':'f1','primary_ids':re_split(v.get('Primary GT')),
                                          'acceptable_ids':re_split(v.get('Acceptable GT') or v.get('허용 GT'))}]})
        retriever.set_corpus(data['corpus']);scores=retriever.score([query(c) for c in cases])
        rankings=[[i for i in rank(s,retriever.ids) if s[i]>-1e19] for s in scores]
        rows=report('selected-'+key,cases,rankings,scores,retriever.ids,'regression',{'architecture':name,'original_corpus_preserved':True})
        base={r['case_id']:r for r in lines(OUT/f'regression/{key}/per_query.jsonl')}
        save_lines(OUT/f'regression/selected-{key}/changes.jsonl',
                   [{'case_id':r['case_id'],'baseline_hit1':base[r['case_id']]['metrics']['1']['semantic_hit'],
                     'selected_hit1':r['metrics']['1']['semantic_hit'],'baseline_top3':base[r['case_id']]['top_ids'][:3],
                     'selected_top3':r['top_ids'][:3]} for r in rows])
    corpus=lines(DATA/'corpus.jsonl');retriever.set_corpus(corpus)
    examples=[c for c in lines(DATA/'diagnostic_examples.jsonl') if c['corpus_version']=='current-faq-1024-v1']
    scores=retriever.score([query(c) for c in examples])
    rows=report('selected-design-examples',examples,
                [[i for i in rank(s,retriever.ids) if s[i]>-1e19] for s in scores],scores,retriever.ids,
                'diagnostics',{'architecture':name,'status':'design_examples_not_approved_holdout'})
    save_lines(OUT/'diagnostics/selected-design-examples/types.jsonl',
               [{'case_id':c['case_id'],'primary_type':c['primary_type'],'tags':c['tags'],'metrics':r['metrics']}
                for c,r in zip(examples,rows)])
    source=[r for r in lines(DATA/'source_rows.jsonl') if r['항목 코드']=='RT' and r['실행 모드']=='실제 검색' and 'FAQ_RAG' in r['기대 처리 경로']]
    rows=[]
    for r in source:
        q=r['사용자 질문'];history=json.loads(r['대화 이력 (JSON)'] or '[]')
        if history:q='대화 이력:\n'+'\n'.join(f"{h['role']}: {h['content']}" for h in history)+'\n현재 질문: '+q
        scores=retriever.score([q])[0];ordered=[retriever.ids[i] for i in rank(scores,retriever.ids) if scores[i]>-1e19][:20]
        context='\n\n'.join(text(corpus[retriever.ids.index(fid)],'question_answer') for fid in ordered[:3])
        rows.append({'execution_id':r['실행 ID'],'repeat_group':r['원본 질문 ID'],'query_hash':stable_hash(q),
                     'rank_hash':stable_hash(ordered),'score_hash':stable_hash([float(scores[retriever.ids.index(fid)]) for fid in ordered]),
                     'context_hash':stable_hash(context),'generation_output_hash':None})
    save_lines(OUT/'diagnostics/selected-repeats.jsonl',rows);groups=defaultdict(list)
    for r in rows:groups[r['repeat_group']].append(r)
    save(OUT/'diagnostics/selected-repeat-summary.json',{'architecture':name,'groups':len(groups),'executions':len(rows),
         **{key+'_unstable_groups':sum(len(set(r[key+'_hash'] for r in g))>1 for g in groups.values()) for key in ['rank','score','context']},
         'fresh_query_encoding_and_scoring_each_execution':True,'generation_evaluated':False})
    print('Frozen architecture regressions and fresh repeated scoring completed',flush=True)

if __name__=='__main__':main()
