"""BGE native sparse and token late interaction, measured after A-D."""
from __future__ import annotations
import argparse,pickle,time
import numpy as np
import torch
from scipy.sparse import csr_matrix
from FlagEmbedding import BGEM3FlagModel
from run_retrieval_benchmark import (OUT,DATA,dataset,query,text,model_path,report,rank,save,jsread,setup,sync,cleanup)

def sparse_matrix(weights):
    rows=[];cols=[];values=[]
    for i,w in enumerate(weights):
        for token,score in w.items():rows.append(i);cols.append(int(token));values.append(float(score))
    return csr_matrix((values,(rows,cols)),shape=(len(weights),250002),dtype=np.float32)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['E','F']);p.add_argument('--split',default='calibration',choices=['calibration','holdout']);p.add_argument('--view',default='question_answer',choices=['question','question_answer']);a=p.parse_args()
    setup()
    if a.split=='holdout' and not(OUT/'selection.json').exists():raise ValueError('freeze selection first')
    cases,corpus=dataset(a.split);ids=[d['faq_id'] for d in corpus]
    model=BGEM3FlagModel(model_path('BAAI/bge-m3'),use_fp16=False,devices='cuda:0',batch_size=16,query_max_length=512,passage_max_length=512)
    corpus_cache=OUT/f'cache/native-{a.view}-corpus.pkl';query_cache=OUT/f'cache/native-{a.split}-query.pkl'
    times={}
    sync();start=time.perf_counter()
    if corpus_cache.exists():doc=pickle.loads(corpus_cache.read_bytes())
    else:
        doc=model.encode([text(d,a.view) for d in corpus],batch_size=16,max_length=512,return_dense=True,return_sparse=True,return_colbert_vecs=True)
        corpus_cache.write_bytes(pickle.dumps(doc))
    sync();times['corpus_encode_or_cache_seconds']=time.perf_counter()-start
    sync();start=time.perf_counter()
    if query_cache.exists():q=pickle.loads(query_cache.read_bytes())
    else:
        q=model.encode([query(c) for c in cases],batch_size=16,max_length=512,return_dense=True,return_sparse=True,return_colbert_vecs=True)
        query_cache.write_bytes(pickle.dumps(q))
    sync();times['query_encode_or_cache_seconds']=time.perf_counter()-start
    dense=q['dense_vecs']@doc['dense_vecs'].T
    st=np.load(OUT/f'cache/dense-{a.view}-corpus.npy')
    cosine=np.sum(st*doc['dense_vecs'],axis=1)/(np.linalg.norm(st,axis=1)*np.linalg.norm(doc['dense_vecs'],axis=1))
    max_abs=float(np.max(np.abs(st-doc['dense_vecs'])))
    times.update(dense_bridge_cosine_min=float(cosine.min()),dense_bridge_max_abs=max_abs,corpus_cache_bytes=corpus_cache.stat().st_size,multi_vector_bytes=sum(v.nbytes for v in doc['colbert_vecs']),multi_vector_token_count=sum(len(v) for v in doc['colbert_vecs']))
    if cosine.min()<0.9999:raise ValueError('native dense bridge parity failed')
    if a.stage=='E':
        sparse=(sparse_matrix(q['lexical_weights'])@sparse_matrix(doc['lexical_weights']).T).toarray()
        for qi in range(min(3,len(cases))):
            for di in range(3):
                reference=float(model.compute_lexical_matching_score(q['lexical_weights'][qi],doc['lexical_weights'][di]))
                if not np.isclose(reference,sparse[qi,di],atol=1e-5):raise ValueError('sparse implementation mismatch')
        report('E-dense-control',cases,[rank(s,ids) for s in dense],dense,ids,a.split,times)
        report('E-sparse',cases,[rank(s,ids) for s in sparse],sparse,ids,a.split,times)
        np.save(OUT/f'cache/{a.split}-native-dense.npy',dense);np.save(OUT/f'cache/{a.split}-native-sparse.npy',sparse)
        dr=[rank(s,ids) for s in dense];sr=[rank(s,ids) for s in sparse]
        rrf=np.empty_like(dense)
        for i,(rd,rs) in enumerate(zip(dr,sr)):
            rrf[i]=0
            for ranking in [rd,rs]:
                for pos,index in enumerate(ranking,1):rrf[i,index]+=1/(60+pos)
        for m in [3,5,10,20]:
            pool=[set(d[:m])|set(s[:m]) for d,s in zip(dr,sr)]
            ranked=[sorted(indices,key=lambda j:(-rrf[i,j],ids[j]))[:m] for i,indices in enumerate(pool)]
            report(f'E-dense-sparse-RRF-M{m}',cases,ranked,rrf,ids,a.split,{**times,'raw_union_mean':float(np.mean([len(p) for p in pool])),'rrf_rank_constant':60,'unique_budget':m})
        np.save(OUT/f'cache/{a.split}-native-rrf.npy',rrf)
    else:
        # Same official MaxSim formula, vectorized over padded corpus buckets.
        buckets=[]
        order=sorted(range(len(ids)),key=lambda i:len(doc['colbert_vecs'][i]))
        for begin in range(0,len(ids),32):
            ix=order[begin:begin+32];length=max(len(doc['colbert_vecs'][i]) for i in ix);dim=doc['colbert_vecs'][ix[0]].shape[1]
            values=torch.zeros((len(ix),length,dim),device='cuda',dtype=torch.float32)
            mask=torch.zeros((len(ix),length),device='cuda',dtype=torch.bool)
            for j,i in enumerate(ix):v=doc['colbert_vecs'][i];values[j,:len(v)]=torch.as_tensor(v,device='cuda');mask[j,:len(v)]=True
            buckets.append((ix,values,mask))
        scores=np.empty((len(cases),len(ids)),dtype=np.float32)
        sync();start=time.perf_counter()
        with torch.inference_mode():
            for i,v in enumerate(q['colbert_vecs']):
                qq=torch.as_tensor(v,device='cuda')
                for ix,dd,mask in buckets:
                    token_scores=torch.einsum('qd,btd->bqt',qq,dd).masked_fill(~mask[:,None,:],-torch.inf)
                    ss=token_scores.max(dim=-1).values.mean(dim=-1).cpu().numpy();scores[i,ix]=ss
                if i%100==0:print(f'F full corpus {i}/{len(cases)} queries',flush=True)
        sync();times['late_interaction_seconds']=time.perf_counter()-start
        for qi in range(min(3,len(cases))):
            for di in range(3):
                reference=float(model.colbert_score(q['colbert_vecs'][qi],doc['colbert_vecs'][di]))
                if not np.isclose(reference,scores[qi,di],atol=2e-5):raise ValueError('colbert implementation mismatch')
        report('F-full',cases,[rank(s,ids) for s in scores],scores,ids,a.split,times)
        np.save(OUT/f'cache/{a.split}-colbert-scores.npy',scores)
        for m in [10,20]:
            pools=[rank(s,ids)[:m] for s in dense]
            rankings=[sorted(pool,key=lambda j:(-scores[i,j],ids[j])) for i,pool in enumerate(pools)]
            report(f'F-dense-pool-M{m}',cases,rankings,scores,ids,a.split,{**times,'candidate_source':'same native dense fixed pool','candidate_budget':m})
    del model;cleanup()

if __name__=='__main__':main()
