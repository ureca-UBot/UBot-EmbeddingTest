"""Trace every reranked FAQ to first-stage rank and actual winning document window."""
import argparse
import json
import os
from collections import defaultdict
import numpy as np
from common import DATA,OUT,N,MAIN_N,inputs,read,lines,save,doc_text,sha

FIELDS=['query','expected_faq_id','faq_id','retrieval_rank','retrieval_cosine',
        'retrieval_a_rank','retrieval_a_cosine','retrieval_b_rank','retrieval_b_cosine',
        'rerank_rank','rerank_score','rank_delta','window_count','winning_window_index',
        'winning_window_text','winning_window_score']

def export_candidate_log(engine,name,view,cases=None,docs=None):
    if cases is None:cases,docs=inputs()
    base=OUT/'engines'/engine
    target=base/name/'candidate_log.jsonl'
    metapath=base/name/'candidate_log_schema.json'
    if target.exists() and metapath.exists():
        metadata=read(metapath)
        assert metadata['rows']==N*20 and metadata['fields']==FIELDS
        return metadata
    summary=read(base/name/'summary.json')
    assert summary['main_case_count']==MAIN_N and summary['case_count']==N
    candidates=read(base/'cache/C-pools.json')
    qa=np.load(base/'cache/B-scores.npy',mmap_mode='r')
    qq=np.load(base/'cache/A-scores.npy',mmap_mode='r')
    scores=np.load(base/'cache'/f'{name}-scores.npy',mmap_mode='r')
    mapping=defaultdict(list)
    if view=='question':
        windows=[{'faq_id':d['faq_id'],'text':doc_text(d,'question')} for d in docs]
        for j in range(len(docs)):mapping[j]=[j]
        winners=None
    else:
        windows=lines(DATA/'reranker_chunks.jsonl')
        positions={d['faq_id']:j for j,d in enumerate(docs)}
        for w,window in enumerate(windows):mapping[positions[window['faq_id']]].append(w)
        winners=np.load(base/'cache'/f'{name}-winning-windows.npy',mmap_mode='r')
        assert winners.shape==scores.shape and winners.dtype==np.int16
    local_index={w:k for values in mapping.values() for k,w in enumerate(values)}
    tmp=target.with_suffix('.jsonl.tmp');rows=main=references=0
    with tmp.open('w',encoding='utf-8') as stream:
        for i,(case,ids) in enumerate(zip(cases,candidates)):
            assert len(ids)==20 and len(set(ids))==20
            reranked=sorted(ids,key=lambda j:(-float(scores[i,j]),j))
            ranks={j:k+1 for k,j in enumerate(reranked)}
            aranks=np.empty(len(docs),dtype=np.int32);branks=np.empty(len(docs),dtype=np.int32)
            aranks[np.argsort(-qq[i],kind='stable')]=np.arange(1,len(docs)+1)
            branks[np.argsort(-qa[i],kind='stable')]=np.arange(1,len(docs)+1)
            for k,j in enumerate(ids):
                w=j if winners is None else int(winners[i,j])
                assert w in mapping[j] and windows[w]['faq_id']==docs[j]['faq_id']
                score=float(scores[i,j]);assert np.isfinite(score) and score>-1e19
                record={'engine':engine,'structure':name,'case_id':case['case_id'],
                    'parent_case_id':case['parent_case_id'],'evaluation_role':case['evaluation_role'],
                    'brand':case['brand'],'cohort':case['cohort'],'query':case['model_query'],
                    'expected_faq_id':case['source_ids'],'faq_id':docs[j]['faq_id'],
                    'retrieval_rank':k+1,'retrieval_cosine':float(max(qq[i,j],qa[i,j])),
                    'retrieval_a_rank':int(aranks[j]),'retrieval_a_cosine':float(qq[i,j]),
                    'retrieval_b_rank':int(branks[j]),'retrieval_b_cosine':float(qa[i,j]),
                    'rerank_rank':ranks[j],'rerank_score':score,'rank_delta':k+1-ranks[j],'window_count':len(mapping[j]),
                    'winning_window_index':local_index[w],'winning_window_text':windows[w]['text'],
                    'winning_window_score':score}
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                rows+=1
                if case['evaluation_role']=='main':main+=1
                else:references+=1
    assert rows==N*20 and main==MAIN_N*20 and references==(N-MAIN_N)*20
    os.replace(tmp,target)
    metadata={'rows':rows,'main_candidate_rows':main,'reference_candidate_rows':references,'fields':FIELDS,
        'faq_candidates_per_query':20,'expected_faq_id_type':'array of all required canonical FAQ IDs; never sent to model',
        'retrieval_rank':'1-based rank within C-M20 candidate pool',
        'retrieval_cosine':'max(question-view cosine, question-answer-view cosine)',
        'retrieval_a_rank':'1-based question-view rank across all 3246 corpus FAQs; ties use corpus order',
        'retrieval_a_cosine':'question-view dense cosine',
        'retrieval_b_rank':'1-based question-answer-view rank across all 3246 corpus FAQs; ties use corpus order',
        'retrieval_b_cosine':'question-answer-view dense cosine',
        'rerank_rank':'1-based descending raw logit; ties use corpus FAQ order',
        'rank_delta':'retrieval_rank - rerank_rank; positive=improved, negative=worsened, zero=unchanged',
        'window_count':'number of scored reranker input windows for this FAQ; question view has 1',
        'winning_window_index':'0-based index within this FAQ; ties use earliest source window',
        'winning_window_text':'exact document text sent to reranker; query is stored separately',
        'winning_window_score':'FAQ maximum raw logit, equal to rerank_score',
        'view':view,'ranking_recomputed_without_inference':True,'sha256':sha(target)}
    save(metapath,metadata)
    print(json.dumps({'candidate_log':engine+'/'+name,'rows':rows,'main_rows':main,'references':references}),flush=True)
    return metadata

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--engine',required=True,choices=['ollama','vllm'])
    p.add_argument('--name',required=True);p.add_argument('--view',required=True,choices=['question','question_answer'])
    args=p.parse_args();export_candidate_log(args.engine,args.name,args.view)
