"""Shortlist only engines that complete the common-lane sweep and pass parity."""
from run_retrieval_benchmark import OUT,save,jsread

def main():
    rows=[]
    for engine in ['ollama','tei','vllm']:
        root=OUT/'serving'/engine
        parity=jsread(root/'parity.json') if (root/'parity.json').exists() else {}
        load=jsread(root/'load_results.json') if (root/'load_results.json').exists() else []
        good=(parity.get('status')=='passed' and len(load)==15 and all(r['failures']==0 for r in load))
        batch1=[r for r in load if r['texts_per_request']==1]
        peak=max(load,key=lambda r:r['texts_per_sec']) if load else None
        rows.append({'engine':engine,'qualified_dense_component':good,'parity_status':parity.get('status','probe_failed'),
                     'completed_cells':len(load),'total_failures':sum(r['failures'] for r in load),
                     'best_batch1_texts_per_second':max((r['texts_per_sec'] for r in batch1),default=None),
                     'peak_measured_cell':peak,'production_adoption':False})
    good=[r for r in rows if r['qualified_dense_component']]
    selected=max(good,key=lambda r:r['best_batch1_texts_per_second'])['engine'] if good else None
    save(OUT/'serving/shortlist.json',{'scope':'BGE dense question embedding component only',
         'qualification':'vector cosine_min >= .9999, complete 15-cell sweep, zero errors',
         'selection_rule':'maximum measured batch=1 throughput among qualified engines',
         'profile':'exploratory_10s_100_attempts_one_repetition','engines':rows,'selected_engine':selected,
         'operational_SLO_not_supplied':True,'reranker_serving_not_measured':True,
         'production_adoption':False,'NO_FAQ_holdout_gate_failed':True})
    print((OUT/'serving/shortlist.json').read_text(encoding='utf-8'),flush=True)

if __name__=='__main__':main()
