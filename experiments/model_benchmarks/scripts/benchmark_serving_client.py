"""Dense common-lane parity and every concurrency/batch combination.

The first measured sweep is exploratory: 10s/100 attempts/one repetition per
cell. It does not claim the precision of the design's 60s/1000/three-run profile.
"""
from __future__ import annotations
import argparse,json,time,threading,hashlib,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import requests
from transformers import AutoTokenizer
from run_retrieval_benchmark import OUT,DATA,lines,save,jsread,model_path,rank
from retrieval_common import fact_metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--engine',choices=['ollama','tei','vllm'],required=True);p.add_argument('--url',required=True);p.add_argument('--seconds',type=float,default=10);p.add_argument('--minimum-attempts',type=int,default=100);a=p.parse_args()
    result_dir=OUT/'serving'/a.engine;result_dir.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(model_path('BAAI/bge-m3'))
    corpus=lines(DATA/'corpus.jsonl');docs=[d['question'] for d in corpus]
    queries=jsread(OUT/'cache/dense-calibration-query_texts.json');qref=np.load(OUT/'cache/dense-calibration-queries.npy')
    keep=[i for i,q in enumerate(queries) if len(tokenizer(q,truncation=False)['input_ids'])<=300]
    selected=keep[:50];pool=[queries[i] for i in keep]
    local=threading.local()
    def call(texts):
        if not hasattr(local,'session'):local.session=requests.Session()
        if a.engine=='ollama':endpoint='/api/embed';payload={'model':'bge-m3','input':texts,'truncate':False,'keep_alive':'30m'}
        elif a.engine=='tei':endpoint='/embed';payload={'inputs':texts,'truncate':False}
        else:endpoint='/v1/embeddings';payload={'model':'BAAI/bge-m3','input':texts,'encoding_format':'float'}
        start=time.perf_counter();response=local.session.post(a.url.rstrip('/')+endpoint,json=payload,timeout=30)
        response.raise_for_status();data=response.json()
        if a.engine=='ollama':v=data['embeddings']
        elif a.engine=='tei':v=data
        else:
            items=sorted(data['data'],key=lambda r:r['index']);v=[r['embedding'] for r in items]
        arr=np.asarray(v,dtype=np.float32)
        if arr.shape!=(len(texts),1024) or not np.isfinite(arr).all():raise ValueError('Invalid embedding response shape or values')
        return arr,time.perf_counter()-start
    try:
        # Batch against individually requested vectors to verify output ordering.
        smoke=pool[:3];batch,_=call(smoke);individual=np.stack([call([q])[0][0] for q in smoke])
        if not np.allclose(batch,individual,atol=0.002):raise ValueError('Batch output order/parity failed')
        dv=np.concatenate([call(docs[i:i+32])[0] for i in range(0,len(docs),32)])
        qv=np.concatenate([call([queries[i] for i in selected[j:j+16]])[0] for j in range(0,len(selected),16)])
        norms=np.linalg.norm(dv,axis=1);dn=dv/norms[:,None];qn=qv/np.linalg.norm(qv,axis=1)[:,None]
        reference=np.load(OUT/'cache/dense-question-corpus.npy');qreference=qref[selected]
        cosine=np.sum(dn*reference,axis=1);qcos=np.sum(qn*qreference,axis=1)
        es=qn@dn.T;rs=qreference@reference.T;ids=[d['faq_id'] for d in corpus]
        overlaps=[];top1=[];fact_deltas=[];p0_drift=[]
        cal_cases=lines(DATA/'cases.jsonl')
        from run_retrieval_benchmark import query
        by_query={query(c):c for c in cal_cases if c['split']=='calibration'}
        for j,(e,r) in enumerate(zip(es,rs)):
            er=rank(e,ids);rr=rank(r,ids);overlaps.append(len(set(er[:10])&set(rr[:10]))/10);top1.append(er[0]==rr[0])
            case=by_query[queries[selected[j]]]
            em=fact_metrics([ids[i] for i in er],case['fact_groups'],3)
            rm=fact_metrics([ids[i] for i in rr],case['fact_groups'],3)
            if em['fact_recall'] is not None:fact_deltas.append(em['fact_recall']-rm['fact_recall'])
            p0_drift.append(bool(e[er[0]]>=.75)!=bool(r[rr[0]]>=.75))
        parity={'status':'passed' if cosine.min()>=0.9999 and qcos.min()>=0.9999 else 'drift_detected','vector_dimension':1024,'corpus_count':len(docs),'query_count':len(selected),'corpus_cosine_min':float(cosine.min()),'query_cosine_min':float(qcos.min()),'normalized_max_abs_error':float(np.max(np.abs(dn-reference))),'raw_norm_min':float(norms.min()),'raw_norm_max':float(norms.max()),'top1_agreement':float(np.mean(top1)),'top10_overlap':float(np.mean(overlaps)),'batch_order_probe':'passed','lane':'dense_question_common_component','long_query_workload':'excluded_over_300_tokens_for_common_lane'}
        parity.update(fact_recall_at_3_delta=float(np.mean(fact_deltas)) if fact_deltas else None,
                      baseline_cosine_P0_decision_drift=float(np.mean(p0_drift)),
                      calibrated_reranker_policy_drift='NOT_EVALUATED_IN_COMPONENT_LANE')
        save(result_dir/'parity.json',parity);np.save(result_dir/'corpus_vectors.npy',dv);np.save(result_dir/'query_vectors.npy',qv)
    except Exception as e:
        save(result_dir/'capability.json',{'status':'FAILED_PROBE','error':str(e),'load_executed':False});raise
    save(result_dir/'capability.json',{'dense':'AVAILABLE','native_sparse':'NOT_PROBED_IN_DENSE_LANE','native_colbert':'NOT_PROBED_IN_DENSE_LANE','parity':parity['status']})
    output=[];gpu_samples=[];stop_monitor=threading.Event()
    def monitor_gpu():
        while not stop_monitor.is_set():
            try:
                value=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5).stdout.strip()
                gpu_samples.append({'timestamp':time.time(),'gpu_memory_and_utilization':value})
            except Exception:pass
            stop_monitor.wait(1)
    monitor=threading.Thread(target=monitor_gpu,daemon=True);monitor.start()
    for concurrency in [1,4,8,16,32]:
        for batch_size in [1,8,32]:
            for warm in range(5):call([pool[(warm+j)%len(pool)] for j in range(batch_size)])
            lock=threading.Lock();latencies=[];errors=[];attempts=0;success=0;start=time.perf_counter()
            def worker(worker_id):
                nonlocal attempts,success
                index=worker_id*37
                while True:
                    with lock:
                        if time.perf_counter()-start>=a.seconds and attempts>=a.minimum_attempts:return
                        attempts+=1
                    texts=[pool[(index+j)%len(pool)] for j in range(batch_size)];index+=batch_size
                    try:
                        _,latency=call(texts)
                        with lock:latencies.append(latency*1000);success+=1
                    except Exception as error:
                        with lock:
                            if len(errors)<3:errors.append(str(error))
            with ThreadPoolExecutor(max_workers=concurrency) as ex:list(ex.map(worker,range(concurrency)))
            elapsed=time.perf_counter()-start
            row={'engine':a.engine,'concurrency':concurrency,'texts_per_request':batch_size,'wall_seconds':elapsed,'attempts':attempts,'successes':success,'failures':attempts-success,'failure_rate':(attempts-success)/attempts,'p50_ms':float(np.percentile(latencies,50)) if latencies else None,'p95_ms':float(np.percentile(latencies,95)) if latencies else None,'p99_ms':float(np.percentile(latencies,99)) if latencies else None,'requests_per_sec':success/elapsed,'texts_per_sec':success*batch_size/elapsed,'error_samples':errors,'retries':0,'profile':'exploratory_10s_100_attempts_one_repetition'}
            output.append(row);save(result_dir/'load_results.json',output);print(json.dumps(row),flush=True)
    stop_monitor.set();monitor.join(timeout=6);save(result_dir/'gpu_samples.json',gpu_samples)
    save(result_dir/'manifest.json',{'lane':'dense_common_component','profile':'exploratory_first_sweep','configured_seconds':a.seconds,'minimum_attempts':a.minimum_attempts,'repetitions':1,'cells':len(output),'workload_text_count':len(pool),'workload_hash':hashlib.sha256(json.dumps(pool,ensure_ascii=False).encode()).hexdigest(),'latency_scope':'client_end_to_end_successes','failure_denominator':'all_attempts','quality_holdout_precedes_serving':(OUT/'holdout/policy_results.json').exists(),'full_60s_1000_three_run_profile_executed':False})

if __name__=='__main__':main()
