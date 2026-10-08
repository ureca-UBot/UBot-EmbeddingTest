"""Final separate hardware probe of the frozen dense retrieval component."""
import time,gc,os,resource,argparse,hashlib,json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from run_retrieval_benchmark import OUT,model_path,jsread,save,sync

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--requests',type=int,default=30);args=p.parse_args()
    assert (OUT/'serving/shortlist.json').exists(),'Engine sweep must precede hardware probe'
    queries=jsread(OUT/'cache/dense-calibration-query_texts.json')
    queries=[q for q in queries if len(q)<250][:60]
    available=torch.cuda.is_available()
    assert available if args.device=='cuda' else not available, 'Device isolation does not match requested probe'
    device_probe={'device':args.device,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'cuda_available':available,'gpu_name':torch.cuda.get_device_name(0) if available else None}
    save(OUT/f'hardware/{args.device}-device-probe.json',device_probe)
    print(json.dumps({'event':'device_verified',**device_probe}),flush=True)
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    results=[];vectors={}
    for device in [args.device]:
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();model=SentenceTransformer(model_path('BAAI/bge-m3'),device=device,model_kwargs={'dtype':torch.float32});model.max_seq_length=512
        if device=='cuda':sync()
        cold=time.perf_counter()-start
        print(json.dumps({'event':'model_ready','device':device,'model_load_seconds':cold}),flush=True)
        vectors[device]=model.encode(queries,batch_size=16,normalize_embeddings=True)
        for batch in [1,8,32]:
            model.encode(queries[:batch],normalize_embeddings=True)
            values=[]
            for i in range(args.requests):
                texts=[queries[(i*batch+j)%len(queries)] for j in range(batch)]
                if device=='cuda':sync()
                start=time.perf_counter();model.encode(texts,batch_size=batch,normalize_embeddings=True)
                if device=='cuda':sync()
                values.append(time.perf_counter()-start)
            row={'device':device,'dtype':'float32','batch':batch,'requests':args.requests,'texts':batch*args.requests,'p50_ms':float(np.percentile(values,50)*1000),'p95_ms':float(np.percentile(values,95)*1000),'p99_ms':float(np.percentile(values,99)*1000),'texts_per_second':args.requests*batch/sum(values),'model_load_seconds':cold,'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'peak_torch_vram_bytes':int(torch.cuda.max_memory_allocated()) if device=='cuda' else 0,'profile':'30_request_exploratory_component_probe_not_engine_load_matrix'}
            results.append(row);print(json.dumps(row),flush=True)
        del model;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    path=OUT/f'hardware/{args.device}.json'
    save(path,{'scope':'same frozen SentenceTransformer dense component; engines shortlisted first; one container per device','engine_shortlist':jsread(OUT/'serving/shortlist.json'),'cpu_threads':torch.get_num_threads(),'query_count':len(queries),'query_hash':hashlib.sha256(json.dumps(queries,ensure_ascii=False).encode()).hexdigest(),'results':results,'cpu_gpu_engine_end_to_end_comparison':False})
    np.save(OUT/f'hardware/{args.device}-vectors.npy',vectors[args.device])

if __name__=='__main__':main()
