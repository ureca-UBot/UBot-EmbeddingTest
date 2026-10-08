from pathlib import Path
import json
from huggingface_hub import HfApi, snapshot_download

root=Path('/workspace/model_benchmarks/configs')
result={}
api=HfApi()
for name in ['BAAI/bge-m3','BAAI/bge-reranker-v2-m3','jinaai/jina-reranker-v2-base-multilingual']:
    info=api.model_info(name)
    files=[s.rfilename for s in info.siblings]
    safe=any(f.endswith('.safetensors') for f in files)
    patterns=['*.json','*.py','*.model','*.txt','*.pt','*.safetensors']
    if not safe:patterns.append('pytorch_model*.bin')
    print('Downloading',name,'revision',info.sha,flush=True)
    path=snapshot_download(name,revision=info.sha,allow_patterns=patterns,max_workers=4)
    result[name]={'revision':info.sha,'snapshot_path':path,'files':[str(p.relative_to(Path(path))) for p in Path(path).rglob('*') if p.is_file()]}
    (root/'resolved_model_revisions.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Ready',name,flush=True)
