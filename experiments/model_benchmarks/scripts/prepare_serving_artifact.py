"""A lossless safetensors representation of the frozen HF weights for TEI."""
from pathlib import Path
import shutil,json,hashlib
import torch
from safetensors.torch import save_file,load_file
from run_retrieval_benchmark import model_path,OUT,save

source=Path(model_path('BAAI/bge-m3'));destination=Path('/models/hf/serving-bge-m3')
destination.mkdir(exist_ok=True)
for path in source.rglob('*'):
    if path.is_file() and path.suffix not in ['.bin','.safetensors']:
        target=destination/path.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
state=torch.load(source/'pytorch_model.bin',map_location='cpu',weights_only=True)
state={k:v.contiguous().clone() for k,v in state.items()}
save_file(state,str(destination/'model.safetensors'),metadata={'format':'pt'})
check=load_file(str(destination/'model.safetensors'))
assert state.keys()==check.keys() and all(torch.equal(state[k],check[k]) for k in state)
save(OUT/'serving/artifact.json',{'source_snapshot':str(source),'artifact_directory':str(destination),'conversion':'lossless_pytorch_to_safetensors','tensor_equality_verified':True,'tensor_count':len(state),'artifact_sha256':hashlib.sha256((destination/'model.safetensors').read_bytes()).hexdigest()})
print('Serving artifact ready:',destination,flush=True)
