"""Pinned HF snapshots and lossless BGE-M3 safetensors serving representation."""
import argparse,gc,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');a=p.parse_args()
 lock=json.loads((ROOT/'environment/runtime-lock.json').read_text());cache=Path('/models/hf/hub');paths={}
 for name,revision in lock['models'].items():
  if not a.verify_only:
   from huggingface_hub import snapshot_download
   snapshot_download(name,revision=revision,cache_dir=cache,max_workers=4,
    allow_patterns=['*.json','*.py','*.model','*.txt','*.pt','*.safetensors','pytorch_model*.bin'])
  folder=cache/('models--'+name.replace('/','--'))/'snapshots'/revision
  if not (folder/'config.json').is_file():raise FileNotFoundError(f'Missing pinned model: {folder}')
  if not any(folder.glob('*.safetensors')) and not any(folder.glob('pytorch_model*.bin')):raise FileNotFoundError(f'Missing weights: {folder}')
  paths[name]=folder
 source=paths['BAAI/bge-m3'];dest=Path('/models/hf/serving-bge-m3')
 for name in ['sparse_linear.pt','colbert_linear.pt']:
  if not (source/name).is_file():raise FileNotFoundError(name)
 if not a.verify_only:
  dest.mkdir(parents=True,exist_ok=True)
  for f in source.rglob('*'):
   if f.is_file() and f.suffix not in ['.bin','.safetensors']:
    target=dest/f.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
  import torch
  from safetensors.torch import save_file
  state=torch.load(source/'pytorch_model.bin',map_location='cpu',weights_only=True)
  state={k:v.contiguous().clone() for k,v in state.items()}
  save_file(state,str(dest/'model.safetensors'),metadata={'format':'pt'})
  del state;gc.collect()
 import torch
 from safetensors.torch import load_file
 expected=torch.load(source/'pytorch_model.bin',map_location='cpu',weights_only=True)
 actual=load_file(str(dest/'model.safetensors'))
 if expected.keys()!=actual.keys() or not all(torch.equal(expected[k],actual[k]) for k in expected):raise ValueError('Serving weights differ from pinned BGE-M3 snapshot')
 for f in source.rglob('*'):
  if f.is_file() and f.suffix in ['.json','.model','.txt'] and digest(f)!=digest(dest/f.relative_to(source)):
   raise ValueError(f'Serving tokenizer/config mismatch: {f.name}')
 print(json.dumps({'models':lock['models'],'tensor_equality_verified':True,'tensor_count':len(expected),'model_safetensors_sha256':digest(dest/'model.safetensors')}))
if __name__=='__main__':main()
