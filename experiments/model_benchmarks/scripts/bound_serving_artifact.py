"""Match the frozen quality length without changing any weights or tokenizer."""
import json,hashlib,os,shutil
from pathlib import Path
from run_retrieval_benchmark import OUT,save

def main():
    original=Path('/models/hf/serving-bge-m3');bounded=Path('/models/hf/serving-bge-m3-512')
    bounded.mkdir(exist_ok=True)
    for path in original.iterdir():
        target=bounded/path.name
        if path.name=='sentence_bert_config.json':
            config=json.loads(path.read_text());config['max_seq_length']=512
            target.write_text(json.dumps(config,indent=2)+'\n')
        elif not target.exists():target.symlink_to(path,target_is_directory=path.is_dir())
    save(OUT/'serving/bounded-artifact.json',{'source':str(original),'path':str(bounded),
         'only_changed_metadata':{'sentence_bert_config.json.max_seq_length':512},
         'model_weights_same_file':os.path.samefile(original/'model.safetensors',bounded/'model.safetensors'),
         'tokenizer_same_file':os.path.samefile(original/'tokenizer.json',bounded/'tokenizer.json'),
         'quality_max_tokens':512,'auto_truncate':False})
    print('Bounded metadata prepared; weights/tokenizer unchanged',flush=True)

if __name__=='__main__':main()
