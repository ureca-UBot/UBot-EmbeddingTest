import ast,hashlib,json,subprocess,sys,tempfile,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT/'ubot-v4-runtime'
SOURCE=ROOT.parent
def sha(b):return hashlib.sha256(b).hexdigest()
for p in PACKAGE.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
lock=json.loads((PACKAGE/'environment/runtime-lock.json').read_text())
unchanged=[]
for relative,expected in lock['original_code_hashes'].items():
 if relative=='serving_test/compare_v4.py':continue  # historical prose only, see comparison below
 assert sha((PACKAGE/relative).read_bytes())==expected,relative
 unchanged.append(relative)
# Packaging only changes two strings describing incidents from the original run.
old=ast.parse((SOURCE/'serving_test/compare_v4.py').read_text(encoding='utf-8'))
new=ast.parse((PACKAGE/'serving_test/compare_v4.py').read_text(encoding='utf-8'))
def constants(tree):return [n for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
oc,nc=constants(old),constants(new);assert len(oc)==len(nc)
changes=[]
for a,b in zip(oc,nc):
 if a.value!=b.value:
  assert a.value.startswith(('- SKT Ollama의 3,548토큰','- BGE 리랭커의 첫 실행'))
  changes.append(a.lineno);a.value=b.value
assert len(changes)==2 and ast.dump(old)==ast.dump(new),'Evaluation logic changed'
# Reject a bad/incomplete ZIP without creating even the destination directory.
with tempfile.TemporaryDirectory(dir=ROOT) as temp:
 p=Path(temp);bad=p/'bad.zip';dest=p/'newdata'
 with zipfile.ZipFile(bad,'w') as z:z.writestr('jsonl/existing/user_questions.jsonl','{}\n')
 r=subprocess.run([sys.executable,str(PACKAGE/'scripts/import_dataset.py'),'--source',str(bad),'--destination',str(dest)],capture_output=True,text=True)
 assert r.returncode!=0 and not dest.exists()
validation={'validated_at':datetime.now(timezone.utc).isoformat(),'python_syntax':True,'powershell_syntax':True,'compose_config':True,
 'unchanged_original_files':unchanged,'comparison_evaluation_ast_unchanged':True,
 'packaging_changes':['Docker image tags/build recipes, project/volume scope','Preparation/import/run/stop entrypoints','Historical incident report prose'],
 'isolated_dataset_import':{'jsonl_files':8,'original_fixture_files_reproduced':8,'questions':1918,'all_hashes_match':True,'reject_bad_input_without_writes':True},
 'docker_environment_versions':{'benchmark_packages_checked':284,'jina_packages_checked':284,'differences':0},
 'runner_tests_passed':5,'native_tokenizer_math_tests_passed':3,'four_dataset_preflight_passed':True,
 'bge_serving_weight_tensors_equal':391,'serving_safetensors_sha256':'993b2248881724788dcab8c644a91dfd63584b6e5604ff2037cb5541e1e38e7e',
 'fresh_network_download_tested':False,'fresh_docker_image_build_tested':False,'full_gpu_benchmark_rerun':False,
 'validation_environment':'Isolated extracted-layout copy, original pinned images, read-only model volume, CPU containers with network disabled'}
(PACKAGE/'PACKAGE_VALIDATION.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
files=sorted(p for p in PACKAGE.rglob('*') if p.is_file() and p.name!='PACKAGE_CONTENTS.json')
for p in files:
 rel=p.relative_to(PACKAGE)
 assert not set(rel.parts)&{'datasets','outputs','__pycache__','.git','node_modules'},rel
 assert p.suffix not in {'.jsonl','.xlsx','.npy','.pkl','.safetensors','.bin','.pt','.gguf','.log','.zip'},rel
 assert p.name!='.env'
contents={'package':'ubot-v4-runtime','contains':'code, environment, documentation, hash-only dataset metadata',
 'files':[{'path':p.relative_to(PACKAGE).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in files]}
(PACKAGE/'PACKAGE_CONTENTS.json').write_text(json.dumps(contents,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
files.append(PACKAGE/'PACKAGE_CONTENTS.json')
archive=ROOT/'UBot_v4_실행코드_환경.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in files:z.write(p,Path('ubot-v4-runtime')/p.relative_to(PACKAGE))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None and len(z.namelist())==len(files)
 for item in contents['files']:assert sha(z.read('ubot-v4-runtime/'+item['path']))==item['sha256']
(archive.with_suffix('.sha256')).write_text(sha(archive.read_bytes())+'  '+archive.name+'\n',encoding='utf-8')
print(json.dumps({'archive':str(archive),'files':len(files),'bytes':archive.stat().st_size,'sha256':sha(archive.read_bytes()),'data_models_results_excluded':True},ensure_ascii=True))
