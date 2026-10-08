"""Create the code/environment-only v4 release from an explicit allowlist."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent
DEST=ROOT/'ubot-v4-runtime'
def write(p,s):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8',newline='\n')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
copied=[]
for name in ['dataset.py','manifest.json','evaluation_contract.json']:
 shutil.copy2(SOURCE/name,DEST/name);copied.append(name)
for name in ['common_v4.py','compare_v4.py','run_native.py','run_rerank.py','run_v4.py','inspect_log.py','test_runner.py','test_native_math.py']:
 target=DEST/'serving_test'/name;target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(SOURCE/'serving_test'/name,target);copied.append('serving_test/'+name)
for name in ['run_dense.ps1','run_remaining.ps1']:
 text=(SOURCE/'serving_test'/name).read_text(encoding='utf-8').replace('ubot-fair-v4','ubot-v4-runtime')
 write(DEST/'serving_test'/name,text)
versions={n:read(DEST/'environment'/f'{n}.json')['packages'] for n in ['base','benchmark','jina']}
for name,base in [('benchmark','base'),('jina','benchmark')]:
 diff=[f'{n}=={v}' for n,v in sorted(versions[name].items()) if versions[base].get(n)!=v]
 write(DEST/'docker'/f'requirements-{name}.lock','\n'.join(diff)+'\n')
write(DEST/'docker/Dockerfile.benchmark','''FROM vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
USER root
COPY docker/requirements-benchmark.lock /tmp/requirements.lock
# All changed/new distributions versus the immutable base image are pinned.
RUN /usr/bin/python3 -m pip install --no-cache-dir --no-deps -r /tmp/requirements.lock
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models/hf TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /workspace/serving_test
ENTRYPOINT ["/usr/bin/python3"]
''')
write(DEST/'docker/Dockerfile.jina','''FROM ubot-v4-benchmark:runtime
# Separate compatibility image for the pinned Jina remote implementation.
COPY docker/requirements-jina.lock /tmp/requirements-jina.lock
RUN /usr/bin/python3 -m pip install --no-cache-dir --no-deps -r /tmp/requirements-jina.lock
''')
compose=(SOURCE/'serving_test/compose.yaml').read_text(encoding='utf-8')
compose=compose.replace('name: ubot-fair-v4','name: ubot-v4-runtime').replace('ubot-retrieval-benchmark:run-v1','ubot-v4-benchmark:runtime').replace('ubot-retrieval-jina:run-v1','ubot-v4-jina:runtime')
compose=compose.replace('  worker:\n    <<: [*runtime, *gpu]','  worker:\n    <<: [*runtime, *gpu]\n    build:\n      context: ..\n      dockerfile: docker/Dockerfile.benchmark')
compose=compose.replace('    image: ubot-v4-jina:runtime','    image: ubot-v4-jina:runtime\n    build:\n      context: ..\n      dockerfile: docker/Dockerfile.jina')
compose=compose.replace('  ollama:\n','''  model-setup:
    <<: *runtime
    profiles: [setup]
    environment:
      HF_HOME: /models/hf
      HF_MODULES_CACHE: /tmp/hf_modules
      HF_HUB_OFFLINE: '0'
      TRANSFORMERS_OFFLINE: '0'
      HF_HUB_DISABLE_TELEMETRY: '1'
      PYTHONDONTWRITEBYTECODE: '1'
      PYTHONUNBUFFERED: '1'
    volumes:
      - ..:/workspace:ro
      - hf-models:/models/hf:rw
    command: [/workspace/scripts/prepare_models.py]
  ollama:
''')
compose=compose.replace("    external: true\n    name: ubot-retrieval-benchmark_hf-benchmark-cache","    name: ${HF_MODELS_VOLUME:-ubot-v4-runtime-hf-models}")
compose=compose.replace("    external: true\n    name: ubot-retrieval-benchmark_ollama-cache","    name: ${OLLAMA_MODELS_VOLUME:-ubot-v4-runtime-ollama-models}")
compose=compose.replace("    external: true\n    name: ubot-readme-benchmark_hf-runtime","    name: ${HF_RUNTIME_VOLUME:-ubot-v4-runtime-hf-runtime}")
write(DEST/'serving_test/compose.yaml',compose)
write(DEST/'serving_test/.env.example','''# Default: independent v4 volumes.
HF_MODELS_VOLUME=ubot-v4-runtime-hf-models
OLLAMA_MODELS_VOLUME=ubot-v4-runtime-ollama-models
HF_RUNTIME_VOLUME=ubot-v4-runtime-hf-runtime

# To reuse the original measured cache on the original machine instead:
# HF_MODELS_VOLUME=ubot-retrieval-benchmark_hf-benchmark-cache
# OLLAMA_MODELS_VOLUME=ubot-retrieval-benchmark_ollama-cache
# HF_RUNTIME_VOLUME=ubot-readme-benchmark_hf-runtime
''')
write(DEST/'.dockerignore','datasets/\nserving_test/outputs/\n.git/\n**/__pycache__/\n*.zip\n')
write(DEST/'.gitignore','datasets/\nserving_test/outputs/\nserving_test/.env\n**/__pycache__/\n')
lock={'python':'3.12.3','base_image':'vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90',
 'original_images':{'ubot-retrieval-benchmark:run-v1':'sha256:7f65c030c30efa601e87a82762833a8a90bddd0145d3c1e606c2ff71efa7cea0','ubot-retrieval-jina:run-v1':'sha256:30a9ff3f129558caa28210ce67aca22ca5000a2f8a4562770226a1761e60c081'},
 'models':{'BAAI/bge-m3':'5617a9f61b028005a4858fdac845db406aefb181','BAAI/bge-reranker-v2-m3':'953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e','jinaai/jina-reranker-v2-base-multilingual':'9cfeff2df7d40d1b78e75e5e9cebec92a99813c9'},
 'ollama_model':{'name':'bge-m3:latest','manifest_digest':'7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab','gguf_sha256':'daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c','quantization':'F16'},
 'original_code_hashes':{name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in copied},
 'fixtures':{key:{name:hashlib.sha256((SOURCE/'datasets'/key/'fixtures'/name).read_bytes()).hexdigest() for name in ['repeat.jsonl','document_attack.jsonl']} for key in ['existing','kt','skt','lgu']}}
write(DEST/'environment/runtime-lock.json',json.dumps(lock,indent=2)+'\n')
print(json.dumps({'exact_source_files':len(copied),'runtime':str(DEST)}))
