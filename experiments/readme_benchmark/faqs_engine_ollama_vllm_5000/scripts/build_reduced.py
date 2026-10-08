"""Freeze an input-only, all-strata subset; never inspect retrieval outputs."""
import hashlib
import json
import re
import shutil
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'faqs_engine_ollama_vllm_30000'
DATA=ROOT/'data'
PARENTS=500

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def save_lines(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def allocate(counts,total,minimum):
    """Minimize squared count error under capacity and mandatory coverage."""
    keys=sorted(counts)
    target={k:total*counts[k]/sum(counts.values()) for k in keys}
    result={k:minimum[k] for k in keys}
    assert sum(result.values())<=total and all(0<=result[k]<=counts[k] for k in keys)
    while sum(result.values())<total:
        key=min((k for k in keys if result[k]<counts[k]),key=lambda k:(2*(result[k]-target[k])+1,k))
        result[key]+=1
    return result,target

assert not (ROOT/'outputs/run-v1/engines').exists(),'Do not replace already executed frozen inputs'
original_manifest=read(SOURCE/'data/manifest.json')
for name,value in original_manifest['hashes'].items():assert sha(SOURCE/'data'/name)==value
pairs=lines(SOURCE/'data/source_pairs.jsonl')
original_cases=lines(SOURCE/'data/cases.jsonl')
strata=defaultdict(list)
for p in pairs:strata[(p['brand'],p['language'],p['category'])].append(p)
groups=Counter((p['brand'],p['language']) for p in pairs)
minimum=Counter((k[0],k[1]) for k in strata)
quota,ideal=allocate(groups,PARENTS,minimum)
selected=[];stratum_report=[]
for group in sorted(groups):
    counts={key:len(value) for key,value in strata.items() if key[:2]==group}
    sq,st=allocate(counts,quota[group],{key:1 for key in counts})
    for key in sorted(counts):
        ordered=sorted(strata[key],key=lambda p:hashlib.sha256(('UBOT-500-20261007|'+p['faq_id']).encode()).hexdigest())
        selected.extend(ordered[:sq[key]])
        stratum_report.append({'brand':key[0],'language':key[1],'category':key[2],
            'original_pairs':counts[key],'selected_pairs':sq[key],
            'original_fraction':counts[key]/len(pairs),'selected_fraction':sq[key]/PARENTS,
            'fraction_delta':sq[key]/PARENTS-counts[key]/len(pairs)})
parent_ids={p['parent_case_id'] for p in selected}
assert len(parent_ids)==len(selected)==PARENTS
main=[c for c in original_cases if c['evaluation_role']=='main' and c['parent_case_id'] in parent_ids]
refs=[c for c in original_cases if c['evaluation_role']=='reference' and c['parent_case_id'] in parent_ids]
assert len(main)==PARENTS*10 and len(refs)==PARENTS
assert set(Counter(c['parent_case_id'] for c in main).values())=={10}
assert len({c['query'] for c in main})==len(main)
selected.sort(key=lambda p:p['parent_case_id'])
for name in ['corpus.jsonl','corpus_views.jsonl','reranker_chunks.jsonl','purpose_catalog.json']:
    DATA.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/'data'/name,DATA/name)
save_lines(DATA/'cases.jsonl',main+refs)
save_lines(DATA/'main_cases.jsonl',main)
save_lines(DATA/'source_pairs.jsonl',selected)
save_lines(DATA/'model_inputs.jsonl',[{'case_id':c['case_id'],'query':c['model_query']} for c in main])
save_lines(DATA/'paired_baselines.jsonl',[r for r in lines(SOURCE/'data/paired_baselines.jsonl') if r['parent_case_id'] in parent_ids])
save(DATA/'selected_parent_ids.json',sorted(parent_ids))
audit={'selected_pairs':PARENTS,'main_questions':len(main),'reference_questions':len(refs),
    'original_selected_pairs':len(pairs),'original_strata':len(strata),'selected_strata':len(stratum_report),
    'all_244_strata_preserved':True,'all_10_variants_per_parent_preserved':True,
    'selection_depends_on_retrieval_results':False,'selection_seed':'UBOT-500-20261007',
    'selection_rule':'minimum 1 per brand/language/category; closest squared count allocation at brand/language then category; deterministic FAQ hash within each stratum',
    'brand_language':[{'brand':k[0],'language':k[1],'original_pairs':groups[k],'selected_pairs':quota[k],
        'unconstrained_ideal_pairs':ideal[k],'count_error':quota[k]-ideal[k]} for k in sorted(groups)],
    'strata':stratum_report,'category_distribution_total_variation_distance':sum(abs(r['fraction_delta']) for r in stratum_report)/2,
    'cohort_counts':dict(Counter(c['cohort'] for c in main)),
    'type_counts':dict(Counter(c['type'] for c in main)),
    'validation_tag_counts':dict(Counter(t for c in main for t in c['validation_tags'])),
    'bilingual_variant_rows':sum(bool(c['language_pair_reference']) for c in main),
    'full_corpus_preserved':3246,'old_excel_data_used':False,'model_vectors_or_scores_reused':False}
assert set(audit['type_counts'])==set(original_manifest['type_counts'])
assert set(audit['validation_tag_counts'])=={t for c in original_cases[:30000] for t in c['validation_tags']}
save(DATA/'selection_audit.json',audit)
filenames=['cases.jsonl','main_cases.jsonl','corpus.jsonl','corpus_views.jsonl','reranker_chunks.jsonl',
           'paired_baselines.jsonl','source_pairs.jsonl','model_inputs.jsonl','purpose_catalog.json',
           'selected_parent_ids.json','selection_audit.json']
manifest={**original_manifest,'source_dataset':'faqs_based_30000 input-only subset, 500 complete parents',
    'source_full_variants_sha256':original_manifest['source_variants_sha256'],
    'source_full_cases_sha256':sha(SOURCE/'data/cases.jsonl'),
    'source_variants_sha256':sha(DATA/'main_cases.jsonl'),'main_rows':len(main),
    'paired_reference_rows':len(refs),'inference_rows':len(main)+len(refs),
    'selected_parent_count':PARENTS,'type_counts':audit['type_counts'],'cohort_counts':audit['cohort_counts'],
    'hashes':{name:sha(DATA/name) for name in filenames},
    'subset_selection':'all 244 strata mandatory; result-blind deterministic proportional allocation',
    'model_vectors_or_scores_reused':False}
manifest['protocol']['ollama_physical_batch']='2048 for <=2048 tokens, 4096 for longer full documents; context 8192; truncate false'
save(DATA/'manifest.json',manifest)

# Reuse benchmark implementation and model identities, never prior inputs/results.
replacements={'33000':'5500','30000':'5000','3000':'500'}
for name in ['common.py','run_experiment.py','test_contracts.py','compare_engines.py','candidate_logs.py']:
    text=(SOURCE/'scripts'/name).read_text(encoding='utf-8')
    text=re.sub(r'(?<!\d)(33000|30000|3000)(?!\d)',lambda m:replacements[m[0]],text)
    text=text.replace('33,000','5,500').replace('30,000','5,000').replace('3,000','500')
    if name=='compare_engines.py':
        text=text.replace("일반 19,358행, 조건 7,642행, 복합 500행을 별도로 집계했다.",
            '일반 '+str(audit['cohort_counts']['general'])+'행, 조건 '+str(audit['cohort_counts']['condition'])+'행, 복합 500행을 별도로 집계했다.')
        text=text.replace("'Ollama의 3,570토큰 답변 문서는 최초 physical batch 2,048 설정에서 HTTP 400을 반환했다. 진단 재현에서 num_batch=4,096, num_ctx=8,192, truncate=false로 전체 3,570토큰 처리와 유한한 1,024차원 벡터를 확인했다. 이후 2,048토큰 이하 입력은 기존 배치 설정을 유지하고 긴 문서에만 4,096을 적용했다. 완료된 질의 5,000행은 재사용했으며 이 실행 안의 체크포인트만 이어갔다. 실패 요청과 진단은 embedding_requests.jsonl 및 physical_batch_diagnostic.json에 보존했다. 이 physical batch 제한과 truncate=false 동작은 [Ollama v0.35.0 원본 코드](https://github.com/ollama/ollama/blob/v0.35.0/server/routes.go)에도 명시되어 있다.'",
            "'앞선 30,000행 실행에서 긴 문서 physical batch 한도를 진단했다. 이번 축소 실행은 입력 길이에 따라 num_batch 2,048/4,096, num_ctx=8,192, truncate=false를 적용한다. 이번 엔진별 질문과 전체 문서 벡터·reranker 점수·native 점수는 새로 계산하며 이전 실행 값을 재사용하지 않는다. [Ollama v0.35.0 원본 코드](https://github.com/ollama/ollama/blob/v0.35.0/server/routes.go)의 관련 동작과 앞선 실측을 기준으로 설정했다.'")
        text=text.replace("'main_candidate_rows':", "'main_candidate_rows':")
        text=text.replace('600,000행과 원문 기준 60,000행','100,000행과 원문 기준 10,000행')
        text=text.replace("report=['# 새 FAQ", "report=['# 새 FAQ")
    (ROOT/'scripts'/name).write_text(text,encoding='utf-8')
(ROOT/'configs').mkdir(parents=True,exist_ok=True)
shutil.copyfile(SOURCE/'configs/resolved_model_revisions.json',ROOT/'configs/resolved_model_revisions.json')
compose=(SOURCE/'compose.yaml').read_text(encoding='utf-8').replace('name: ubot-faqs-engine-30000','name: ubot-faqs-engine-5000')
(ROOT/'compose.yaml').write_text(compose,encoding='utf-8')
ps=(SOURCE/'run_all.ps1').read_text(encoding='utf-8')
ps=re.sub(r'(?<!\d)(33000|30000|3000)(?!\d)',lambda m:replacements[m[0]],ps)
ps=re.sub(r"foreach \(\$taskProject in @\([^\n]+\)\) \{",
    "foreach ($taskProject in @('ubot-readme-benchmark','ubot-engine-calibration-3000','ubot-engine-calibration-30000','ubot-faqs-engine-30000','ubot-faqs-engine-5000')) {",ps)
(ROOT/'run_all.ps1').write_text(ps,encoding='utf-8')

paused={'status':'paused','reason':'User reduced evaluation to 500 FAQ parents x 10 variants; this 30000-row run remains preserved.',
    'timestamp':datetime.now(timezone.utc).isoformat(),'active_containers':0,
    'ollama_completed':['A','B','C-M20','D-bge-question-M20'],
    'ollama_question_answer_checkpoint':read(SOURCE/'outputs/run-v1/engines/ollama/cache/D-bge-question_answer-M20-progress.json'),
    'vllm_started':False,'new_execution_folder':ROOT.name,
    'partial_window_winners_not_finalized':True}
save(SOURCE/'outputs/run-v1/paused_status.json',paused)
save(SOURCE/'outputs/run-v1/run_status.json',paused)

intro=f'''# 새 FAQ 500쌍 × 10개 · Ollama/vLLM

사용자 요청에 따라 새 faqs.jsonl 기반 3,000쌍에서 **500쌍 전체 변형 10개, 총 5,000질문**을 선정한다. 원문 비교 질문 500개는 별도로 실행하며 본 평가 평균에 합산하지 않는다. 검색 코퍼스는 원래 3,246 FAQ를 그대로 유지한다.

500쌍 선정은 검색 점수나 정답률을 보지 않고 수행했다. 244개 브랜드·언어·카테고리 구간마다 최소 한 쌍을 포함한 뒤, 브랜드/언어 및 카테고리의 원래 개수 비율에 가장 가까운 정수 배분을 사용했다. 구간 내 선택은 고정 FAQ 해시 순서다. 구간 보존 때문에 비율이 정확히 같지는 않다. [선정 감사](data/selection_audit.json)에 오차를 기록한다. 모든 primary variation 유형과 기존 validation tag를 보존했다. 일반 {audit['cohort_counts']['general']:,}행, 조건 {audit['cohort_counts']['condition']:,}행, 복합 500행이며 한영 대응어 변형은 {audit['bilingual_variant_rows']:,}행이다.

축소 전 30,000행 실행은 ../faqs_engine_ollama_vllm_30000 에 중지 상태로 보존한다. 코드·고정 모델 가중치만 재사용하며 질문·문서 벡터, reranker logits, native 점수, 임계값은 새로 계산한다. 이전 Excel 테스트는 사용하지 않는다.

Ollama → vLLM 순서로 엔진당 A/B/C, BGE/Jina×질문/질문+답변 네 조합, sparse/hybrid와 full/pool MaxSim 대조군까지 11개 구조를 실행한다. 각 dense 엔진은 기동→readiness→실제 HTTP 질의→종료→HTTP 폐쇄/컨테이너 0 확인 후 보조 모델을 실행한다. TEI는 제외한다.

Dense는 8,192토큰, reranker는 1,024토큰 한도이며 truncation을 사용하지 않는다. Ollama는 physical batch를 입력 길이에 따라 2,048/4,096으로 설정한다. Reranker 질문+답변은 후보 FAQ의 모든 원문 보존 창을 처리하고 최고 raw logit으로 합친다. 동점 창은 먼저 나온 창을 선택한다. 후보 budget은 20개다.

Ollama GGUF F16과 vLLM HF FP32의 차이도 결과에 포함한다. D는 공통 HF/CUDA FP16, E/F는 공통 FlagEmbedding/CUDA FP32다. 엔진 자체의 reranker/sparse/multi-vector API 측정으로 해석하지 않는다.

후보 로그 candidate_log.jsonl은 query, expected_faq_id(필수 canonical FAQ 배열), faq_id, retrieval_rank/cosine, retrieval_a_rank/cosine, retrieval_b_rank/cosine, rerank_rank/score, rank_delta, window_count, winning_window_index/text/score를 기록한다. A/B 순위는 전체 3,246 FAQ의 1-based 순위, retrieval_rank는 C 후보 내 1-based 순위, winning_window_index는 해당 FAQ 내 0-based 순서다. rank_delta = retrieval_rank - rerank_rank이고 양수 개선, 음수 하락, 0 유지다. 조합별 본 평가 100,000 후보행과 원문 비교 10,000 후보행이다.

GT는 독립 검수된 원자 사실이 아닌 FAQ 지원 묶음이다. NO_FAQ가 없어 거절 threshold/margin은 null이며 F1을 적합하지 않는다. 구조 선정은 같은 긍정 데이터의 순위 진단으로, Holdout이나 운영 적용 확정 결과가 아니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\\run_all.ps1
```

진행은 outputs/run-v1/run_status.json, 최종 비교는 outputs/run-v1/새FAQ_Ollama_vLLM_비교.md, 완료 증빙은 completion_audit.json이다. 원천 문서 내용은 데이터로만 처리한다.
'''
(ROOT/'README.md').write_text(intro,encoding='utf-8')
print(json.dumps({k:audit[k] for k in ['selected_pairs','main_questions','reference_questions','selected_strata','cohort_counts','type_counts','bilingual_variant_rows','category_distribution_total_variation_distance']},ensure_ascii=False),flush=True)
