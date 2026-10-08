"""Render measured artifacts into a report and a complete source-row ledger."""
import json, hashlib, re
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/retrieval/run-20261006-v1';DATA=ROOT/'faq/retrieval_run_v2'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def pct(x):return 'N/A' if x is None else f'{x*100:.2f}%'
def number(x):return 'N/A' if x is None else f'{x:.2f}'
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mb(text):
    match=re.search(r'([\d.]+)\s*([KMGT]?i?B)',text)
    if not match:return None
    value,unit=match.groups();scale={'B':1,'KiB':1024,'MiB':1024**2,'GiB':1024**3,'TiB':1024**4,'kB':1000,'MB':10**6,'GB':10**9}
    return float(value)*scale.get(unit,1)

def main():
    manifest=read(DATA/'split_manifest.json');selection=read(OUT/'selection.json');name=selection['selected_with_top1_gate']
    policy=read(OUT/'holdout/policy_results.json');gates=read(OUT/'quality-gates.json');engines=read(OUT/'serving/shortlist.json')
    cases=lines(DATA/'cases.jsonl');corpus=lines(DATA/'corpus.jsonl');qset={d['question'] for d in corpus}
    exact=sum(c['query'] in qset for c in cases if c['query_mode']=='standalone')
    duplicates=Counter(d['body_hash'] for d in corpus)
    ledger=[];case_lookup={c['source']['execution_id']:c for c in cases}
    special={r['execution_id']:r for r in lines(OUT/'diagnostics/special-conditions.jsonl')}
    repeats={r['execution_id'] for r in lines(OUT/'diagnostics/selected-repeats.jsonl')}
    for r in lines(DATA/'source_rows.jsonl'):
        eid=r['실행 ID'];c=case_lookup.get(eid);s=special.get(eid)
        status='MEASURED_SOURCE_LABEL_RETRIEVAL' if c else 'MEASURED_SPECIAL_RETRIEVAL' if s else 'NOT_EVALUATED_NON_FAQ_ROUTE' if 'FAQ_RAG' not in str(r['기대 처리 경로']) else 'NOT_SCORED_PENDING_GT'
        ledger.append({'execution_id':eid,'source_category':r['항목 코드'],'status':status,'split':c['split'] if c else None,
                       'reporting_cohort':c['reporting_cohort'] if c else None,'repeat_stability_measured':eid in repeats,
                       'special_retrieval_condition_reproduced':s.get('retrieval_condition_reproduced') if s else None,
                       'generation_executed':False,'external_API_executed':False,'domain_review_complete':False})
    (OUT/'source-3000-ledger.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in ledger),encoding='utf-8')
    doc=['# UBot BGE-M3 실제 실행 결과 — 2026-10-06','',
         '**검색 순위 개선은 확인했으나 동결한 정책의 운영 적용은 보류한다.** 선택 구성은 질문 벡터 Top 20 → BGE reranker에 질문 입력 → 최종 Top 3다. Holdout 핵심 검색의 threshold 적용 전 AllFactsHit@3 family 평균은 90.81% → 94.49%였지만, 정답 없는 질문 40건 중 9건을 채택했다. 기존 235문항 회귀의 Top 1도 94.89% → 92.77%로 하락했다. `quality-gates.json`에 실패를 그대로 기록했다.','',
         '이 결과는 원천 라벨 기반의 **잠정 실험**이다. 독립 도메인 검수와 실제 서비스 실패 로그가 없어 운영 품질 인증으로 사용할 수 없다. Holdout 이후 구조·threshold를 변경하지 않았다.','',
         '## 실행 범위와 데이터','',
         '| 항목 | 실제 실행 또는 처리 |','|---|---|',
         '| 원천 | 동일 질문을 포함한 두 엑셀을 6,000건으로 합산하지 않음; 4개 원천 snapshot/SHA 보존 |',
         '| 정상 코퍼스 | 1,024건: FAQ_RAG 985 / MAP_API 15 / USER_INFO 24, intent 사전 필터 없음 |',
         '| 실제 FAQ 검색 | 1,997행: calibration 1,331 / holdout 666 |',
         '| 특수 조건 | 680행: HR 200 / EC 200 / CF 200 / AD 30 / RT 50 |',
         '| 비 FAQ 경로 | 290행: API·개인 조회·범위 밖 처리는 이 retrieval runner에서 미실행 |',
         '| GT 확인 필요 | 33행: 검색 정답 채점 보류; 이 중 일부 RT는 안정성만 측정 |',
         '| 회귀 | 기존 코퍼스를 각각 보존한 57·235 FAQ 문항. 57개는 235개와 겹치므로 독립 292건으로 합산하지 않음 |',
         '| 세부 설계 예시 | 20건 중 정상 코퍼스 17건 A/B/선택 구성 진단, CF/AD는 원천 fixture 트랙으로 실행 |',
         '| 한영 용어 동등성 | 원문 질문과 7개 용어 치환을 공유 GT로 묶은 116쌍. 문장 전체 번역 시험과 구분 |','',
         f"920 family, 590 연결 component로 split을 고정했다. 같은 GT signature·원본 ID·시나리오·동일 질문을 연결해 간접 중복을 막았다. 핵심 일반 검색은 calibration 792행/516 family, holdout 476행/278 family다. 원문 코퍼스 질문과 정확히 같은 standalone 입력은 전체 실제 FAQ 검색 중 {exact}행이다. 같은 본문 hash의 중복 그룹은 {sum(v>1 for v in duplicates.values())}개다. 정답 범위를 자동 확대하거나 실패 문항을 제거하지 않았다.",
         '', '대화 검색은 고정 이력 연결 규칙을 모든 방식에 동일하게 적용했다. 정답용 검색질문·기대 답변·GT ID를 query나 후보 강제 주입에 사용하지 않았다. 반복 행은 일반 핵심 평균에서 분리했다. 상세 3,000행 처리 상태는 `source-3000-ledger.jsonl`에서 확인할 수 있다.','',
         '## 고정 환경','',
         'Docker Desktop 4.90.0 / Linux Engine 29.7.2 / RTX 3060 12,288 MiB / NVIDIA driver 595.95. 품질 비교는 FP32, TF32 끄기, 정규화한 cosine, 최대 512 token, dense batch 16으로 실행했다. A/B 입력 길이 점검에서는 truncation이 없었다. 모델 버전은 `resolved_model_revisions.json`에 고정했다.','',
         '| 모델 | HF revision |','|---|---|']
    for model,info in read(ROOT/'configs/resolved_model_revisions.json').items():doc.append(f"| {model} | `{info['revision']}` |")
    doc += ['', 'BGE 품질 런타임은 Torch 2.13.0+cu130 / Transformers 5.17.0 / SentenceTransformers 6.1.0 / FlagEmbedding 1.4.2다. Jina는 외부 custom Python과 Transformers 5의 호환성 문제 때문에 별도 Transformers 4.51.3 환경에서 실행했다. 네트워크 없음·읽기 전용 root/model/scripts/data·결과 폴더만 쓰기로 제한했다. 이 차이는 특히 속도 비교에서 통제되지 않은 adapter 차이로 취급한다.','',
         '## Calibration의 구조 비교','',
         '아래 값은 threshold 적용 전, core_general의 **family 평균**이다. Hit@1은 사실 하나 이상, AllFactsHit@3는 필요한 모든 사실, FactRecall@10은 필요한 사실의 비율이다. 서로 다른 지표를 같은 Recall로 발표하지 않는다.','',
         '| 구성 | Hit@1 | Hit@3 | AllFactsHit@3 | FactRecall@10 |','|---|---:|---:|---:|---:|']
    for item in selection['candidate_results']:
        doc.append(f"| {item['name']} | {pct(item['semantic_hit@1'])} | {pct(item['semantic_hit@3'])} | {pct(item['all_facts_hit@3'])} | {pct(item['fact_recall@10'])} |")
    boot=selection['paired_bootstrap_all_facts_at_3']
    doc += ['',f"선택 구성과 A의 calibration AllFactsHit@3 차이는 {pct(boot['mean_delta'])}p다. 연결 component {boot['components']}개를 재표집한 paired bootstrap 95% 구간은 {pct(boot['ci95'][0])}p ~ {pct(boot['ci95'][1])}p다. 이 구간은 calibration 결과이며 holdout의 신뢰구간으로 옮겨 쓰지 않는다.",
            '', 'C는 view별 union 크기와 같은 후보 예산으로 줄인 결과를 구분했다. `C-M*/volume_controls.jsonl`에는 A/B의 Top 2K·실제 union 크기 대조군이 있다. Native sparse는 공식 lexical weight dot, F는 token MaxSim 후 query token 평균을 사용했고 공식 helper와 대조했다. Native dense와 SentenceTransformers dense의 최소 cosine은 0.99999988이었다.','',
            'F 질문 표현은 13,809 document token / 56,561,664 bytes의 token vector를 저장했다. C의 두 문서 벡터와 다른 표현이다. 캐시를 사용한 F 단계의 작은 Torch peak는 전체 모델을 GPU에 상주시킨 서빙 VRAM으로 해석하지 않는다.','',
            'Native E/F와 dense+sparse RRF는 검수 gate가 미완료인 상태에서 잠정 탐색 진단으로 실행했다. 검수된 실패 원인에 따른 정식 조건부 채택으로 발표하지 않는다. BM25 추가는 보류했다. 남은 lexical/entity 원인과 GT 모호성은 독립 검수가 필요한 상태이며 `diagnostics/calibration-failures.jsonl`에 query별 가설·후보·rank를 저장했다. 자동 가설을 검수된 LEXICAL_MISS로 바꾸지 않았다.','',
            '## 동결 정책과 Holdout','',
            'Threshold 전 검색 결과를 유형별로 분리했다. 다음 표도 family 평균이다. 문맥 검색은 고정 이력 연결의 retrieval 결과이며 rewrite 모델 품질을 측정한 값이 아니다. targeted_condition은 원천 조건 문항이며 독립 검수된 EXACT_ENTITY 성능으로 바꾸지 않는다.','',
            '| split / 유형 | 행 / family | A AllFactsHit@3 | 선택 구성 AllFactsHit@3 |','|---|---:|---:|---:|']
    for split in ['calibration','holdout']:
        a=read(OUT/f'{split}/A/summary.json')['cohorts'];b=read(OUT/f'{split}/{name}/summary.json')['cohorts']
        for cohort in ['core_general','targeted_condition','contextual']:
            ar=a[cohort];br=b[cohort]
            doc.append(f"| {split} / {cohort} | {ar['rows']} / {ar['families']} | {pct(ar['family_mean']['all_facts_hit@3'])} | {pct(br['family_mean']['all_facts_hit@3'])} |")
    doc += ['', '아래는 고정 threshold까지 적용한 최종 정책이다. NO_FAQ는 positive class이고, binary 혼동행렬은 full FAQ_EXISTS와 NO_FAQ 문항만 사용한다. Partial·문맥·반복을 무조건 binary 표본에 섞지 않았다.','',
            '| 구성·정책 | 고정 threshold | 핵심 AllFactsHit@3 (family 평균) | NO_FAQ P / R / F1 | 정답 없는 질문 오채택 | 정답 있는 질문 거절 |','|---|---:|---:|---|---:|---:|']
    for r in policy['results']:
        n=r['no_faq'];doc.append(f"| {r['experiment']} / {r['policy']} | {r['threshold']:.6f} | {pct(r['family_mean_core_AllFactsHit_at_3'])} | {pct(n['precision'])} / {pct(n['recall'])} / {pct(n['f1'])} | {n['fn']}/{n['tp']+n['fn']} ({pct(n['unsafe_accept_rate'])}) | {n['fp']}/{n['fp']+n['tn']} ({pct(n['detector_false_positive_rate'])}) |")
    doc += ['', 'NO_FAQ를 positive class로 정의했다. 오채택률은 NO_FAQ detector의 false-negative rate이고, 정답 있는 질문의 거절률이 detector false-positive rate다. Calibration의 오채택 ≤1% 제약을 holdout에서 만족하지 못했다. 원시 reranker 점수는 cosine이나 확률이 아니므로 기존 0.75를 재사용하지 않는다. Margin 분포는 calibration에서 진단만 했고 GT를 runtime 판정에 쓰지 않았다.','',
            'Holdout 구조 선택과 threshold는 결과 생성 전 `selection.json`으로 동결했고 hash를 결과에 기록했다. 현재 어댑터 의존성 때문에 B/C 후보 캐시와 동일 BGE 입력의 부가 후보 결과도 산출됐다. 최종 정책 평가는 동결 baseline/challenger만 사용했고 부가 결과로 재선정하지 않았다. 이 실행상 차이를 숨기지 않는다.','',
            '## 원래 코퍼스 회귀','', '| 세트 | A Top1 / Top3 | 선택 구성 Top1 / Top3 | Top1 좋아짐 / 나빠짐 |','|---|---|---|---|']
    for key in ['legacy_60','legacy_235']:
        a=read(OUT/f'regression/{key}/summary.json')['cohorts']['legacy_regression']['family_mean']
        b=read(OUT/f'regression/selected-{key}/summary.json')['cohorts']['legacy_regression']['family_mean']
        changes=lines(OUT/f'regression/selected-{key}/changes.jsonl')
        good=sum(r['selected_hit1']>r['baseline_hit1'] for r in changes);bad=sum(r['selected_hit1']<r['baseline_hit1'] for r in changes)
        doc.append(f"| {len(changes)}문항 | {pct(a['semantic_hit@1'])} / {pct(a['semantic_hit@3'])} | {pct(b['semantic_hit@1'])} / {pct(b['semantic_hit@3'])} | {good} / {bad} |")
    doc += ['', '235문항의 Top1 하락은 1%p 허용 기준을 넘었다. `regression/selected-legacy_235/changes.jsonl`과 per_query에 각 문항의 전후 후보를 남겼다. 새 source FAQ의 동일 ID를 원래 회귀 GT에 이식하지 않았다.','',
            '## 언어·특수 조건·반복','', '| 진단 | 결과 |','|---|---|']
    lang=read(OUT/'diagnostics/language-summary.json')['views']
    for view,r in lang.items():doc.append(f"| 한영 용어 116쌍 / {view} | 쌍 모두 Hit@3 {pct(r['pair_hit_at_3'])}; 영어 표현 Hit@3 {pct(r['en_term_hit_at_3'])}; Top10 overlap {pct(r['top10_overlap'])} |")
    if (OUT/'diagnostics/selected-language-summary.json').exists():
        r=read(OUT/'diagnostics/selected-language-summary.json');doc.append(f"| 한영 용어 116쌍 / 선택 구성 | 쌍 모두 Hit@3 {pct(r['pair_hit_at_3'])}; 영어 표현 Hit@3 {pct(r['en_term_hit_at_3'])}; Top10 overlap {pct(r['pair_top10_overlap'])} |")
    special_summary=read(OUT/'diagnostics/special-summary.json');repeat=read(OUT/'diagnostics/selected-repeat-summary.json')
    doc += [f"| HR/EC/CF/AD/특수 RT | {special_summary['retrieval_condition_reproduced']}/{special_summary['execution_rows']} retrieval 조건 재현 |",
            f"| 문서 공격 | 공격 문서 context 포함 {special_summary['attack_exposed_in_context']}회 (AD와 RT의 재노출 포함); 생성 방어 NOT_EVALUATED |",
            f"| 선택 구성 실제 반복 | {repeat['groups']}그룹 / {repeat['executions']} fresh encode+rerank; rank·score·context 불안정 그룹 모두 0 |",'',
            'HR은 FAQ807–827의 오답 후보 풀, EC는 빈 후보 결과를 강제하는 원천 계약으로 실행했다. CF는 두 버전 문서와 지정 distractor를 함께 검색했고 날짜 prefilter로 한 버전을 지우지 않았다. 정상 코퍼스에는 유효기간 메타데이터가 없어 CF를 실제 정책 변경 검증으로 발표하지 않는다. 공격 문서는 데이터로 취급했다. 생성 모델을 호출하지 않아 실제 LLM 입력 노출·LLM 행동·output hash는 측정하지 않았다. 위 재현율은 검색 단계만의 값이다.','',
            '독립 검수된 16유형 전면 커버리지는 아직 확보되지 않았다. 세부 설계 예시·용어 치환은 posthoc 진단이고 승인된 holdout이 아니다. 실제 서비스 REAL_FAILURE는 로그가 제공되지 않아 0건이다. API·생성 성공을 검색 성공으로 대체하지 않았다.','',
            '## 서빙 엔진 — 켜기·확인·측정·끄기·확인','',
            '엔진은 하나씩 실행했다. 컨테이너 정상 실행과 embedding probe 후 측정하고, stop 뒤 컨테이너 종료·HTTP 응답 종료·전체 GPU 메모리 snapshot을 확인한 다음 엔진을 시작했다. 종료 증거는 각 엔진의 `shutdown-verification.json`에 저장한다.','',
            '공통 측정 범위는 **질문 dense embedding 컴포넌트**다. 선택 reranker까지 포함한 end-to-end UBot 지연이 아니다. 엔진마다 코퍼스 벡터를 다시 만들었고 query/corpus를 서로 다른 엔진에서 혼용하지 않았다. 기준은 FP32 HF weights이며 Ollama GGUF F16은 별도 artifact 변형이다. TEI/vLLM용 safetensors는 원래 state dict의 모든 tensor와 torch.equal로 대조했다.','',
            '성능은 엔진과 현재 구성의 조합에 대한 값이다. TEI는 batch token 8,192·batch request 32·수용 한도 2,048, vLLM은 batch token 8,192·eager 실행·GPU 이용 예산 0.5를 사용했다. 엔진별 최적 튜닝이나 전체 batching 설정 일치는 보장하지 않는다. readiness 시간은 API 준비 확인 시간이며 Ollama 모델의 최초 GPU 적재 시간을 따로 측정한 값이 아니다. 모델 cold-load latency와 서버 내부 개별 batch trace는 이번 부하 client에서 분리 기록하지 않았다.','',
            '동시성 1/4/8/16/32 × batch 1/8/32, 각 최소 100 attempt·10초·1회다. 최소 요청 수 때문에 실제 실행 시간이 10초를 넘는 조건이 있다. 워밍업 5회는 제외하고 재시도 없이 성공 요청 latency와 전체 attempt 실패율을 분리했다. 60초·1,000요청·3회 설계 기본값은 실행하지 않았다. p99와 엔진 순위는 1차 측정의 제한을 갖는다.','',
            'TEI 첫 시작은 원본 SentenceTransformers의 최대 길이 8,192 설정으로 준비 확인까지 183.32초가 걸렸다. 품질 런과 동일한 512로 메타데이터를 맞추고 CUDA cache를 보존한 이후 시작은 약 7초였다. 길이 설정과 cache 상태가 함께 바뀌었으므로 차이를 단일 원인의 cold-start 개선으로 발표하지 않는다. 모델 tensor와 tokenizer는 변경하지 않았다.','',
            'TEI 수용 한도 512의 첫 15조건 결과도 `tei-default-capacity-512/`에 보존했다. 동시성 32 × batch 32에서 10,461회 중 10,397회가 HTTP 429였다. 엔진을 끄고 종료를 확인한 뒤 수용 한도 2,048의 구성으로 15조건 전체를 다시 실행했다. 아래 TEI 표는 최종 구성이며 실패한 첫 측정을 삭제하거나 합산하지 않았다. TEI의 GELU tanh 근사와 기준 Torch의 GELU 차이는 실제 parity 결과와 함께 취급한다.','',
            '| 엔진 | 벡터 최소 cosine (문서 / query) | Top1 일치 / Top10 overlap | 부하 조건 | 최고 texts/s | 자격 |','|---|---|---|---|---:|---|']
    for r in engines['engines']:
        engine=r['engine'];root=OUT/'serving'/engine;p=read(root/'parity.json') if (root/'parity.json').exists() else {}
        peak=r['peak_measured_cell'];c_cos=format(p['corpus_cosine_min'],'.8f') if p else 'N/A';q_cos=format(p['query_cosine_min'],'.8f') if p else 'N/A'
        qualification='통과' if r['qualified_dense_component'] else f"미통과: parity {r['parity_status']}, 실패 {r['total_failures']}건"
        doc.append(f"| {engine} | {c_cos} / {q_cos} | {pct(p.get('top1_agreement'))} / {pct(p.get('top10_overlap'))} | {r['completed_cells']}/15 | {number(peak['texts_per_sec']) if peak else 'N/A'} | {qualification} |")
    doc += ['',f"1차 dense 컴포넌트 후보: `{engines['selected_engine']}`. 동일 벡터 gate와 15조건 무오류를 통과한 엔진 중 batch=1 최고 throughput으로 좁혔다. 운영 SLO와 장시간 측정이 없어 제품 도입 결론으로 쓰지 않는다. Sparse/ColBERT 서빙 API는 선택 구조에 사용하지 않아 이번 dense lane에서 probe하지 않았다."]
    for engine in ['ollama','tei','vllm']:
        root=OUT/'serving'/engine;load=read(root/'load_results.json') if (root/'load_results.json').exists() else []
        doc += ['',f'### {engine} 실제 부하 결과','', '| 동시성 | batch | p50 ms | p95 ms | p99 ms | texts/s | 실패/시도 |','|---:|---:|---:|---:|---:|---:|---:|']
        for r in load:doc.append(f"| {r['concurrency']} | {r['texts_per_request']} | {number(r['p50_ms'])} | {number(r['p95_ms'])} | {number(r['p99_ms'])} | {number(r['texts_per_sec'])} | {r['failures']}/{r['attempts']} |")
        if not load:doc.append('| — | — | N/A | N/A | N/A | N/A | probe 실패; 0 throughput으로 채점하지 않음 |')
        mem=lines(root/'docker-memory-samples.jsonl') if (root/'docker-memory-samples.jsonl').exists() else []
        vals=[mb(r['stats']['MemUsage']) for r in mem];vals=[v for v in vals if v is not None]
        gpu=read(root/'gpu_samples.json') if (root/'gpu_samples.json').exists() else []
        gv=[float(r['gpu_memory_and_utilization'].split(',')[0]) for r in gpu if r['gpu_memory_and_utilization']]
        doc += ['',f"샘플링한 서버 cgroup RAM 최대: {number(max(vals)/1024**2) if vals else 'N/A'} MiB. 전체 device VRAM 최대: {number(max(gv)) if gv else 'N/A'} MiB (Windows display 프로세스 포함)."]
    doc += ['', '## 마지막 CPU/GPU 별도 비교','',
            '엔진 후보를 좁힌 뒤 동일 FP32 SentenceTransformer dense 컴포넌트를 CPU 전용 컨테이너와 GPU 컨테이너에서 따로 실행했다. device마다 모델을 켜고 batch 1/8/32 각 30요청을 실행한 뒤 컨테이너를 종료했다. 이 측정도 엔진+reranker 전체 서비스 비교와 구분한다.','',
            '| device | batch | p50 ms | p95 ms | p99 ms | texts/s | peak process RSS MiB | Torch VRAM MiB |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for device in ['cpu','cuda']:
        hw=read(OUT/f'hardware/{device}.json')
        for r in hw['results']:doc.append(f"| {device} | {r['batch']} | {number(r['p50_ms'])} | {number(r['p95_ms'])} | {number(r['p99_ms'])} | {number(r['texts_per_second'])} | {number(r['peak_process_rss_bytes']/1024**2)} | {number(r['peak_torch_vram_bytes']/1024**2)} |")
    parity_hw=read(OUT/'hardware/cpu-gpu.json')
    doc += ['',f"CPU thread 8, query {parity_hw['query_count']}개, CPU/GPU 최소 vector cosine {parity_hw['parity_min_cosine']:.8f}. p99는 30요청 탐색 측정이므로 장시간 tail latency 추정으로 쓰지 않는다.",
            '', '## 재현·검증과 다음 데이터 버전','',
            '재현 명령과 각 이미지·포트는 `DOCKER_실행_안내.md`와 compose.yaml에 있다. Dataset manifest 검증, 설계용 20건 스키마·의미 계약, 4개 source hash, 지표 unit test 22개를 확인했다. 실제 런은 pending source labels를 명시한 탐색 경로이며 승인 데이터 검증을 우회해 승인 상태로 바꾸지 않았다.','',
            '현 정책을 운영에 옮기지 않는다. 다음 버전에서 GT/조건·상품명·한영 문장 쌍을 독립 검수하고 실제 실패 로그를 추가한 뒤, NO_FAQ와 기존 회귀의 실패를 calibration에서 개선하고 새로운 holdout으로 확인해야 한다. 현재 holdout을 다시 threshold 튜닝에 사용하지 않는다.','',
            'Jina의 일반 컨테이너 custom Python 실행은 자동 승인 검토에서 파일 접근·외부 전송 위험으로 거절됐다. 이후 네트워크 차단과 쓰기 범위 제한을 적용한 별도 컨테이너 실행은 승인되어 완료했다.','']
    (OUT/'실행결과_20261006.md').write_text('\n'.join(doc),encoding='utf-8')
    save(OUT/'execution-ledger-summary.json',{'source_rows':len(ledger),'by_status':dict(Counter(r['status'] for r in ledger)),
         'quality_stages_completed':['A','B','C','D_BGE_Q','D_BGE_QA','D_JINA_Q','D_JINA_QA','E','F'],
         'holdout_policy_evaluation_count':1,'production_adoption':False,'generation_executed':False,
         'source_manifest_hash':digest(DATA/'split_manifest.json'),'selection_hash':digest(OUT/'selection.json'),
         'report':'실행결과_20261006.md'})
    print('Measured report and all 3000 source-row statuses written',flush=True)

if __name__=='__main__':main()
