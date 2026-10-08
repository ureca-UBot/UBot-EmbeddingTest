"""Completion audit and direct matched-input engine/architecture report."""
import json
from collections import Counter, defaultdict
import numpy as np
from common import ROOT, DATA, OUT, FIELDS, read, lines, save, save_lines, sha, inputs, doc_text, rankings

STRUCTURES=['A','B','C-union-K20','D-bge-question','D-bge-question_answer','D-jina-question','D-jina-question_answer']

def memory_summary(base):
    records=lines(base/'memory_samples.jsonl')
    gpu=[];cpu=[]
    for r in records:
        try: gpu.append(float(r['host_gpu_memory_utilization'].split(',')[0]))
        except (ValueError,AttributeError,KeyError): pass
        try:
            value=json.loads(r['docker_stats'])['MemUsage'].split('/')[0].strip()
            for suffix,mult in [('GiB',1024**3),('MiB',1024**2),('KiB',1024),('GB',10**9),('MB',10**6),('kB',1000),('B',1)]:
                if value.endswith(suffix): cpu.append(float(value[:-len(suffix)])*mult);break
        except (ValueError,TypeError,KeyError): pass
    return {'samples':len(records),'peak_sampled_gpu_mib':max(gpu,default=None),
            'peak_sampled_container_memory_bytes':max(cpu,default=None),
            'scope':'host GPU total and container cgroup memory, sampled approx every 4 seconds; not per-process GPU allocation'}

def main():
    cases,docs=inputs()
    from control_gt import revised_cases,reaggregate
    cases=revised_cases(cases,docs,write_outputs=False)
    reaggregate(cases,docs,STRUCTURES)
    byid={d['faq_id']:i for i,d in enumerate(docs)}
    case_indices={c['case_id']:i for i,c in enumerate(cases)}
    comparison=[];audits={};distributions={};engine_results={};language_controls=[];purpose_controls=[]
    report=['# Ollama · vLLM 서빙 재테스트','',
        'FAQ 원문 1,024개를 대상으로 새 합성 질문 500쌍×10개=5,000개와 목적별 통제 질문 21개를 두 엔진 모두에 실제 입력했다. 기존 평가 질문·기존 모델 추론 결과는 사용하지 않았다.',
        '', '질문은 README 검증 목적을 기준으로 작성했다. 주 평가 5,000개는 FAQ 질문과 어휘가 많이 겹치는 자동 표면 변형이다. 자동 변형 및 원천 FAQ ID 기반 GT는 독립적인 의미·대체 정답 검수가 미완료다. 따라서 높은 검색 수치가 실제 서비스 질문의 성능을 입증하지 않으며 아래 수치는 진단값으로만 해석한다. 조건·복수 사실·한영 의미 대조 등 목적별 통제 21개는 따로 추적한다.',
        '', '결과 확인 중 프리미엄 월 기본요금의 대체 FAQ와 비교 문항의 대체 사실을 빠뜨린 라벨을 발견해 통제 문항 세 개를 수정했다. `data/control_gt_corrections.json`에 원래 라벨·새 라벨·FAQ 원문 근거를 보존했다. 질문, 코퍼스, 모델 점수는 바꾸지 않았고 최초 로그는 `.initial_gt` 파일로 보존했다. 아래 지표와 후보 로그는 보완한 사실 GT를 사용한다.',
        '', '## 검색 구조별 진단 결과','',
        '| 엔진 | 구조 | 핵심 5,000 Hit@1 | Hit@3 | FactRecall@10 | 전체 문항 |',
        '|---|---|---:|---:|---:|---:|']
    for engine in ['ollama','vllm']:
        base=OUT/'engines'/engine
        assert read(base/'stop_verification.json')['verified_stopped']
        load=read(base/'serving_load.json');assert [r['concurrency'] for r in load['runs']]==[1,4,8,16,32]
        execution=read(base/'embedding_execution.json');assert execution['query_rows_sent']==len(cases)
        reqs=[r for r in lines(base/'embedding_requests.jsonl') if r['role']=='queries' and r['success']]
        count=Counter(i for r in reqs for i in range(r['first_row'],r['first_row']+r['texts']))
        assert set(count)==set(range(len(cases))) and set(count.values())=={1}
        pools=read(base/'cache/C-pools.json');expected_rows=sum(map(len,pools))
        a=np.load(base/'cache/A-scores.npy');b=np.load(base/'cache/B-scores.npy')
        ar,br=rankings(a),rankings(b);ai,bi=np.empty_like(ar),np.empty_like(br)
        rank_values=np.broadcast_to(np.arange(len(docs))+1,ar.shape)
        np.put_along_axis(ai,ar,rank_values,axis=1);np.put_along_axis(bi,br,rank_values,axis=1)
        log_audits=[];engine_rows=[]
        for structure in STRUCTURES:
            summary=read(base/structure/'summary.json')
            assert summary['inference_rows']==len(cases) and summary['main_rows']==5000
            main_group=summary['groups']['main_core']['row_mean']
            entry={'engine':engine,'structure':structure,**main_group}
            comparison.append(entry);engine_rows.append(entry)
            if structure.startswith('C-'):
                # C@K is per-view K union, not a K-row ranked prefix.
                entry['candidate_metric_semantics']='per-view K complete union'
            label=structure+' (각 뷰 K=20 합집합)' if structure.startswith('C-') else structure
            report.append(f"| {engine} | {label} | {main_group['hit@1']*100:.2f}% | {main_group['hit@3']*100:.2f}% | {main_group['fact_recall@10']*100:.2f}% | {len(cases)} |")
            score_rows=lines(base/structure/'per_query.jsonl')
            for row in score_rows:
                if row['evaluation_role']=='purpose_control':
                    purpose_controls.append({'engine':engine,'structure':structure,**row})
            language_cases=[(i,c) for i,c in enumerate(cases) if c.get('language_pair_id')]
            if language_cases:
                score_matrix=a if structure=='A' else b if structure=='B' else np.maximum(a,b) if structure.startswith('C-') else np.load(base/'cache'/f'{structure}-scores.npy',mmap_mode='r')
                for i,case in language_cases:
                    rr=list(map(int,rankings(score_matrix[i:i+1])[0])) if structure in ['A','B'] else (
                        pools[i] if structure.startswith('C-') else sorted(pools[i],key=lambda j:(-float(score_matrix[i,j]),j)))
                    gold={fid for fact in case['required_facts'] for fid in fact['acceptable_faq_ids']}
                    rank=min((p+1 for p,j in enumerate(rr) if docs[j]['faq_id'] in gold),default=None)
                    language_controls.append({'engine':engine,'structure':structure,'pair_id':case['language_pair_id'],
                        'case_id':case['case_id'],'language':case['language'],'query':case['query'],
                        'first_gold_rank':rank,'hit3':rank is not None and rank<=3,
                        'top3':[docs[j]['faq_id'] for j in rr[:3]],'stage1_miss':rank is None})
            score1=np.asarray([r['top_scores'][0] for r in score_rows[:5000]])
            margin=np.asarray([r['top_scores'][0]-r['top_scores'][1] for r in score_rows[:5000]])
            distributions[engine+'/'+structure]={'top1':{str(p):float(np.quantile(score1,p)) for p in [0,.01,.05,.5,.95,.99,1]},
                'margin':{str(p):float(np.quantile(margin,p)) for p in [0,.01,.05,.5,.95,.99,1]},
                'threshold':None,'margin_threshold':None,'status':'NOT_FITTED_UNREVIEWED_NO_FAQ',
                'reason':'Two source-control NO_FAQ examples are not independently reviewed calibration/holdout negatives.'}
            if not structure.startswith('D-'): continue
            scores=np.load(base/'cache'/f'{structure}-scores.npy',mmap_mode='r')
            schema=read(base/structure/'candidate_log_schema.json')
            assert schema['requested_fields']==FIELDS and schema['row_count']==expected_rows
            logpath=base/structure/'candidate_log.jsonl'
            assert sha(logpath)==schema['sha256']
            view='question_answer' if structure.endswith('question_answer') else 'question'
            row_count=0;seen_cases=Counter();seen_ranks=defaultdict(set)
            with logpath.open(encoding='utf-8') as stream:
                for line in stream:
                    row=json.loads(line);i=case_indices[row['case_id']];j=byid[row['faq_id']]
                    assert all(f in row for f in FIELDS)
                    assert row['query']==cases[i]['query'] and row['model_query']==cases[i]['model_query']
                    assert pools[i][row['retrieval_rank']-1]==j
                    assert row['rank_delta']==row['retrieval_rank']-row['rerank_rank']
                    assert row['retrieval_a_rank']==int(ai[i,j]) and row['retrieval_b_rank']==int(bi[i,j])
                    assert row['retrieval_a_cosine']==float(a[i,j]) and row['retrieval_b_cosine']==float(b[i,j])
                    assert row['retrieval_cosine']==float(max(a[i,j],b[i,j]))
                    assert row['rerank_score']==float(scores[i,j]) and row['winning_window_score']==row['rerank_score']
                    assert row['window_count']==1 and row['winning_window_index']==0
                    assert row['winning_window_text']==doc_text(docs[j],view)
                    seen_cases[i]+=1;seen_ranks[i].add(row['rerank_rank']);row_count+=1
            assert row_count==expected_rows and len(seen_cases)==len(cases)
            assert all(seen_cases[i]==len(pools[i]) and seen_ranks[i]==set(range(1,len(pools[i])+1)) for i in range(len(cases)))
            log_audits.append({'structure':structure,'rows':row_count,'all_16_requested_fields_verified':True,
                'full_document_text_verified':True,'entire_corpus_AB_rank_verified':True,'rank_delta_verified':True,
                'rank_changes':schema['rank_changes']})
        language=lines(base/'language_pair_results.jsonl')
        repeat=read(base/'repeat_summary.json')
        engine_results[engine]={'embedding':execution,'serving':load,'memory':memory_summary(base),'repeat':repeat,
            'language_pairs':{'pairs':len(language),'ko_hit3':float(np.mean([r['ko_metrics']['hit'] for r in language])),
                'variant_hit3':float(np.mean([r['variant_metrics']['hit'] for r in language])),
                'mean_rank_delta':float(np.mean([r['rank_delta'] for r in language])),
                'mean_top20_overlap':float(np.mean([r['top20_overlap'] for r in language]))},
            'stop_verified':True,'readiness':read(base/'readiness.json')}
        audits[engine]={'full_matched_input_run':True,'main_rows':5000,'purpose_controls':len(cases)-5000,
            'candidate_pairs_per_D':expected_rows,'D_logs':log_audits,'serving_concurrencies_complete':True,
            'independent_semantic_review_complete':False}
    report+=['', 'C의 Hit@K는 각 뷰 Top-K 합집합의 Candidate Recall이다. D의 Hit@K는 C 후보 전체를 리랭킹한 뒤 상위 K개다. C는 중복 제거 후 재절삭하지 않았으며, max(A,B) cosine은 후보의 진단용 순서에만 사용했다.',
        '', '## 서빙 부하 테스트','',
        '각 동시성에서 같은 질문 160개를 warm 상태로 측정했다. HTTP 요청 하나당 질문 하나이며 클라이언트가 응답을 기다리는 closed-loop 측정이다. p95는 임베딩 API 왕복 시간으로 검색·리랭킹 전체 응답 시간은 아니다.',
        '', '| 엔진 | 동시 요청 | 성공/실패 | 요청/초 | p50 ms | p95 ms | p99 ms | Top1 일치율 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for engine,value in engine_results.items():
        for r in value['serving']['runs']:
            latency=r['latency_seconds']
            report.append(f"| {engine} | {r['concurrency']} | {r['successes']}/{r['failures']} | {r['requests_per_second']:.2f} | {latency['p50']*1000:.1f} | {latency['p95']*1000:.1f} | {latency['p99']*1000:.1f} | {r['top1_consistency']*100:.2f}% |")
    report+=['', '## 한영 동등 표현 · 반복 · 메모리','',
        '| 엔진 | 한영 표기 쌍 | 한국어 Hit@3 | 변형 Hit@3 | 평균 Rank Delta | 반복의 서로 다른 Context Hash | 샘플 GPU 최고 MiB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for engine,r in engine_results.items():
        p=r['language_pairs']
        report.append(f"| {engine} | {p['pairs']} | {p['ko_hit3']*100:.2f}% | {p['variant_hit3']*100:.2f}% | {p['mean_rank_delta']:.2f} | {r['repeat']['distinct_context_hashes']} | {r['memory']['peak_sampled_gpu_mib']} |")
    aq=np.load(OUT/'engines/ollama/cache/queries.npy');vq=np.load(OUT/'engines/vllm/cache/queries.npy')
    cross_cos=np.sum(aq*vq,axis=1)
    cross_rank={}
    for structure in STRUCTURES:
        left=lines(OUT/'engines/ollama'/structure/'per_query.jsonl')
        right=lines(OUT/'engines/vllm'/structure/'per_query.jsonl')
        assert [r['case_id'] for r in left]==[r['case_id'] for r in right]
        cross_rank[structure]={'cases':len(left),
            'top1_same_fraction':float(np.mean([x['top_ids'][0]==y['top_ids'][0] for x,y in zip(left,right)])),
            'mean_top3_overlap':float(np.mean([len(set(x['top_ids'][:3])&set(y['top_ids'][:3]))/3 for x,y in zip(left,right)])),
            'top3_identical_order_fraction':float(np.mean([x['top_ids'][:3]==y['top_ids'][:3] for x,y in zip(left,right)]))}
    cross={'query_embedding_cosine_quantiles':{str(p):float(np.quantile(cross_cos,p)) for p in [0,.01,.5,.95,1]},
           'rank_consistency':cross_rank,
           'comparison_limit':'Ollama cached GGUF F16 vs vLLM HF float32; model format/precision differs, not isolated engine effect'}
    report+=['', '## 별도 한영 의미 동등 통제 질문','',
        '아래 두 질문은 삭제한 eSIM의 앱 재발급이라는 같은 사실 GT를 공유한다. 위 124개 단순 용어 치환 쌍과 별도로 해석한다. A/B rank는 전체 코퍼스, C/D rank는 C 후보 합집합 안의 순위다.',
        '', '`실수로 삭제한 이심은 앱에서 어떻게 다시 발급받나요?`',
        '', '`How can I reissue an accidentally deleted eSIM through the app?`',
        '', '| 엔진 | 구조 | 한국어 정답 Rank | 영어 정답 Rank | 한국어 Hit@3 | 영어 Hit@3 |',
        '|---|---|---:|---:|---|---|']
    for engine in ['ollama','vllm']:
        for structure in STRUCTURES:
            ko=next(r for r in language_controls if r['engine']==engine and r['structure']==structure and r['language']=='ko')
            en=next(r for r in language_controls if r['engine']==engine and r['structure']==structure and r['language']=='en')
            report.append(f"| {engine} | {structure} | {ko['first_gold_rank']} | {en['first_gold_rank']} | {ko['hit3']} | {en['hit3']} |")
    report+=['', '## 엔진 간 벡터·순위 일관성','',
        f"전체 질문의 엔진 간 벡터 cosine 최솟값은 {float(cross_cos.min()):.8f}, 중앙값은 {float(np.median(cross_cos)):.8f}다. 엔진 내부 동시성 변화에서 순위가 유지되는 것과 두 엔진의 벡터가 완전히 같은 것은 구별한다.",
        '', '| 구조 | 엔진 간 Top1 ID 일치율 | 평균 Top3 겹침 | Top3 순서까지 일치율 |',
        '|---|---:|---:|---:|']
    for name,r in cross_rank.items():
        report.append(f"| {name} | {r['top1_same_fraction']*100:.2f}% | {r['mean_top3_overlap']*100:.2f}% | {r['top3_identical_order_fraction']*100:.2f}% |")
    report+=['', '## 로그와 실행 범위','',
        '각 엔진의 D 4개 구조에서 모든 후보 FAQ에 요청한 16개 필드를 기록했다. `candidate_log.jsonl`은 query→후보 FAQ 순서이며 `rerank_rank`로 정렬하면 리랭킹 결과를 재현할 수 있다. `rank_changes.jsonl`에 IMPROVED/WORSENED/UNCHANGED/STAGE1_MISS를 기록했다.',
        '', '`expected_faq_id`는 허용 정답 ID 목록이다. 사실별 대체 FAQ와 복수 필요 사실은 `required_facts`로 구별한다. A/B rank는 코퍼스 전체의 1-based rank다. `rank_delta = retrieval_rank - rerank_rank`이며 window_count=1, winning_window_index=0, winning_window_text는 실제 FAQ 전문이다.',
        '', 'Dense는 각 엔진의 HTTP API로 실제 추론했다. D 리랭커는 두 엔진 후보에 공통 HF/PyTorch CUDA FP16 모델을 각각 실행했다. Ollama/vLLM 자체가 리랭커까지 서빙한 결과가 아니다.',
        '', 'Ollama GGUF F16과 vLLM HF float32를 사용했으므로 서빙 소프트웨어뿐 아니라 모델 형식·정밀도의 영향도 포함한다. 배치·동시성 변화에 따른 순위/벡터 일관성은 개별 요청 로그에서 확인할 수 있다.',
        '', '두 엔진 모두 기동/health 확인, 실제 측정, 종료/포트 해제 확인을 완료했다. Threshold는 독립 검수한 NO_FAQ가 부족해 피팅하지 않았고 구조별 Top1·margin 분포만 저장했다. Holdout·LLM 생성·문서 공격의 생성 안전성은 실행하지 않았다. 실제 실패 로그와 정책 버전이 없어 REAL_FAILURE/TEMPORAL_VERSION은 만들지 않았다. Sparse/Multi-vector/TEI/CPU 비교는 이번 실행 범위에 포함하지 않았다.']
    save_lines(OUT/'purpose_control_results.jsonl',purpose_controls)
    save(OUT/'language_control_results.json',language_controls)
    save(OUT/'comparison.json',{'matched_inputs':True,'structures':comparison,'engines':engine_results,'cross_engine':cross,
        'language_controls':language_controls,
        'calibration':'score diagnostics only; no fitted threshold or holdout claims','gt_status':'independent semantic review pending'})
    save(OUT/'score_margin_distributions.json',distributions)
    save(OUT/'completion_audit.json',{'completed':True,'engines':audits,'requested_fields':FIELDS,
        'input_hashes':read(DATA/'manifest.json')['hashes'],'all_engine_stop_checks_passed':True,
        'control_gt_corrections_sha256':sha(DATA/'control_gt_corrections.json'),
        'evaluation_cases_sha256':sha(DATA/'evaluation_cases.jsonl')})
    (OUT/'comparison.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({'completed':True,'structures':len(comparison),'logs':sum(len(a['D_logs']) for a in audits.values()),
        'candidate_rows':sum(sum(l['rows'] for l in a['D_logs']) for a in audits.values()),'report':str(OUT/'comparison.md')}),flush=True)

if __name__=='__main__': main()
