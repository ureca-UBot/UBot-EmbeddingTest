"""Audit and compare 33 matrices, streaming one engine/structure at a time."""
import csv,gc,hashlib,json
from collections import Counter
import numpy as np
from common import ROOT,OUT,N,NAMES,inputs,read,lines,save,save_lines,sha
ENGINES=['ollama','vllm','tei']
def csv_save(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def pct(v):return '미집계' if v is None else f'{100*v:.2f}%'
def first_rank(row,gold):return min([row['top_ids'].index(v)+1 for v in gold if v in row['top_ids']] or [999])
def audit(rows,cs):
    assert len(rows)==N and [r['case_id'] for r in rows]==[c['case_id'] for c in cs]
    assert [r['query'] for r in rows]==[c['query'] for c in cs]
    assert Counter(r['type'] for r in rows)==Counter(c['type'] for c in cs)
    assert all(r['query_hash']==hashlib.sha256(c['model_query'].encode()).hexdigest() for r,c in zip(rows,cs))
def main():
    cs,ds=inputs();manifest=read(ROOT/'data/manifest.json');validation=read(OUT/'input_validation.json')
    ko={c['parent_case_id']:i for i,c in enumerate(cs) if c['variant_index']==1}
    result={};table=[];types=[];purposes=[];languages=[];failcounts={};regcounts={}
    for engine in ENGINES:
        base=OUT/'engines'/engine;assert read(base/'stop_verification.json')['verified_stopped']
        selection=read(base/'selection.json');assert selection['all_30000_rows_executed'] and selection['stages']==11
        ex=read(base/'embedding_execution.json');assert ex['distinct_execution_rows']==N and not ex['hf_dense_fallback']
        requests=lines(base/'embedding_requests.jsonl');requests=[r for r in requests if r.get('success') and r['role']=='query_all_30000']
        assert {i for r in requests for i in range(r['first_row'],r['first_row']+r['texts'])}==set(range(N))
        policies=read(base/'policies.json');summaries={};cr=lines(base/'C-M20/per_query.jsonl');audit(cr,cs)
        for name in NAMES:
            s=read(base/name/'summary.json');rr=lines(base/name/'per_query.jsonl');audit(rr,cs);assert s['case_count']==N
            summaries[name]=s;p=policies[name];g=s['cohort']['general']['family_mean'];t=s['cohort']['condition']['family_mean']
            accepted=np.array([r['top_scores'][0]>=p['threshold'] and r['top_scores'][0]-r['top_scores'][1]>=p['margin'] for r in rr])
            table.append({'engine':engine,'structure':name,'rows_executed':N,'general_hit1':g['hit@1'],'general_hit3':g['hit@3'],
                          'general_all_sources3':g['all_sources@3'],'general_all_sources10':g['all_sources@10'],
                          'general_all_sources20':g['all_sources@20'],'condition_all_sources3':t['all_sources@3'],
                          **{k:p[k] for k in ['no_faq_f1','no_faq_precision','no_faq_recall','no_faq_false_acceptance',
                                             'answerable_acceptance','answerable_accepted_and_all_sources3','threshold','margin']}})
            for kind,out,key in [('type',types,'type'),('validation_tag',purposes,'tag')]:
                for label,v in s[kind].items():
                    eligible=[i for i,c in enumerate(cs) if c['cohort'] in ['general','condition','no_faq'] and
                              (label in c['validation_tags'] if kind=='validation_tag' else c['type']==label)]
                    negative=[i for i in eligible if cs[i]['no_faq_truth']];positive=[i for i in eligible if not cs[i]['no_faq_truth']]
                    out.append({'engine':engine,'structure':name,key:label,'rows':v['rows'],'ranking_rows':v['ranking_rows'],
                                'calibration_policy_rows':len(eligible),'no_faq_rows':len(negative),
                                'no_faq_false_acceptance':float(np.mean(accepted[negative])) if negative else None,
                                'answerable_rows':len(positive),'answerable_false_rejection':float(np.mean(~accepted[positive])) if positive else None,
                                **v['family_mean']})
            lp=[];fails=[];reg=[]
            for i,(c,b,a) in enumerate(zip(cs,cr,rr)):
                gold=c['source_ids'];category=None
                if c['primary_variation']=='KR_EN_EQUIVALENT':
                    k=rr[ko[c['parent_case_id']]]
                    lp.append({'ko_case_id':k['case_id'],'mixed_case_id':a['case_id'],'top1_same':k['top_ids'][0]==a['top_ids'][0],
                               'ko_all_sources3':k['metrics']['3']['all_sources'],'mixed_all_sources3':a['metrics']['3']['all_sources'],
                               'ko_top3':k['top_ids'][:3],'mixed_top3':a['top_ids'][:3]})
                if c['cohort']=='no_faq':
                    reject=a['top_scores'][0]<p['threshold'] or a['top_scores'][0]-a['top_scores'][1]<p['margin']
                    if not reject:category='NO_FAQ_FALSE_POSITIVE'
                elif gold and not a['metrics']['3']['all_sources']:
                    category='RETRIEVAL_MISS' if not a['metrics']['20']['all_sources'] else 'RANKING_ERROR'
                if category:fails.append({'case_id':c['case_id'],'type':c['type'],'category':category,
                                          'validation_tags':c['validation_tags'],'source_ids':gold,'top3':a['top_ids'][:3],
                                          'source_id_proxy_requires_review':True})
                if name.startswith('D-') and gold:
                    before=first_rank(b,gold);after=first_rank(a,gold)
                    status='STAGE1_MISS' if before==999 else 'IMPROVED' if after<before else 'WORSENED' if after>before else 'UNCHANGED'
                    reg.append({'case_id':c['case_id'],'type':c['type'],'baseline_rank':None if before==999 else before,
                                'reranked_rank':None if after==999 else after,'rank_delta':None if before==999 else before-after,'status':status,
                                'all_sources3_before':b['metrics']['3']['all_sources'],'all_sources3_after':a['metrics']['3']['all_sources']})
            save_lines(base/name/'language_pair_deltas.jsonl',lp);save_lines(base/name/'failures.jsonl',fails)
            valid=[r for r in lp if r['ko_all_sources3'] is not None]
            languages.append({'engine':engine,'structure':name,'pairs':len(lp),'ranking_pairs':len(valid),
                              'ko_basis':'executed V01 Korean paraphrase, not unexecuted original query',
                              'top1_agreement':float(np.mean([r['top1_same'] for r in lp])),
                              'ko_all_sources3':float(np.mean([r['ko_all_sources3'] for r in valid])),
                              'mixed_all_sources3':float(np.mean([r['mixed_all_sources3'] for r in valid])),
                              'mixed_improvements3':sum(r['mixed_all_sources3']>r['ko_all_sources3'] for r in valid),
                              'mixed_regressions3':sum(r['mixed_all_sources3']<r['ko_all_sources3'] for r in valid)})
            failcounts[engine+'/'+name]=dict(Counter(r['category'] for r in fails))
            if reg:save_lines(base/name/'reranker_deltas.jsonl',reg);regcounts[engine+'/'+name]=dict(Counter(r['status'] for r in reg))
            del rr,lp,fails,reg;gc.collect();print(json.dumps({'audited':engine+'/'+name,'rows':N}),flush=True)
        stability=read(base/'repeat_stability.json')
        result[engine]={'selection':selection,'structures':summaries,'embedding_execution':ex,
                        'repeat_stability':{name:{'groups':len(groups),'unstable_rank_groups':sum(v['rank_hashes']>1 for v in groups.values()),
                                                 'max_top1_score_range':max(v['top1_score_range'] for v in groups.values())}
                                           for name,groups in stability.items()}}
        del cr;gc.collect()
    for filename,values in [('comparison.csv',table),('per_type_comparison.csv',types),('per_purpose_comparison.csv',purposes),('language_comparison.csv',languages)]:csv_save(filename,values)
    save(OUT/'comparison.json',result);save(OUT/'failure_counts.json',failcounts);save(OUT/'reranker_regression_counts.json',regcounts)
    deltas={};changed=[]
    for name in NAMES:
        reference=lines(OUT/'engines/ollama'/name/'per_query.jsonl')
        for engine in ['vllm','tei']:
            rr=lines(OUT/'engines'/engine/name/'per_query.jsonl');improved=worsened=0
            for a,b in zip(reference,rr):
                before=a['metrics']['3']['all_sources'];after=b['metrics']['3']['all_sources']
                if before is None:continue
                improved+=int(after>before);worsened+=int(after<before)
                if after!=before:changed.append({'structure':name,'engine':engine,'case_id':a['case_id'],'type':a['type'],
                                                'primary_variation':a['primary_variation'],'ollama_all_sources3':before,'engine_all_sources3':after,
                                                'ollama_top3':a['top_ids'][:3],'engine_top3':b['top_ids'][:3]})
            deltas[name+'/'+engine]={'paired_rows':N,'top1_agreement_with_ollama':sum(a['top_ids'][0]==b['top_ids'][0] for a,b in zip(reference,rr))/N,
                                    'top10_overlap_with_ollama':sum(len(set(a['top_ids'][:10])&set(b['top_ids'][:10]))/10 for a,b in zip(reference,rr))/N,
                                    'improvements_all_sources3':improved,'regressions_all_sources3':worsened}
            del rr;gc.collect()
        del reference;gc.collect()
    save(OUT/'paired_deltas.json',deltas);save_lines(OUT/'changed_source_coverage.jsonl',changed)
    parity={};reference=np.load(OUT/'engines/ollama/cache/query_all_30000.npy',mmap_mode='r');ql=np.load(OUT/'query_token_lengths.npy')
    for engine in ['vllm','tei']:
        q=np.load(OUT/'engines'/engine/'cache/query_all_30000.npy',mmap_mode='r');cos=np.empty(N,dtype=np.float32)
        for i in range(0,N,256):cos[i:i+256]=np.sum(q[i:i+256]*reference[i:i+256],axis=1)
        parity[engine]={key:{'rows':int(mask.sum()),'min_cosine':float(cos[mask].min()),'mean_cosine':float(cos[mask].mean())}
                        for key,mask in [('all',np.ones(N,dtype=bool)),('above_512',ql>512),('at_most_512',ql<=512)]}
    save(OUT/'dense_vector_parity.json',parity)
    text=['# 30,000행 엔진별 Calibration 비교','',
          '동일한 3,000쌍×10변형을 Ollama·vLLM·TEI 각각의 BGE-M3 HTTP API로 전부 실행했다. 원본 15유형×2,000행, 행 순서, 반복 2,000행을 보존했다. 엔진당 11개 구조로 총 990,000행의 검색 결과를 계산했다.','',
          f"입력 SHA256: `{manifest['source_variants_sha256']}`. Query 최대 {validation['queries_max_tokens']}토큰이며 512토큰 초과 {validation['over_old_512_limit']:,}행도 원문 전체를 넣었다. 공통 입력 검증 범위는 1,024토큰이다. Ollama/vLLM은 1,024토큰 설정, TEI는 원래 8,192토큰 모델 설정을 썼다. Dense·reranker·native 입력 truncation은 없다.",'',
          '독립 의미·Ground Truth 검수는 미완료다. 지표는 추적 FAQ ID를 이용한 SourceHit·SourceCoverage·AllSourcesHit이며 atomic-fact 또는 생성 정답률이 아니다. 같은 Calibration 데이터로 구조와 임계값을 선택했으며 Holdout 결과가 아니다.','',
          '## 실행 구조','',
          'FAQ 원문 1,024개를 검색했다. A는 질문 dense, B는 질문+답변 dense, C는 두 view Top-20 union에서 FAQ ID 중복 제거 후 max cosine으로 20개 선택한다. D는 각 엔진 C 후보에 BGE/Jina×질문/질문+답변 네 조합을 적용한다. D 조합당 600,000 pairs, 총 7,200,000 pairs를 계산했다. 질의 및 반복의 inference 중복 제거는 없다.','',
          'D는 공통 HF/PyTorch CUDA FP16, E/F는 공통 FlagEmbedding CUDA FP32다. 각 D 조합의 FP32/FP16 표본 점수 차이는 precision-probe 파일에 기록했으며 두 정밀도가 같은 품질이라고 가정하지 않는다. E hybrid와 F pool의 dense 후보는 해당 엔진 API 결과다. E sparse/F full은 별도로 재계산한 공통 대조군이다. 엔진 자체 sparse/multi-vector API를 비교한 결과로 해석하지 않는다. Ollama GGUF F16과 vLLM/TEI FP32의 포맷·정밀도 차이도 포함한다.','',
          '엔진별 기동→readiness→전체 HTTP 실행→종료→HTTP 폐쇄·활성 컨테이너 0 확인 후 보조 모델을 실행하고 다음 엔진으로 넘어갔다. 가중치·이미지 외에는 기존 실험 또는 다른 엔진의 inference 결과를 재사용하지 않았다. 입력은 질문과 고정 원본 대화 이력뿐이며 제공 Context·정답·source ID·API 응답·persona는 제외했다.','',
          '## 같은 구조에서 엔진 비교','',
          '| 구조 | Ollama 일반 AllSources@3 | vLLM 일반 AllSources@3 | TEI 일반 AllSources@3 |','|---|---:|---:|---:|']
    for name in NAMES:text.append('| '+name+' | '+' | '.join(pct(result[e]['structures'][name]['cohort']['general']['family_mean']['all_sources@3']) for e in ENGINES)+' |')
    text+=['','일반 집계는 SF·NC·PS 6,000행, 조건 집계는 CE·MC 4,000행의 source family별 평균이다. RT를 일반 성능 평균에 넣지 않고 200개의 동일 입력 그룹×10회 안정성으로 평가했다. 나머지 유형과 Context 부족 100행도 실행하며 별도 집계한다.','']
    for engine in ENGINES:
        text += [f'## {engine}','',
                 '| 구조 | 일반 Hit@1 | 일반 Hit@3 | 일반 AllSources@3 | 조건 AllSources@3 | NO_FAQ F1 |',
                 '|---|---:|---:|---:|---:|---:|']
        for r in [r for r in table if r['engine']==engine]:text.append(f"| {r['structure']} | {pct(r['general_hit1'])} | {pct(r['general_hit3'])} | {pct(r['general_all_sources3'])} | {pct(r['condition_all_sources3'])} | {r['no_faq_f1']:.4f} |")
        sel=result[engine]['selection'];p=sel['policy'];l=next(r for r in languages if r['engine']==engine and r['structure']==sel['selected'])
        st=result[engine]['repeat_stability'][sel['selected']]
        text+=['',f"선정 구조 **{sel['selected']}**, score threshold **{p['threshold']:.8f}**, margin **{p['margin']:.8f}**. `top1 < threshold OR top1-top2 < margin`이면 NO_FAQ로 거절한다. 일반·조건 10,000 + NO_FAQ 5,900, 합계 15,900행에서 F1 최대·동률 시 정답 질문 채택률 최대 기준으로 계산했다. Calibration F1={p['no_faq_f1']:.4f}, precision={p['no_faq_precision']:.4f}, recall={p['no_faq_recall']:.4f}이며 별도 Holdout은 없다. 임계값은 다른 구조에 이식하지 않는다.",
               f"KR_EN_EQUIVALENT {l['pairs']:,}쌍은 실제 실행한 같은 부모의 V01 한국어 변형을 대응 입력으로 썼다. 원본 한국어 질문을 따로 실행했다고 주장하지 않는다. 한국어 AllSources@3={pct(l['ko_all_sources3'])}, 혼용어={pct(l['mixed_all_sources3'])}; 개선 {l['mixed_improvements3']:,}, 악화 {l['mixed_regressions3']:,}쌍이다. 독립 번역 정확도 검수 결과는 아니다.",
               f"선정 구조 반복 200그룹 중 rank가 바뀐 그룹 {st['unstable_rank_groups']}개, Top-1 점수 최대 범위 {st['max_top1_score_range']:.8g}. 실제 HTTP 요청 실패 {result[engine]['embedding_execution']['request_failures']}건. Batch latency는 순차 가변 batch 진단값이며 동시 부하 성능 비교가 아니다.",'']
    text+=['## 상세 결과','',
           '`comparison.csv`는 33개 구조 지표·임계값·NO_FAQ 오류, `per_type_comparison.csv`와 `per_purpose_comparison.csv`는 유형/목적별 결과, `language_comparison.csv`는 한국어/혼용어 비교다. 각 구조 `per_query.jsonl`·`failures.jsonl`, D `reranker_deltas.jsonl`에는 전체 순위·후보 누락·개선/악화가 있다. `paired_deltas.json`·`changed_source_coverage.jsonl`·`dense_vector_parity.json`은 엔진 간 변화, 엔진별 `repeat_stability.json`·`embedding_requests.jsonl`·`stop_verification.json`은 실행 증빙이다.','',
           'REAL_FAILURE 실제 로그 0행, 문서 공격 fixture·API 호출·생성·CF 통제 정책 최종 답변·persona 검증 미실행, 독립 annotation 미완료라는 범위에서 해석한다. 한국어 V01과 혼용어는 같은 부모의 표현 변형이므로 영어 표기만의 인과효과로 해석하지 않는다.','',
           'BGE reranker의 FP16 설정과 수치 차이 가능성은 [공식 모델 카드](https://huggingface.co/BAAI/bge-reranker-v2-m3)에 설명되어 있다. Jina의 1,024토큰 입력 및 표준 attention 사용 방식은 [공식 모델 카드](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)를 확인했다.','']
    report=OUT/'엔진별_Calibration_30000_비교.md';report.write_text('\n'.join(text),encoding='utf-8')
    save(OUT/'completion_audit.json',{'engines':3,'structures_per_engine':11,'case_rows_per_structure':N,'case_structure_results':990000,
         'reranker_pairs_total':7200000,'full_queries_and_order_verified':True,'type_counts_each_engine':manifest['type_counts'],
         'repeat_rows_each_structure':2000,'all_engines_stopped':True,'hf_dense_fallback':False,'input_truncated':False,
         'holdout_executed':False,'independent_annotation_complete':False,'input_sha256':sha(ROOT/'data/cases.jsonl')})
    save(OUT/'execution_code_identity.json',{'scripts':{p.name:sha(p) for p in (ROOT/'scripts').glob('*.py')},
                                          'compose_sha256':sha(ROOT/'compose.yaml'),'run_all_sha256':sha(ROOT/'run_all.ps1')})
    save(OUT/'artifact_inventory.json',{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_inventory.json'})
    print(str(report),flush=True)
if __name__=='__main__':main()
