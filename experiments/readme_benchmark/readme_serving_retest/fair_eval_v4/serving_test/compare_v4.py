"""Finalize frozen diagnostics and audit every requested candidate log row."""
import gzip,math
from collections import Counter,defaultdict
import numpy as np
from common_v4 import *

STRUCTURES=('A','B','C-fixed20','C-raw-union','D-bge-question','D-bge-question_answer','D-jina-question','D-jina-question_answer',
 'E-sparse-question','E-sparse-question_answer','E-dense-sparse-rrf','F-multi-question','F-multi-question_answer')
def pct(x):return '—' if x is None else f'{100*x:.2f}%'

def refresh(key,engine,qs,docs):
 root=base(key,engine);a=np.load(root/'cache/A-scores.npy');b=np.load(root/'cache/B-scores.npy');c=read(root/'cache/C-pools.json')
 merged=np.maximum(a,b)
 for name in STRUCTURES:
  prior=read(root/name/'summary.json');execution=prior['execution'];union=None
  if name=='A':scores=a;rr=ranks(a)
  elif name=='B':scores=b;rr=ranks(b)
  elif name=='C-fixed20':scores=merged;rr=[x[:20] for x in c]
  elif name=='C-raw-union':scores=merged;rr=c;union={k:pool(a,b,k) for k in KS}
  elif name.startswith('D-'):
   scores=np.load(OUT/key/'shared_reranker'/name/'scores.npy',mmap_mode='r');rr=[sorted(ids,key=lambda j:(-float(scores[i,j]),j)) for i,ids in enumerate(c)]
  else:
   if name=='E-dense-sparse-rrf':scores=1./(60+inverse(b))+1./(60+inverse(np.load(OUT/key/'native/question_answer-sparse.npy')))
   else:
    view='question_answer' if name.endswith('question_answer') else 'question';kind='sparse' if name.startswith('E-') else 'multi'
    scores=np.load(OUT/key/'native'/f'{view}-{kind}.npy',mmap_mode='r')
   rr=ranks(scores)
  summarize(key,engine,name,qs,docs,rr,scores,execution,union)

def language_pairs(key,engine,name,qs,rows):
 qm={q['question_id']:q for q in qs};rm={r['question_id']:r for r in rows};pairs=defaultdict(list);result=[]
 for q in qs:
  for p in q.get('pair_ids',[]):pairs[p].append(q)
 for pid,members in pairs.items():
  if not any('KR_EN_EQUIVALENT' in q['purpose_tags'] for q in members):continue
  original=next((q for q in members if q['language']=='ko' and q['primary_purpose']!='KR_EN_EQUIVALENT'),None)
  if original is None:original=next((q for q in members if q['language']=='ko'),None)
  if original is None:continue
  for variant in members:
   if variant['question_id']==original['question_id'] or variant['primary_purpose']!='KR_EN_EQUIVALENT':continue
   if original['required_facts']!=variant['required_facts']:continue
   x,y=rm[original['question_id']],rm[variant['question_id']];ar,br=x['first_gold_rank'],y['first_gold_rank']
   result.append(dict(pair_id=pid,original_question_id=original['question_id'],variant_question_id=variant['question_id'],
    original_query=original['query'],variant_query=variant['query'],original_first_gold_rank=ar,variant_first_gold_rank=br,
    rank_delta=ar-br if ar is not None and br is not None else None,top20_overlap=len(set(x['top_ids'])&set(y['top_ids']))/20,
    original_metrics=x['metrics'],variant_metrics=y['metrics']))
 save_lines(base(key,engine)/name/'language_pair_results.jsonl',result)
 return len(result)

def audit_log(key,engine,name,qs,docs):
 root=base(key,engine)/name;schema=read(root/'candidate_log_schema.json');path=root/'candidate_log.jsonl.gz'
 assert sha(path)==schema['sha256'];ids={d['faq_id'] for d in docs};qm={q['question_id']:q for q in qs};counts=Counter();ranks_seen=defaultdict(set);rranks=defaultdict(set);num=0;maxwindows=0
 with gzip.open(path,'rt',encoding='utf-8') as f:
  for line in f:
   row=json.loads(line);assert set(FIELDS)<=set(row)
   q=qm[row['question_id']];assert row['dataset_id']==key and row['engine']==engine and row['structure']==name
   assert row['faq_id'] in ids and set(row['expected_faq_ids'])<=ids
   assert row['expected_faq_ids']==q['answer_faq_ids']==row['expected_faq_id']
   assert row['required_facts']==q['required_facts'] and row['query']==q['query'] and row['model_query']==q['model_query']
   assert row['split']==q['split'];counts[q['question_id']]+=1;ranks_seen[q['question_id']].add(row['retrieval_rank'])
   for field in ['retrieval_a_rank','retrieval_b_rank']:assert 1<=row[field]<=len(docs)
   for field in ['retrieval_a_cosine','retrieval_b_cosine']:assert math.isfinite(row[field]) and -1.00001<=row[field]<=1.00001
   if name.startswith('D-'):
    assert row['rank_delta']==row['retrieval_rank']-row['rerank_rank']
    assert math.isfinite(row['rerank_score']) and row['winning_window_score']==row['rerank_score']
    assert row['window_count']>=1 and 0<=row['winning_window_index']<row['window_count'] and row['winning_window_text']
    rranks[q['question_id']].add(row['rerank_rank']);maxwindows=max(maxwindows,row['window_count'])
   else:assert row['rerank_rank'] is None and row['rerank_score'] is None and row['rank_delta'] is None
   if name.startswith(('E-','F-')):assert row['retrieval_cosine'] is None and math.isfinite(row['retrieval_representation_score'])
   num+=1
 assert num==schema['row_count'] and set(counts)==set(qm)
 for qid,n in counts.items():
  assert ranks_seen[qid]==set(range(1,n+1))
  if name.startswith('D-'):assert rranks[qid]==set(range(1,n+1))
 return {'rows':num,'sha256':schema['sha256'],'question_count':len(counts),'max_windows':maxwindows,'passed':True}

def main():
 all_summary={};audits=[];consistency={};distributions=[];failures=[];load_results={};repeats={};attacks={}
 assert read(OUT/'frozen_manifest.json')==read(DATASET_ROOT/'manifest.json')
 for key in KEYS:
  d,qs,docs=load(key);all_summary[key]={};load_results[key]={};repeats[key]={};attacks[key]={}
  for engine in ENGINES:
   root=base(key,engine);assert read(root/'dense_complete.json')['complete'] and read(root/'stop_verification.json')['verified_stopped']
   assert read(root/'embedding_execution.json')['query_rows']==len(qs)
   if engine=='ollama':assert read(root/'model_details.json')['details']['quantization_level']=='F16'
   refresh(key,engine,qs,docs);all_summary[key][engine]={}
   for name in STRUCTURES:
    directory=root/name;s=read(directory/'summary.json');rows=lines(directory/'per_query.jsonl')
    assert len(rows)==len(qs);all_summary[key][engine][name]=s
    lp=language_pairs(key,engine,name,qs,rows)
    audits.append(dict(dataset_id=key,engine=engine,structure=name,language_pairs=lp,**audit_log(key,engine,name,qs,docs)))
    for split in ('development','calibration','holdout'):
     for status in ('FAQ_EXISTS','NO_FAQ','PARTIAL_FACTS_EXIST','UNDERSPECIFIED','NOT_APPLICABLE'):
      selected=[r for r in rows if r['split']==split and r['corpus_status']==status]
      distributions.append(dict(dataset_id=key,engine=engine,structure=name,split=split,status=status,rows=len(selected),
       score_quantiles=np.quantile([r['top_scores'][0] for r in selected],[0,.25,.5,.75,1]).tolist() if selected else [],
       margin_quantiles=np.quantile([r['top1_margin'] for r in selected],[0,.25,.5,.75,1]).tolist() if selected else [],threshold_fitted=False))
    if name.startswith('D-'):
     before={r['question_id']:r for r in lines(root/'C-raw-union/per_query.jsonl')}
     for row in rows:
      if not row['required_facts']:continue
      prior=before[row['question_id']];old=prior['first_gold_rank'];new=row['first_gold_rank'];reasons=[]
      if not prior['metrics']['20']['all_facts_hit']:reasons.append('STAGE1_MISS_OR_MISSING_FACT')
      if new!=1 and new is not None:reasons.append('RANKING_ERROR')
      if old is not None and new is not None and new>old:reasons.append('RERANKER_REGRESSION')
      if reasons:failures.append(dict(dataset_id=key,engine=engine,structure=name,question_id=row['question_id'],split=row['split'],query=row['query'],
        purpose=row['purpose'],reasons=reasons,expected_faq_ids=row['expected_faq_ids'],top_ids=row['top_ids'][:5],before_rank=old,after_rank=new,
        source_status='synthetic diagnostic failure; not an authenticated historical REAL_FAILURE'))
   load_results[key][engine]=read(root/'serving_load.json');repeats[key][engine]=read(root/'repeat_summary.json');attacks[key][engine]=read(root/'document_attack_summary.json')
   assert len(load_results[key][engine]['runs'])==5 and all(r['failures']==0 for r in load_results[key][engine]['runs'])
   assert repeats[key][engine]['runs']==100 and attacks[key][engine]['fixtures']==10
  consistency[key]={}
  for role in ('queries','corpus_question','corpus_question_answer'):
   x=np.load(base(key,'ollama')/'cache'/f'{role}.npy');y=np.load(base(key,'vllm')/'cache'/f'{role}.npy');cos=np.sum(x*y,axis=1)
   consistency[key][role]={'rows':len(x),'cosine_min':float(cos.min()),'cosine_mean':float(cos.mean()),'cosine_p01':float(np.quantile(cos,.01)),
     'max_absolute_component_delta':float(np.max(np.abs(x-y)))}
  for name in STRUCTURES:
   x=lines(base(key,'ollama')/name/'per_query.jsonl');y=lines(base(key,'vllm')/name/'per_query.jsonl')
   assert [r['question_id'] for r in x]==[r['question_id'] for r in y]
   consistency[key][name]={'top1_id_agreement':float(np.mean([a['top_ids'][0]==b['top_ids'][0] for a,b in zip(x,y)])),
     'mean_top20_overlap':float(np.mean([len(set(a['top_ids'])&set(b['top_ids']))/20 for a,b in zip(x,y)])),
     'hit1_disagreements':[a['question_id'] for a,b in zip(x,y) if a['metrics']['1']['hit']!=b['metrics']['1']['hit']]}
 save(OUT/'engine_consistency.json',consistency);save(OUT/'results.json',all_summary);save(OUT/'load_results.json',load_results)
 save_lines(OUT/'answerability_score_distributions.jsonl',distributions);save_lines(OUT/'retrieval_failures.jsonl',failures)
 save(OUT/'completion_audit.json',{'completed':True,'datasets':4,'engine_dataset_runs':8,'structures_per_run':len(STRUCTURES),'structures_total':len(audits),
  'candidate_log_rows':sum(x['rows'] for x in audits),'candidate_logs':audits,'query_rows_per_engine':sum(len(load(k)[1]) for k in KEYS),
  'load_requests':sum(r['requests'] for v in load_results.values() for e in v.values() for r in e['runs']),
  'repeated_requests':sum(e['runs'] for v in repeats.values() for e in v.values()),'attack_fixtures':80,
  'embedding_failed_requests_including_recovered':sum(read(base(k,e)/'embedding_execution.json')['request_failures'] for k in KEYS for e in ENGINES),
  'threshold_fitted':False,'independent_semantic_review':False,'generation_attack_test_executed':False,
  'dataset_manifest_sha256':sha(DATASET_ROOT/'manifest.json'),'timestamp':now()})
 report=['# v4 네 데이터셋 · Ollama / vLLM 실제 측정','',
  '4개 코퍼스와 질문셋을 각각 독립적으로 두 엔진에 입력했다. 각 실행은 기동 → HTTP 측정 → 종료 → 포트·컨테이너 종료 확인 순서로 진행했다. 입력은 v4 동결 파일만 읽었고, 과거 실행의 임베딩·점수는 재사용하지 않았다. v4가 계승한 v3 작성 문항은 v4 질문셋에 포함된다.','',
  '**성적의 범위:** 독립 의미 검수 전 합성 진단이다. 아래 검색 지표는 지원 사실이 있는 모든 하위 요구(일반·특수조건·문맥·부분 답변 포함)의 행 평균이다. NO_FAQ·모호한 질문·순수 조회는 검색 정확도 분모에 넣지 않는다. 일반 검색, 목적별, 분할별, 근거 위치별, 연결 그룹 평균은 `results.json`에 분리했다. 임계값은 피팅하지 않았다.','',
  '**모델:** Ollama BGE-M3 GGUF F16 / vLLM BGE-M3 HF float16. 정밀도를 맞췄지만 포맷·연산 구현은 다르다. BGE/Jina 리랭커는 공통 HF/PyTorch CUDA float16으로 실행했다. Sparse/ColBERT는 공통 FlagEmbedding native head로 전체 코퍼스를 계산했다. 이 보조 모델을 엔진 자체가 서빙한 결과로 해석하면 안 된다.','']
 labels={'existing':'기존 Excel','kt':'KT','skt':'SKT','lgu':'LG U+'}
 for key in KEYS:
  d,qs,docs=load(key)
  report += [f'## {labels[key]} — FAQ {len(docs):,}, 질문 {len(qs):,}','',
   '| 구조 | Ollama Hit@1 | vLLM Hit@1 | Ollama FactRecall@10 | vLLM FactRecall@10 | Ollama AllFactsHit@10 | vLLM AllFactsHit@10 |',
   '|---|---:|---:|---:|---:|---:|---:|']
  for name in STRUCTURES:
   if name=='C-raw-union':continue
   o=all_summary[key]['ollama'][name]['groups']['all_supported_subtasks']['row_mean'];v=all_summary[key]['vllm'][name]['groups']['all_supported_subtasks']['row_mean']
   report.append('| '+name+' | '+' | '.join(pct(x) for x in (o['hit@1'],v['hit@1'],o['fact_recall@10'],v['fact_recall@10'],o['all_facts_hit@10'],v['all_facts_hit@10']))+' |')
  report += ['', 'A=질문 Dense, B=질문+답변 Dense, C-fixed20=두 검색의 합집합을 max cosine으로 정렬한 뒤 20개 제한. D는 같은 원시 C 합집합 전체를 리랭크한 뒤 Top-K를 평가한다. E/F의 순수 sparse·multi 결과는 공통 보조 모델 측정이므로 두 엔진 열이 같다. E-dense-sparse-rrf만 엔진별 B 점수 순위와 공통 sparse를 결합한다.','',
   '| 엔진 | A Recall@20(사실) | B Recall@20(사실) | C 원시 합집합 Recall(각20→최대40) |','|---|---:|---:|---:|']
  for engine in ENGINES:
   values=[all_summary[key][engine][n]['groups']['all_supported_subtasks']['row_mean']['fact_recall@20'] for n in ('A','B','C-raw-union')]
   report.append('| '+engine+' | '+' | '.join(pct(x) for x in values)+' |')
  report += ['', '| 동시 요청 | Ollama req/s | vLLM req/s | Ollama p95(ms) | vLLM p95(ms) |','|---:|---:|---:|---:|---:|']
  for o,v in zip(load_results[key]['ollama']['runs'],load_results[key]['vllm']['runs']):
   report.append(f"| {o['concurrency']} | {o['requests_per_second']:.2f} | {v['requests_per_second']:.2f} | {1000*o['latency_seconds']['p95']:.2f} | {1000*v['latency_seconds']['p95']:.2f} |")
  report += ['',f"B Top-1 FAQ ID 일치율: {pct(consistency[key]['B']['top1_id_agreement'])}. 임베딩 query cosine 평균: {consistency[key]['queries']['cosine_mean']:.8f}.",'',
   f"반복 A 검색: 각 엔진 10개 질문 × 10회. 순위가 달라진 질문은 Ollama {repeats[key]['ollama']['queries_with_rank_change']}개, vLLM {repeats[key]['vllm']['queries_with_rank_change']}개.",'']
  language=[all_summary[key]['vllm'][n]['groups']['purpose:KR_EN_EQUIVALENT'] for n in ('B','D-bge-question_answer','D-jina-question_answer')]
  report += [f"한영 동일 의미 표현 묶음(vLLM, {language[0]['scored_rows']}문항)의 Hit@1: 질문+답변 Dense {pct(language[0]['row_mean']['hit@1'])}, BGE 질문+답변 리랭커 {pct(language[1]['row_mean']['hit@1'])}, Jina 질문+답변 리랭커 {pct(language[2]['row_mean']['hit@1'])}. 짝지은 한국어·영문/혼합 표현의 개별 순위 차이는 language_pair_results.jsonl에 저장했다.",'']
 report += ['## 해석과 남은 검증','',
  '- 속도는 웜업 이후 같은 64질문을 동시성 1/4/8/16/32에서 실행한 단일 진단이다. 클라이언트 HTTP 왕복·직렬화가 포함되며 긴 시간 포화 부하나 운영 SLA 측정이 아니다. 엔진·데이터셋 실행 순서는 고정되어 있다.',
  '- 전체 원문을 임베딩했다. 리랭커는 토큰 쌍 1,024, 문서 겹침 128, 구간별 raw logit 최대값을 사용한다. 긴 문서는 기회가 많아지는 효과가 있으므로 window_count와 길이별 결과를 함께 봐야 한다.',
  '- SKT Ollama의 3,548토큰 문서가 초기 물리 배치 크기 2,048 때문에 HTTP 400을 한 번 반환했다. 긴 입력은 4,096 배치를 사용하도록 수정하고 실패 지점부터 재개했다. 오류와 복구 기록은 incidents/skt_ollama_batch2048에 보존했다. 부하 시험 오류 수와 별도 집계했다.',
  '- BGE 리랭커의 첫 실행은 설치된 Transformers에서 제거된 토크나이저 API 때문에 추론 전에 중단됐다. XLM-R 입력 쌍과 padding을 공식 토크나이저 출력에 대조한 뒤 재실행했다. 초기 오류 로그는 incidents/reranker_tokenizer_api에 보존했다.',
  '- 원시 Dual Vector 합집합은 후보 예산이 더 크다. 각20→최대40의 Recall을 단일 Top20과 동등 예산 성능으로 주장하지 않는다.',
  '- 3,836개 전체 질문 입력(두 엔진 합계), 부하 2,560요청, 반복 800요청을 구분했다. 반복은 독립 질문 수를 늘리지 않는다.',
  '- 문서 공격은 엔진별·데이터셋별 10개씩 B 검색과 구성된 Context를 측정했다. 생성 LLM을 실행하지 않았으므로 공격 문구가 Context에 들어간 것을 지시 위반으로 판정하지 않는다.',
  '- NO_FAQ/부분 답변/개인 조회/명확화는 섞지 않았다. 점수·margin 분포만 기록했고 검수 전 라벨로 임계값을 최적화하지 않았다.',
  '- holdout도 사전에 고정한 구조 행렬의 진단에 포함했다. 이번 결과를 보고 구조·임계값·질문을 조정한다면 이후 최종 검증에는 새 holdout이 필요하다.',
  '- 실제 운영 실패 이력과 정책 버전 검증은 이번 합성 검색 실패로 대신하지 않았다. 한영쌍 결과와 리랭커 악화 사례는 각 구조별 로그로 확인한다.','',
  '## 결과 파일','',
  '- `results.json`: 데이터셋·엔진·구조·목적·분할·근거 위치별 검색 지표',
  '- `<dataset>/<engine>/<structure>/candidate_log.jsonl.gz`: 요청한 전체 필드와 복수 정답·사실별 근거',
  '- `<dataset>/<engine>/<structure>/language_pair_results.jsonl`: 한영쌍 순위·Top20 겹침·사실 검색 지표',
  '- `retrieval_failures.jsonl`: Stage1 miss, ranking error, reranker regression 진단',
  '- `answerability_score_distributions.jsonl`: NO_FAQ 등 상태별 score/margin 분포',
  '- `engine_consistency.json`: 임베딩 및 순위의 엔진 간 차이',
  '- `completion_audit.json`: 요청 로그 전수 감사',
  '- `<dataset>/<engine>/stop_verification.json`, `final_shutdown.json`: 종료 확인','']
 (OUT/'comparison.md').write_text('\n'.join(report),encoding='utf-8')
 print(json.dumps({'complete':True,'candidate_log_rows':sum(x['rows'] for x in audits),'structures':len(audits)}),flush=True)

if __name__=='__main__':main()
