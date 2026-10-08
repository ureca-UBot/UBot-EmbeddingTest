"""Offline review pages and lexical candidate queue; no benchmark models used."""
import html
import json
import math
import re
from collections import Counter,defaultdict
import build_dataset as b

def terms(text):
    words=re.findall(r'[가-힣A-Za-z0-9]+',text.casefold())
    result=set(words)
    for word in words:
        if re.search('[가-힣]',word):result.update(word[i:i+3] for i in range(len(word)-2))
    return result

def candidates(docs,questions):
    postings=defaultdict(set);qterms={}
    for d in docs:
        qterms[d['faq_id']]=terms(d['question'])
        for t in terms(d['question']+' '+d['answer']):postings[t].add(d['faq_id'])
    n=len(docs);result=[]
    for q in questions:
        weights=Counter()
        for t in terms(q['query']):
            for fid in postings.get(t,[]):
                weights[fid]+=math.log(1+n/(1+len(postings[t])))*(2 if t in qterms[fid] else 1)
        known=set(q['answer_faq_ids'])
        best=[{'faq_id':fid,'lexical_review_score':round(value,6)}
              for fid,value in sorted(weights.items(),key=lambda x:(-x[1],x[0])) if fid not in known][:5]
        result.append({'question_id':q['question_id'],'dataset_id':q['dataset_id'],
            'corpus_scanned':n,'candidates':best,'decision':'unreviewed_not_gold',
            'method':'word_and_korean_trigram_overlap_for_review_only'})
    return result

TEMPLATE='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ 질문·정답 검수</title><style>
*{box-sizing:border-box}body{font:15px/1.65 system-ui,"Malgun Gothic",sans-serif;margin:0;color:#182238;background:#f4f6fa}header{padding:20px 28px;background:#152947;color:white}header h1{margin:0;font-size:24px}a{color:#1768c3}header a{color:#a9d6ff}main{padding:20px;max-width:1680px;margin:auto}.notice{padding:12px 16px;background:#fff4da;border:1px solid #dfbd6d;border-radius:8px}nav{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}input,select,button{font:inherit;border:1px solid #aab8c9;border-radius:6px;padding:7px 10px;background:white}input{flex:1;min-width:260px}button{cursor:pointer}section{display:grid;grid-template-columns:390px 1fr;gap:18px}aside,article{background:white;border:1px solid #d6dfe9;border-radius:8px;padding:16px}aside{max-height:78vh;overflow:auto}article{overflow-wrap:anywhere}aside button{display:block;text-align:left;width:100%;margin:6px 0;border-color:#e1e7ed}aside button.active{border-color:#1768c3;background:#eaf2ff}.tag{color:#526581;font-size:13px}.question{font-size:20px;font-weight:650}pre{white-space:pre-wrap;word-break:break-word;font:14px/1.7 inherit;background:#f6f8fc;padding:12px}details{border-top:1px solid #dce3ed;padding:10px 0}summary{cursor:pointer;font-weight:600}.missing{color:#a33220}.stats{margin:10px 0;color:#526581}h2{font-size:18px}h3{font-size:16px}code{font-size:13px}@media(max-width:900px){section{display:block}aside{max-height:260px;margin-bottom:15px}}
</style><header><a href="index.html">← 4개 데이터셋</a><h1>__TITLE__ · 질문과 FAQ 근거</h1><div>__COUNTS__</div></header><main>
<div class="notice">독립 의미 검수 전 합성 작성본입니다. 정답 후보 추천은 라벨이 아닙니다. 원문의 정책은 수집 시점 자료이며 현행 정책 보증이 아닙니다. 이 화면의 라벨·근거·후보는 모델 입력에 전달하지 않습니다.</div>
<nav><select id="mode"><option value="queries">사용자 질문</option><option value="corpus">FAQ 전체</option></select><select id="split"><option value="">모든 분할</option><option>development</option><option>calibration</option><option>holdout</option></select><select id="purpose"><option value="">모든 목적</option></select><input id="search" placeholder="질문·FAQ ID·분야 검색"><button id="clear">초기화</button></nav>
<div class="stats" id="stats"></div><section><aside id="list"></aside><article id="detail">목록에서 질문 또는 FAQ를 선택하세요.</article></section></main>
<script>const DATA=__DATA__;
const docs=new Map(DATA.docs.map(d=>[d.faq_id,d]));const facts=new Map(DATA.facts.map(f=>[f.fact_id,f]));const candidates=new Map(DATA.candidates.map(c=>[c.question_id,c]));
const $=id=>document.getElementById(id);let selected='';
function el(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e}
function faq(fid,open=false){const d=docs.get(fid);if(!d)return el('p','ID를 찾을 수 없음: '+fid,'missing');const block=el('details');block.open=open;block.append(el('summary',fid+' · '+d.question),el('div',d.category,'tag'),el('pre',d.answer));return block}
function show(q){selected=q.question_id||q.faq_id;const panel=$('detail');panel.replaceChildren();if(q.faq_id){panel.append(faq(q.faq_id,true));return}
 panel.append(el('div',q.question_id+' · '+q.split+' · '+q.corpus_status,'tag'),el('p',q.user_question,'question'),el('p',(q.purpose_tags||[]).join(' / ')+' · '+q.language,'tag'));
 if(q.dialogue_history?.length){panel.append(el('h2','앞선 발화'),el('pre',q.dialogue_history.map(x=>x.content).join('\\n')),el('h3','고정 검색 입력'),el('pre',q.model_query))}
 panel.append(el('h2','정답 FAQ ID 전체 목록'),el('pre',JSON.stringify(q.answer_faq_ids,null,2)));
 panel.append(el('p','각 사실 안의 FAQ는 대체 가능(OR)합니다. 복수 사실을 모두 답하려면 사실별로 하나 이상의 FAQ가 필요(AND)합니다.'));
 for(const f of q.required_facts){const info=facts.get(f.fact_id);panel.append(el('h3',f.fact_id),el('p',info?.requested_information||info?.statement||'요구 사실'));
  if(!f.acceptable_faq_ids.length)panel.append(el('p','현재 코퍼스에 근거 없음으로 제안됨. 부재 의미 검수 필요.','missing'));
  for(const fid of f.acceptable_faq_ids){panel.append(faq(fid));const s=info?.support?.find(x=>x.faq_id===fid);if(s)panel.append(el('pre',(s.quotes||[s.quote]).join('\\n…\\n')))}
 }
 if(q.unresolved_information)panel.append(el('h2','미해결 정보 / 확인이 필요한 문맥'),el('p',q.unresolved_information,'missing'));
 if(q.missing_information)panel.append(el('h2','부족한 정보'),el('p',q.missing_information,'missing'));
 if(q.unsupported_facts?.length)panel.append(el('h2','검색 채점에서 제외한 미지원 사실'),el('pre',JSON.stringify(q.unsupported_facts,null,2)));
 if(q.tool_requirements?.length)panel.append(el('h2','개인 정보·실시간 조회 필요'),el('pre',JSON.stringify(q.tool_requirements,null,2)));
 if(q.diagnostic_panels?.length)panel.append(el('h2','검색 구조 비교용 근거 위치'),el('pre',JSON.stringify(q.diagnostic_panels,null,2)));
 for(const near of q.near_miss_candidates||[]){panel.append(el('h3','혼동 후보 · 정답 제외'),el('p',near.reason),faq(near.faq_id))}
 if(q.interpretation_candidate_ids?.length){panel.append(el('h2','가능한 해석의 관련 문서 · 정답 목록 아님'));for(const fid of q.interpretation_candidate_ids)panel.append(faq(fid))}
 const suggestions=candidates.get(q.question_id)?.candidates||[];
 if(suggestions.length){panel.append(el('h2','대체 정답 누락 검토 후보 · 미검수'),el('p','문자 유사도로 제안했으며 정답으로 자동 추가하지 않았습니다. 제목의 대상과 답변의 조건·예외를 함께 확인하세요.'));for(const c of suggestions)panel.append(faq(c.faq_id))}
 panel.append(el('details'));const raw=panel.lastChild;raw.append(el('summary','라벨 JSON'),el('pre',JSON.stringify(q,null,2)));
}
function render(){const term=$('search').value.toLocaleLowerCase(),mode=$('mode').value,split=$('split').value,purpose=$('purpose').value;const source=mode==='corpus'?DATA.docs:DATA.questions;const rows=source.filter(q=>(mode==='corpus'||(!split||q.split===split)&&(!purpose||q.primary_purpose===purpose))&&(!term||JSON.stringify(q).toLocaleLowerCase().includes(term)));$('stats').textContent=rows.length+'개 표시 / '+source.length+'개';$('list').replaceChildren();for(const q of rows){const id=q.question_id||q.faq_id;const btn=el('button',id+'\\n'+(q.user_question||q.question),id===selected?'active':'');btn.onclick=()=>{show(q);render()};$('list').append(btn)}}
for(const p of [...new Set(DATA.questions.map(q=>q.primary_purpose))].sort())$('purpose').append(el('option',p));
for(const id of ['mode','split','purpose','search'])$(id).addEventListener('input',render);$('clear').onclick=()=>{$('search').value='';$('split').value='';$('purpose').value='';render()};render();show(DATA.questions[0]);
</script></html>'''

def build():
    out=b.ROOT/'review';out.mkdir(exist_ok=True);all_candidates=[];stats={}
    labels={'existing':'기존 Excel 기반 · v4','kt':'KT · v4','skt':'SKT · v4','lgu':'LG U+ · v4'}
    links=[]
    for key,label in labels.items():
        directory=b.ROOT/'datasets'/key
        docs=b.read(directory/'faq_pairs.jsonl');qs=b.read(directory/'user_questions.jsonl');fs=b.read(directory/'facts.jsonl')
        queue=candidates(docs,qs)
        all_candidates+=queue
        data=json.dumps({'docs':docs,'questions':qs,'facts':fs,'candidates':queue},ensure_ascii=False).replace('<','\\u003c')
        page=TEMPLATE.replace('__TITLE__',html.escape(label)).replace('__COUNTS__',f'FAQ {len(docs):,}개 · 사용자 질문 {len(qs):,}개').replace('__DATA__',data)
        (out/f'{key}.html').write_text(page,encoding='utf-8')
        stats[key]={'faq_count':len(docs),'question_count':len(qs),'source_categories':dict(Counter(d['category'] for d in docs)),
            'query_categories':dict(Counter(q['category'] for q in qs)),
            'splits':{s:{'count':sum(q['split']==s for q in qs),'status':dict(Counter(q['corpus_status'] for q in qs if q['split']==s)),
                'purpose':dict(Counter(q['primary_purpose'] for q in qs if q['split']==s))} for s in ('development','calibration','holdout')},
            'language_counts':dict(Counter(q['language'] for q in qs)),
            'same_normalized_question_duplicates':len(qs)-len({b.norm(q['query']) for q in qs}),
            'retrieval_input_unique_count':len({b.norm(q['model_query']) for q in qs}),
            'multi_answer_questions':sum(len(q['answer_faq_ids'])>1 for q in qs),
            'review_candidate_count':sum(len(c['candidates']) for c in queue)}
        links.append(f'<li><a href="{key}.html">{html.escape(label)}</a> — FAQ {len(docs):,} / 질문 {len(qs):,}</li>')
    total=sum(v['question_count'] for v in stats.values())
    (out/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>독립 코퍼스 4개 · v4 검수</title><style>body{font:17px/1.8 system-ui;margin:50px;max-width:1000px;color:#182238}a{color:#1768c3}li{margin:15px 0}</style><h1>통신사별 FAQ 평가셋 v4</h1><p>각 페이지의 질문·정답·검색 대상은 해당 데이터셋에 속합니다. 임베딩 전에 데이터셋을 선택합니다.</p><ul>'+''.join(links)+f'</ul><p>총 {total:,}개 사용자 질문. 네 데이터셋이 각각 15개 목적의 작성 최소 기준을 만족하도록 구성했습니다.</p><p>독립 의미 검수 전 합성 작성본입니다. 실제 실패·정책 버전은 증거 확보 상태를 별도 기록했고, 반복·문서 공격은 실행 전 fixture입니다. 새 모델 측정 결과가 아닙니다. 코퍼스별 주제 구성 차이, 작은 부분집합, 정답 누락 가능성이 남아 있습니다.</p><p><a href="../README.md">구성·분할·목적별 작성 범위 보기</a></p></html>',encoding='utf-8')
    b.write(b.ROOT/'data/alternative_review_candidates.jsonl',all_candidates)
    (b.ROOT/'data/quality_report.json').write_text(json.dumps({'datasets':stats,'independent_review':'pending',
        'candidate_method':'lexical only; never automatically gold','model_outputs_used_to_author_questions':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Created 4 offline review pages, quality report and lexical candidate queue.')

if __name__=='__main__':build()
