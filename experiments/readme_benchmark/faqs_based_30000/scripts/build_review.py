"""Portable, offline review UI containing all cases and the supplied evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / 'data', ROOT / 'outputs'


def read(name):
    return [json.loads(s) for s in (DATA / name).read_text(encoding='utf-8').splitlines()]


def safe_json(value):
    # Untrusted source content must not close a script or become executable HTML.
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


def main():
    report = json.loads((OUT / 'validation.json').read_text(encoding='utf-8'))
    assert report['passed'], 'Structural validation must pass before delivery'
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    rows = read('variations.jsonl')
    faqs = {f['faq_id']: f for f in read('corpus.jsonl')}
    competitors = {r['parent_case_id']: r for r in read('hard_negative_candidates.jsonl')}
    slim = [{k: r[k] for k in ['case_id', 'parent_case_id', 'brand', 'category', 'source_language', 'query', 'primary_variation', 'validation_tags', 'language_pairs', 'source_faq_ids', 'split_group_id', 'added_user_scope']} | {'changes': [{k: v for k, v in t.items() if k != 'background'} for t in r['change_trace']]} for r in rows]
    catalog = json.loads((DATA / 'purpose_catalog.json').read_text(encoding='utf-8'))
    payload = {'manifest': manifest, 'rows': slim, 'faqs': faqs, 'competitors': competitors, 'catalog': catalog, 'validation': report}
    template = '''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>새 FAQ 기반 변형 질문 30,000행 검토</title>
<style>
:root{font-family:"Malgun Gothic",system-ui,sans-serif;color:#203047;background:#edf1f6;font-size:14px}*{box-sizing:border-box}body{margin:0}header{background:#17375a;color:#fff;padding:25px 32px}h1{font-size:25px;margin:0 0 10px}h2{font-size:18px}p{line-height:1.7;margin:10px 0}main{padding:24px 32px;max-width:1800px;margin:auto}.muted{color:#53647a}.note{background:#fff5dc;border:1px solid #e0c68b;padding:12px 16px;border-radius:8px;margin:16px 0}.stats{display:flex;flex-wrap:wrap;gap:10px}.stat{background:white;border:1px solid #dce3ed;border-radius:8px;min-width:150px;padding:15px}.stat b{display:block;font-size:23px;margin-top:5px}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;background:white;border-radius:8px;padding:14px;margin:18px 0}input,select,button{font:inherit;padding:9px;border:1px solid #b9c6d7;border-radius:5px;background:#fff}input{min-width:300px;flex:1}button{cursor:pointer}button:hover{background:#eaf1fa}button:disabled{opacity:.45;cursor:default}.pager{display:flex;gap:12px;align-items:center;margin:12px 0}.tableWrap{overflow:auto;background:#fff;border-radius:8px}table{border-collapse:collapse;width:100%;min-width:1000px}th{background:#dde8f5;text-align:left;padding:13px;vertical-align:top}td{padding:14px;border-bottom:1px solid #e4e9f1;vertical-align:top;line-height:1.65}td.query{width:42%;min-width:390px;white-space:pre-wrap;overflow-wrap:anywhere}.scroll{max-height:360px;overflow:auto}.badge{display:inline-block;border-radius:4px;padding:2px 6px;font-size:11px;background:#edf2f8;margin:2px}details{margin-top:8px}summary{cursor:pointer;color:#245c95;font-weight:600}pre{font:inherit;line-height:1.6;white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fb;padding:12px;border-radius:5px}.evidence{max-width:650px;min-width:300px}.id{font-family:Consolas,monospace;font-size:12px}a{color:#245c95}.catalog{font-size:12px}footer{margin:20px 0;color:#53647a;font-size:12px}.empty{text-align:center;padding:40px}
</style></head><body>
<header><h1>새 FAQ 기반 변형 질문 30,000행</h1><p>사용자가 제공한 faqs.jsonl에서 3,000쌍을 선택하고 각각 10개를 생성했습니다. 원천 근거와 변형 요소를 함께 검토할 수 있습니다.</p></header>
<main><div class="stats" id="stats"></div>
<div class="note">규칙 기반 자동 생성 초안입니다. 구조 검증은 통과했지만 독립적인 의미·정답 검수는 미완료입니다. 정답 없는 질문이 없어 NO_FAQ 임계값을 확정할 수 없습니다. 기존 Calibration은 일시 중지 상태입니다.</div>
<details><summary>검증 목적별 범위와 한계</summary><div class="tableWrap"><table class="catalog"><thead><tr><th>목적</th><th>검증할 실패</th><th>이번 생성 범위 / 한계</th><th>주변형 행</th><th>태그 행</th></tr></thead><tbody id="catalog"></tbody></table></div></details>
<p class="muted">원문의 상품명·숫자·조건을 유지합니다. V10은 실제 FAQ 두 개를 결합하므로 두 근거를 모두 확인해야 합니다. 같은 FAQ의 원본 중복 ID는 허용 가능한 대체 ID입니다. 경쟁 FAQ 목록은 검수 전 후보입니다.</p>
<div class="controls"><label>브랜드 <select id="brand"><option value="">전체</option></select></label><label>검증 태그 <select id="tag"><option value="">전체</option></select></label><label>원천 언어 <select id="lang"><option value="">전체</option><option value="ko">한국어</option><option value="en">영어</option></select></label><input id="search" placeholder="원본 ID, 질문, 카테고리 검색" aria-label="검색"><button id="clear">초기화</button></div>
<div class="pager"><button id="prev">이전</button><span id="count"></span><button id="next">다음</button></div>
<div class="tableWrap"><table><thead><tr><th>문항 / 분류</th><th>변형 질문</th><th>목적 / 변경</th><th>원문 질문·답변 / 경쟁 후보</th></tr></thead><tbody id="cases"></tbody></table></div>
<footer id="footer"></footer></main>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
const d=JSON.parse(document.getElementById('data').textContent);
const byId=id=>document.getElementById(id);
const node=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=String(text);if(cls)e.className=cls;return e;};
const number=v=>v.toLocaleString('ko-KR');
for(const [label,value] of [['원천 FAQ',d.manifest.source_records],['선택 쌍',3000],['변형 질문',30000],['분류 조합',d.manifest.represented_strata],['한영 변형',d.manifest.validation_tag_counts.KR_EN_EQUIVALENT],['검수 전 원천',d.manifest.source_review_records]]){const c=node('div',undefined,'stat');c.append(node('span',label),node('b',number(value)));byId('stats').append(c);}
for(const brand of [...new Set(d.rows.map(r=>r.brand))].sort()){const o=node('option',brand);o.value=brand;byId('brand').append(o);}
for(const tag of [...new Set(d.rows.flatMap(r=>r.validation_tags))].sort()){const o=node('option',tag);o.value=tag;byId('tag').append(o);}
for(const p of d.catalog){const tr=node('tr');for(const value of [p.tag,p.validation_purpose,p.scope_and_limit,number(p.primary_rows),number(p.tagged_rows)])tr.append(node('td',value));byId('catalog').append(tr);}
for(const r of d.rows)r.searchText=[r.case_id,r.parent_case_id,r.query,r.category,...r.source_faq_ids].join(' ').toLocaleLowerCase();
let filtered=d.rows,page=0;const pageSize=25;
function faqCard(f){const box=node('details');box.append(node('summary',f.question));box.append(node('div',f.brand+' / '+f.category,'muted'),node('div',f.faq_id,'id'),node('pre',f.answer));const aliases=node('details');aliases.append(node('summary','원천 ID '+f.source_faq_ids.length+'개'),node('pre',f.source_faq_ids.join('\n')));box.append(aliases);for(const url of f.source_urls){try{const u=new URL(url);if(!['http:','https:'].includes(u.protocol))continue;const a=node('a','원천 링크');a.href=url;a.target='_blank';a.rel='noopener noreferrer';box.append(a,node('br'));}catch{}}return box;}
function render(){const tbody=byId('cases');tbody.replaceChildren();const start=page*pageSize,end=Math.min(filtered.length,start+pageSize);byId('count').textContent=number(filtered.length)+'행 / '+(filtered.length?number(start+1)+'–'+number(end):'검색 결과 없음');byId('prev').disabled=page===0;byId('next').disabled=end>=filtered.length;
if(!filtered.length){const tr=node('tr'),td=node('td','검색 결과가 없습니다.','empty');td.colSpan=4;tr.append(td);tbody.append(tr);return;}
for(const r of filtered.slice(start,end)){const tr=node('tr');const info=node('td');info.append(node('strong',r.case_id),node('div',r.brand),node('div',r.category,'muted'),node('div',r.source_language+' / '+r.split_group_id,'id'));if(r.added_user_scope)info.append(node('div','원문 대상이 생략되어 사용자 범주를 명시','muted'));
const q=node('td',undefined,'query');q.append(node('div',r.query,'scroll'));
const purpose=node('td');purpose.append(node('strong',r.primary_variation));const tags=node('div');for(const tag of r.validation_tags)tags.append(node('span',tag,'badge'));purpose.append(tags);for(const pair of r.language_pairs)purpose.append(node('div',pair.ko+' ↔ '+pair.en));const change=node('details');change.append(node('summary','변경 기록'),node('pre',JSON.stringify(r.changes,null,2)));purpose.append(change);
const evidence=node('td',undefined,'evidence');for(const id of r.source_faq_ids)evidence.append(faqCard(d.faqs[id]));const competitors=d.competitors[r.parent_case_id];const neg=node('details');neg.append(node('summary','경쟁 후보 · 오답 검수 전'));for(const c of [...competitors.same_brand_candidates,...competitors.cross_brand_candidates]){const line=node('div',c.brand+' / '+c.question+' / 유사도 '+c.char_tfidf_similarity.toFixed(3));neg.append(line);}evidence.append(neg);tr.append(info,q,purpose,evidence);tbody.append(tr);}}
function apply(){const query=byId('search').value.trim().toLocaleLowerCase(),brand=byId('brand').value,tag=byId('tag').value,lang=byId('lang').value;filtered=d.rows.filter(r=>(!brand||r.brand===brand)&&(!tag||r.validation_tags.includes(tag))&&(!lang||r.source_language===lang)&&(!query||r.searchText.includes(query)));page=0;render();}
for(const id of ['brand','tag','lang'])byId(id).addEventListener('change',apply);let timer;byId('search').addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(apply,180);});byId('clear').addEventListener('click',()=>{for(const id of ['brand','tag','lang','search'])byId(id).value='';apply();});byId('prev').addEventListener('click',()=>{page--;render();});byId('next').addEventListener('click',()=>{page++;render();});
byId('footer').textContent='원천 SHA256: '+d.manifest.source_sha256+' · 이 파일은 오프라인 검토 화면입니다. 수정 결과를 자동 저장하지 않습니다. 근거는 제공된 스냅샷입니다.';render();
</script></body></html>'''
    # Escape JS source newlines inside join(), before substituting data. Never
    # use source text as HTML markup, event handlers, or executable JavaScript.
    template = template.replace("join('\n')", "join('\\n')")
    output = OUT / '질문_30000행_검토.html'
    output.write_text(template.replace('__PAYLOAD__', safe_json(payload)), encoding='utf-8')
    inventory = []
    for p in [ROOT / 'README.md', ROOT / 'compose.yaml']:
        inventory.append({'path': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    for folder in ['source', 'data', 'scripts', 'outputs']:
        for p in sorted((ROOT / folder).glob('*')):
            if p.is_file() and p.name != 'artifact_inventory.json':
                inventory.append({'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    (OUT / 'artifact_inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'review_html': str(output), 'rows': len(rows), 'bytes': output.stat().st_size, 'inventory_files': len(inventory)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
