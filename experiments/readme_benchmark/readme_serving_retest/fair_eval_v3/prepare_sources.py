"""Read-only source audit and score-blind stratified authoring queue.
The provided FAQ question is never exported as an evaluation query.
"""
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path(r"C:\Users\eongp\Downloads\faqs(1).jsonl")
SOURCE_HASH = "76d867e4b4e183e6b52c8edfb31ec7da4e74f8495b4a2a2ce10fb2472ecf8a2d"
SEED = "fair-v3-source-coverage-20261007"
BRANDS = {"SKT", "KT", "LG U+"}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r,ensure_ascii=False,sort_keys=True) for r in rows)+"\n",encoding="utf-8",newline="\n")


def normalize(s):
    return re.sub(r"\s+"," ",s).strip()


def prepare():
    if digest(SOURCE) != SOURCE_HASH:
        raise ValueError("Source changed; review before using a different snapshot")
    (ROOT/"source").mkdir(parents=True,exist_ok=True)
    shutil.copyfile(SOURCE,ROOT/"source/faqs.jsonl")
    raw=[json.loads(x) for x in SOURCE.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    groups=defaultdict(list); quarantine=[]; outside=[]
    for rowno,r in enumerate(raw,1):
        if r['brand'] not in BRANDS:
            outside.append({"source_row":rowno,"source_faq_id":r['faq_id'],"brand":r['brand'],"reason":"user requested only SKT/KT/LG U+"})
            continue
        if r.get("content_status")!="ANSWER_CAPTURED" or not r.get("answer","").strip() or not r.get("question","").strip():
            quarantine.append({"source_row":rowno,"source_faq_id":r["faq_id"],"reason":"no independently captured text answer","record":r})
        else:
            groups[(r['brand'],normalize(r['question']),normalize(r['answer']))].append((rowno,r))
    docs=[]
    for key,rs in groups.items():
        rowno,r=rs[0]
        fid="FAQ-"+{"SKT":"SKT","KT":"KT","LG U+":"LGU"}[r['brand']]+"-"+hashlib.sha256("\x1f".join(key).encode()).hexdigest()[:12]
        docs.append({"faq_id":fid,"brand":r['brand'],"category":r['category'],"question":r['question'],"answer":r['answer'],
            "language":r['language'],"source_scope":"provided_jsonl:"+r['brand'],"source_rows":[n for n,_ in rs],
            "source_faq_ids":[x['faq_id'] for _,x in rs],"source_urls":sorted({x['source_url'] for _,x in rs}),
            "retrieved_on":sorted({x['retrieved_on'] for _,x in rs}),"asset_urls":r['answer_asset_urls'],
            "alternate_categories":sorted({x['category'] for _,x in rs}),"support_review":"source_snapshot_only"})
    byq=defaultdict(list)
    for d in docs:byq[(d['brand'],normalize(d['question']))].append(d)
    repeated={k:v for k,v in byq.items() if len(v)>1}
    hold_ids={d['faq_id'] for v in repeated.values() for d in v}
    seed_candidates=[]
    for d in docs:
        d['authoring_hold_reason']=("same_brand_question_has_different_answers_or_scope" if d['faq_id'] in hold_ids else
            "answer_has_external_asset_dependency_to_review" if d['asset_urls'] else
            "short_or_fragment_answer_to_review" if len(normalize(d['answer']))<12 else None)
        if d['authoring_hold_reason'] is None:seed_candidates.append(d)
    strata=defaultdict(list)
    for d in seed_candidates:strata[(d['brand'],d['category'],d['language'])].append(d)
    # Every eligible brand/category/language stratum is represented. Square-root
    # allocation tempers large categories without forcing equal FAQ variants.
    target=900
    quotas={s:1 for s in strata}
    while sum(quotas.values())<min(target,len(seed_candidates)):
        s=max((s for s in strata if quotas[s]<len(strata[s])),key=lambda s:(math.sqrt(len(strata[s]))/(quotas[s]+1),s))
        quotas[s]+=1
    selected=[]
    for s in sorted(strata):
        ranked=sorted(strata[s],key=lambda d:hashlib.sha256((SEED+d['faq_id']).encode()).hexdigest())
        selected+=ranked[:quotas[s]]
    for i,d in enumerate(selected,1):d['authoring_id']=f"S{i:04}"
    # This is an authoring catalog, never an embedding input. The builder
    # exports a separate corpus for each explicitly selected dataset.
    save(ROOT/'authoring/corpus_catalog.jsonl',docs)
    save(ROOT/'data/quarantine.jsonl',quarantine)
    save(ROOT/'source/excluded_other_brands.jsonl',outside)
    save(ROOT/'authoring/queue.jsonl',selected)
    save(ROOT/'data/question_answer_scope_review.jsonl',[{"brand":k[0],"question":k[1],"faq_ids":[d['faq_id'] for d in v],
        "status":"needs_scope_or_compatibility_review; not automatically a contradiction"} for k,v in repeated.items()])
    audit={"source_path":str(SOURCE),"source_sha256":SOURCE_HASH,"identical_to_prior_provided_faqs_jsonl":True,
        "raw_records":len(raw),"canonical_text_faqs":len(docs),"collapsed_same_brand_exact_qa_duplicates":sum(len(rs)-1 for rs in groups.values()),
        "quarantined_no_text_answer":len(quarantine),"same_question_different_answer_groups":len(repeated),
        "new_carrier_corpus_total":len(docs),"new_scope_counts":dict(Counter(d['brand'] for d in docs)),
        "other_brand_records_excluded":len(outside),"existing_corpus_merged_into_carrier_corpora":False,
        "authoring_seed_count":len(selected),"eligible_strata_count":len(strata),"selected_strata_count":len(quotas),
        "source_question_is_not_eval_query":True,"old_generated_questions_reused_for_new_carriers":False,"old_model_results_loaded":False,
        "authoring_selection":"minimum one per eligible brand/category/language + bounded square-root allocation + deterministic hash",
        "dataset_ids":["existing","kt","skt","lgu"],
        "mixed_scope_rule":"Embedding, index, cache, retrieval and results are isolated per predeclared dataset; no pooled index or gold-based routing",
        "no_faq_rule":"Existing v2 NO_FAQ labels apply only to existing; never copied into kt, skt or lgu"}
    (ROOT/'data/source_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':prepare()
