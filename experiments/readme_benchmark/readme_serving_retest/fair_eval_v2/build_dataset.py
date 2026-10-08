"""Build an auditable synthetic dataset without loading any model result.

Usage: python -X utf8 build_dataset.py
Only the local authoring files, frozen FAQ corpus and old *queries* are read.
The latter are used solely for a contamination audit after authoring.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from fact_spec import ALIASES, ATOMIC, COMPARISON_EXPORT_SLOTS, CONTRAST_EXCLUSIONS

ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT.parent / "data/corpus.jsonl"
SPLITS = ("development", "calibration", "holdout")
BASELINE_HASHES = {
    "corpus.jsonl": "c361681bb3911fa25d2ba9d4c339637bd4e13a77bc827874242697731444913d",
    "cases.jsonl": "64ed82884092d5eaa57159e90f756f5843ef7bcd00572abb6fa167fd05a85b95",
    "evaluation_cases.jsonl": "c54ee3358199e1fbef268c22488fdb14f3cfcfa0026b2d74dab826c47a261a0a",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj, lines=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = ("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in obj)
               if lines else json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
    path.write_text(content + "\n", encoding="utf-8", newline="\n")


def read_rows(name, width):
    rows = []
    for number, line in enumerate((ROOT / "authoring" / name).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        fields = line.split("|")
        if len(fields) != width:
            raise ValueError(f"{name}:{number}: expected {width} columns, got {len(fields)}")
        rows.append(fields)
    return rows


def faq_id(number):
    return f"FAQ-{int(number):03d}"


def normalized(text):
    return "".join(re.findall(r"[\w]", text.casefold()))


def grams(text):
    text = normalized(text)
    return {text[i:i + 2] for i in range(len(text) - 1)}


def connected_split(cases, ratios, seed):
    """Join before splitting. Labels balance allocation; no model scores enter."""
    parent = list(range(len(cases)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, case in enumerate(cases):
        keys = ["family:" + x for x in case["policy_families"]]
        keys += ["fact:" + x["fact_id"] for x in case["required_facts"]]
        keys += ["faq:" + x for x in case["expected_faq_ids"]]
        keys += ["pair:" + x for x in case["pair_ids"]]
        keys += ["missing:" + x["missing_fact_id"] for x in case["unsupported_facts"]]
        for key in keys:
            if key in owners:
                parent[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = defaultdict(list)
    for i, case in enumerate(cases):
        groups[find(i)].append(case)

    def features(group):
        out = Counter(total=len(group))
        out["distinct_negative_claims"] = len({u["missing_fact_id"] for c in group
             if c["corpus_status"] == "NO_FAQ" for u in c["unsupported_facts"]})
        for c in group:
            out["category:" + c["category"]] += 1
            out["purpose:" + c["primary_purpose"]] += 1
            out["status:" + c["corpus_status"]] += 1
            if c["corpus_status"] == "FAQ_EXISTS":
                out["positive_category:" + c["category"]] += 1
            if c["corpus_status"] == "NO_FAQ":
                out["no_faq_category:" + c["category"]] += 1
        return out

    groups = list(groups.values())
    signature = lambda g: hashlib.sha256((seed + "|" + min(c["case_id"] for c in g)).encode()).hexdigest()
    groups.sort(key=lambda g: (-len(g), signature(g)))
    fs = [features(g) for g in groups]
    totals = sum(fs, Counter())
    group_presence = Counter(key for feat in fs for key in feat if feat[key])
    coverage_keys = {key for key, n in group_presence.items() if n >= len(SPLITS)
                     and key.startswith(("purpose:", "positive_category:"))}
    allocations = {s: Counter() for s in SPLITS}

    def cost(counts):
        result = 0.0
        for split in SPLITS:
            for key, total in totals.items():
                target = ratios[split] * total
                weight = 5 if key in {"total", "distinct_negative_claims"} else (3 if key.startswith("status:") else 1)
                result += weight * (counts[split][key] - target) ** 2 / max(target, 1)
                if key in coverage_keys and counts[split][key] == 0:
                    result += 200
        return result

    assignments = []
    for feat in fs:
        def incremental(s):
            allocations[s].update(feat)
            v = cost(allocations)
            allocations[s].subtract(feat)
            return v, SPLITS.index(s)
        chosen = min(SPLITS, key=incremental)
        assignments.append(chosen)
        allocations[chosen].update(feat)

    # Deterministic label-only local improvements, never a search over outcomes.
    for _ in range(10):
        changed = False
        for i, feat in enumerate(fs):
            current = assignments[i]
            best, best_cost = current, cost(allocations)
            allocations[current].subtract(feat)
            for s in SPLITS:
                allocations[s].update(feat)
                score = cost(allocations)
                allocations[s].subtract(feat)
                if score < best_cost - 1e-9:
                    best, best_cost = s, score
            allocations[best].update(feat)
            assignments[i] = best
            changed |= current != best
        if not changed:
            break

    for g, assigned in zip(groups, assignments):
        component = "G-" + hashlib.sha256("|".join(sorted(c["case_id"] for c in g)).encode()).hexdigest()[:12]
        for case in g:
            case["split"] = assigned
            case["leakage_group_id"] = component
    return len(groups)


def build():
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    for name, expected in BASELINE_HASHES.items():
        if sha(ROOT.parent / "data" / name) != expected:
            raise ValueError(f"V1 source changed: {name}. Do not silently rebuild against different inputs.")
    corpus = {x["faq_id"]: x for x in map(json.loads, CORPUS_PATH.read_text(encoding="utf-8").splitlines())}
    slots, facts, cases, exclusions = {}, {}, [], []
    occurrences = Counter()

    for row in read_rows("facts_a.tsv", 6) + read_rows("facts_b.tsv", 6):
        number, family, evidence, purpose, query, unused_alternate = row
        occurrences[number] += 1
        slot = f"{number}-{occurrences[number]}"
        doc = corpus[faq_id(number)]
        if evidence not in doc["answer"]:
            raise ValueError(f"Evidence missing: {slot}")
        specs = ATOMIC.get(slot, [(evidence, evidence)])
        ids = []
        for index, (quote, statement) in enumerate(specs, 1):
            if quote not in doc["answer"]:
                raise ValueError(f"Atomic quote missing: {slot}:{index}")
            fid = f"F-{slot}-{index}"
            supports = [{"faq_id": doc["faq_id"], "quote": quote,
                         "scope_question": doc["question"], "review": "author_source_check"}]
            for alias in ALIASES.get(f"{slot}:{index}", []):
                alt = corpus[faq_id(alias)]
                supports.append({"faq_id": alt["faq_id"], "quote": alt["answer"],
                                 "scope_question": alt["question"], "review": "author_support_proposal"})
            facts[fid] = {"fact_id": fid, "statement": statement, "policy_family": family,
                          "source_slot": slot, "source_question_scope": doc["question"],
                          "support": supports, "acceptable_faq_ids": sorted(s["faq_id"] for s in supports),
                          "independent_support_review": "pending"}
            ids.append(fid)
        slots[slot] = {"fact_ids": ids, "family": family, "category": doc["category"],
                       "purpose": purpose, "query": query, "source_faq_id": doc["faq_id"]}

    def new_case(cid, query, purpose, selected_slots, status="FAQ_EXISTS", selected_facts=None, **extra):
        fact_ids = selected_facts if selected_facts is not None else [f for s in selected_slots for f in slots[s]["fact_ids"]]
        required = [{"fact_id": f, "acceptable_faq_ids": facts[f]["acceptable_faq_ids"]} for f in dict.fromkeys(fact_ids)]
        category = slots[selected_slots[0]]["category"]
        case = {
            "case_id": cid, "query": query, "model_query": query, "category": category,
            "primary_purpose": purpose, "purpose_tags": [purpose], "language": "ko",
            "corpus_status": status,
            "expected_action": {"FAQ_EXISTS": "ANSWER_IF_CONTEXT_SUPPORTS_ALL_FACTS", "NO_FAQ": "ABSTAIN_UNSUPPORTED",
                                "PARTIAL_FACTS_EXIST": "ANSWER_SUPPORTED_PART_AND_STATE_LIMIT", "UNDERSPECIFIED": "CLARIFY"}[status],
            "required_facts": required,
            "expected_faq_ids": sorted({d for f in required for d in f["acceptable_faq_ids"]}),
            "policy_families": sorted({slots[s]["family"] for s in selected_slots}),
            "pair_ids": [], "unsupported_facts": [], "dialogue_history": [],
            "origin": "new_assistant_authored_synthetic", "review_status": "independent_review_pending",
            "source_faq_ids": sorted({slots[s]["source_faq_id"] for s in selected_slots}),
            "eligible_for_final_benchmark": False,
            "evaluation_track": ("context" if purpose == "CONTEXT_RETRIEVAL" else
                "answerability" if status != "FAQ_EXISTS" else
                "core" if purpose in {"PARAPHRASE", "MULTI_INTENT", "COMPARISON"} else "controlled"),
        }
        case.update(extra)
        cases.append(case)
        return case

    for slot, source in slots.items():
        new_case("Q-" + slot, source["query"], source["purpose"], [slot])
    by_id = {c["case_id"]: c for c in cases}
    for slot, kind, query in read_rows("paired_variants.tsv", 3):
        purpose = {"english": "KR_EN_EQUIVALENT", "mixed": "KR_EN_EQUIVALENT", "typo": "TYPO", "long": "LONG_QUERY"}[kind]
        base = by_id["Q-" + slot]
        pair = f"PAIR-{slot}-" + ("language" if kind in {"english", "mixed"} else kind)
        if pair not in base["pair_ids"]:
            base["pair_ids"].append(pair)
        new_case(f"Q-{slot}-{kind}", query, purpose, [slot], pair_ids=[pair], paired_base_id=base["case_id"],
                 language={"english": "en", "mixed": "ko-en"}.get(kind, "ko"),
                 purpose_tags=sorted({purpose, slots[slot]["purpose"]}),
                 paired_invariants=["intent", "entity", "conditions", "numbers_and_units", "required_facts"])

    for i, (purpose, refs, query) in enumerate(read_rows("bundles.tsv", 3), 1):
        refs = refs.split(",")
        if purpose == "COMPARISON" and refs[0] not in COMPARISON_EXPORT_SLOTS:
            exclusions.append({"authoring_row": f"bundles.tsv:{i}", "reason": "Different unrelated attributes, not a clean comparison"})
            continue
        new_case(f"B-{i:03}", query, purpose, refs)

    missing_rows = read_rows("missing_facts.tsv", 5)
    missing = {}
    for i, (anchor, kind, query, proposition, rationale) in enumerate(missing_rows, 1):
        m = {"missing_fact_id": f"U-{i:03}", "requested_fact": proposition, "absence_rationale": rationale,
             "subtype": kind, "corpus_sha256": sha(CORPUS_PATH), "full_corpus_absence_review": "pending",
             "absence_is_not_proven_by_keyword_search": True}
        missing[i] = m
        new_case(f"N-{i:03}", query, "NO_FAQ", [anchor], status="NO_FAQ", selected_facts=[],
                 policy_families=[f"unsupported-{i:03}"], unsupported_facts=[m], negative_subtype=kind,
                 category_role="topic_stratum_only; not sent to model")
    negatives = {c["case_id"]: c for c in cases if c["corpus_status"] == "NO_FAQ"}
    for index, language, query in read_rows("missing_language_pairs.tsv", 3):
        number = int(index)
        base = negatives[f"N-{number:03}"]
        pair = f"PAIR-N-{number:03}-language"
        if pair not in base["pair_ids"]:
            base["pair_ids"].append(pair)
        anchor = missing_rows[number - 1][0]
        new_case(f"N-{number:03}-{language}", query, "NO_FAQ", [anchor], status="NO_FAQ", selected_facts=[],
                 policy_families=base["policy_families"], unsupported_facts=base["unsupported_facts"],
                 negative_subtype=base["negative_subtype"], language=language, paired_base_id=base["case_id"],
                 pair_ids=[pair], purpose_tags=["NO_FAQ", "KR_EN_EQUIVALENT"],
                 paired_invariants=["intent", "entity", "conditions", "numbers_and_units", "missing_fact"])
    for i, (anchor, missing_index, query) in enumerate(read_rows("partial.tsv", 3), 1):
        new_case(f"P-{i:03}", query, "PARTIAL", [anchor], status="PARTIAL_FACTS_EXIST",
                 unsupported_facts=[missing[int(missing_index)]])
    for i, (slot, history, raw, rewrite) in enumerate(read_rows("context.tsv", 4), 1):
        pair = f"CONTEXT-{i:03}"
        new_case(f"A-{i:03}", raw, "AMBIGUOUS_QUERY", [slot], status="UNDERSPECIFIED", selected_facts=[],
                 pair_ids=[pair], missing_information="대상 서비스/변경/행위의 선행 문맥", source_faq_ids=[])
        new_case(f"T-{i:03}", raw, "CONTEXT_RETRIEVAL", [slot], pair_ids=[pair],
                 dialogue_history=[{"role": "user", "content": history}], model_query=rewrite,
                 rewrite_version="human-authored-context-rule-v2.1",
                 rewrite_provenance="Only user history and current turn; no FAQ answer supplied to rewrite")
    for kind, name, left, right, changed, qleft, qright, note in read_rows("contrasts.tsv", 8):
        if name in CONTRAST_EXCLUSIONS:
            exclusions.append({"authoring_row": name, "reason": CONTRAST_EXCLUSIONS[name]})
            continue
        refs = [left, right]
        for side, ref, query in zip(("a", "b"), refs, (qleft, qright)):
            slot, index = ref.split(":")
            fids = [slots[slot]["fact_ids"][int(index) - 1]]
            opposite, other_index = refs[1 if side == "a" else 0].split(":")
            other_fid = slots[opposite]["fact_ids"][int(other_index) - 1]
            rival = sorted(set(facts[other_fid]["acceptable_faq_ids"]) - set(facts[fids[0]]["acceptable_faq_ids"]))
            new_case(f"D-{name}-{side}", query, kind, [slot], selected_facts=fids,
                     pair_ids=["CONTRAST-" + name], contrast_condition=changed, contrast_note=note,
                     distractor_faq_ids=rival if kind == "HARD_NEGATIVE" else [],
                     answer_reasoning_evaluated_by_retrieval=False)

    component_count = connected_split(cases, protocol["split"]["ratios"], protocol["split"]["seed"])
    for c in cases:
        c["binary_answerability_target"] = {"FAQ_EXISTS": 1, "NO_FAQ": 0}.get(c["corpus_status"])
        c["binary_fit_eligible"] = c["binary_answerability_target"] is not None and c["primary_purpose"] not in {"LONG_QUERY", "TYPO", "CONTEXT_RETRIEVAL"}
        c["aggregate_unit"] = c["leakage_group_id"]

    # Suggested review candidates are a model-blind character-bigram helper.
    # They neither create GT aliases nor certify that an answer is absent.
    cg = {fid: grams(d["question"] + " " + d["answer"]) for fid, d in corpus.items()}
    idf = {g: math.log((1 + len(corpus)) / (1 + sum(g in ds for ds in cg.values()))) for g in set().union(*cg.values())}
    authored_by_id = {c["case_id"]: c for c in cases}
    for c in cases:
        review_query = authored_by_id.get(c.get("paired_base_id"), c)["model_query"]
        qg = grams(review_query)
        ordered = sorted(corpus, key=lambda d: (-sum(idf[g] for g in sorted(qg & cg[d])), d))
        c["review_candidate_faq_ids"] = [d for d in ordered if d not in c["expected_faq_ids"]][:12]

    facts_list = list(facts.values())
    old = list(map(json.loads, (ROOT.parent / "data/cases.jsonl").read_text(encoding="utf-8").splitlines()))
    old_queries = {normalized(c.get("model_query", c["query"])) for c in old}
    source_queries = [normalized(d["question"]) for d in corpus.values()]
    overlaps = []
    for c in cases:
        nq = normalized(c["model_query"])
        if nq in old_queries:
            overlaps.append({"case_id": c["case_id"], "kind": "old_exact_query"})
        for d, sq in zip(corpus.values(), source_queries):
            if sq and sq in nq:
                overlaps.append({"case_id": c["case_id"], "kind": "source_question_substring", "faq_id": d["faq_id"]})

    counts = lambda key, rows=cases: dict(sorted(Counter(c[key] for c in rows).items()))
    audit = {
        "version": "fair_eval_v2", "case_count": len(cases), "fact_count": len(facts),
        "category_counts": counts("category"), "primary_purpose_counts": counts("primary_purpose"),
        "corpus_status_counts": counts("corpus_status"), "track_counts": counts("evaluation_track"),
        "split_counts": counts("split"), "connected_components": component_count,
        "multi_fact_cases": sum(len(c["required_facts"]) > 1 for c in cases),
        "source_question_or_old_query_overlap": overlaps,
        "unique_model_queries": len({c["model_query"] for c in cases}),
        "paired_groups": len({p for c in cases for p in c["pair_ids"]}),
        "unique_missing_fact_claims": len(missing),
        "no_faq_connected_groups": len({c["leakage_group_id"] for c in cases if c["corpus_status"] == "NO_FAQ"}),
        "language_by_status": {status: counts("language", [c for c in cases if c["corpus_status"] == status])
                               for status in sorted({c["corpus_status"] for c in cases})},
        "independently_reviewed_cases": 0, "eligible_for_final_benchmark": 0,
        "v1_data_hashes_unchanged": BASELINE_HASHES,
        "previous_result_files_read": [], "model_runs_on_v2": 0,
        "excluded_drafts": exclusions, "unused_alternate_base_queries": len(slots),
        "remaining_limits": [
            "Synthetic queries from the same author; no independent semantic/alias/absence adjudication yet.",
            "Source facts have appeared in earlier experiments; new query holdout is not unseen knowledge.",
            "Purpose/category balance is diagnostic, not an estimate of actual traffic distribution.",
            "Full English and code mixing are covered; transliteration/abbreviation coverage is limited.",
            "Natural long questions are distractor controls, not an 8K-token or reranker-window stress test.",
            "Only 34 distinct NO_FAQ fact claims in 33 connected groups; language variants are not independent negatives.",
            "Partial queries are synthetic combinations; evaluate in their own track.",
            "Human approval can change aliases and connected components; freeze a new version before model runs.",
        ],
        "split_breakdown": {s: {key: counts(key, [c for c in cases if c["split"] == s])
                                  for key in ("category", "primary_purpose", "corpus_status")}
                            for s in SPLITS},
        "distinct_no_faq_claims_by_split": {s: len({u["missing_fact_id"] for c in cases
             if c["split"] == s and c["corpus_status"] == "NO_FAQ" for u in c["unsupported_facts"]}) for s in SPLITS},
    }
    inputs = [ROOT / "protocol.json", ROOT / "fact_spec.py", ROOT / "build_dataset.py", ROOT / "dataset.py", ROOT / "test_dataset.py"] + sorted((ROOT / "authoring").glob("*.tsv"))
    manifest = {"dataset_version": "fair_eval_v2", "status": "frozen_author_draft_pending_independent_review",
                "size_policy": protocol["size_policy"], "corpus_sha256": sha(CORPUS_PATH),
                "input_hashes": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in inputs},
                "counts": {"cases": len(cases), "facts": len(facts), "families": component_count},
                "holdout_access": "author-visible, never used in V2 model runs", "release_eligible": False}
    dump(ROOT / "data/cases.jsonl", cases, lines=True)
    dump(ROOT / "data/facts.jsonl", facts_list, lines=True)
    dump(ROOT / "data/absence_claims.jsonl", list(missing.values()), lines=True)
    for s in SPLITS:
        split_cases = [c for c in cases if c["split"] == s]
        dump(ROOT / f"data/splits/{s}.jsonl", split_cases, lines=True)
        # Inference cannot consume target IDs, answer text, purpose/category or status.
        dump(ROOT / f"data/inputs/{s}.jsonl", [{"case_id": c["case_id"], "query": c["model_query"]} for c in split_cases], lines=True)
    dump(ROOT / "data/audit.json", audit)
    manifest["artifact_hashes"] = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
                                   for p in sorted((ROOT / "data").rglob("*.jsonl"))}
    dump(ROOT / "data/manifest.json", manifest)
    dump(ROOT / "review/decisions.example.json", {
        "case_id": cases[0]["case_id"], "reviewer": "REPLACE_WITH_INDEPENDENT_REVIEWER",
        "independent_of_author": True, "decision": "approve_or_reject", "semantic_check": False,
        "atomic_fact_check": False, "alternative_support_check": False,
        "full_corpus_absence_check": False, "corpus_sha256": sha(CORPUS_PATH),
        "cases_sha256": sha(ROOT / "data/cases.jsonl"), "notes": "Record evidence and corrections; never rubber-stamp."})
    decisions_path = ROOT / "review/decisions.jsonl"
    if not decisions_path.exists():
        decisions_path.write_text("", encoding="utf-8")
    render_review(cases, facts, corpus, audit)
    print(json.dumps({k: audit[k] for k in ("case_count", "fact_count", "connected_components", "split_counts", "primary_purpose_counts", "source_question_or_old_query_overlap")}, ensure_ascii=False, indent=2))


def render_review(cases, facts, corpus, audit):
    esc = lambda value: html.escape(str(value), quote=True)
    sections = []
    for c in cases:
        evidence = "".join("<li><b>" + esc(facts[f["fact_id"]]["statement"]) + "</b><ul>" + "".join(
            f'<li><a href="corpus.html#{esc(s["faq_id"])}">{esc(s["faq_id"])}</a> {esc(s["quote"])}</li>'
            for s in facts[f["fact_id"]]["support"]) + "</ul></li>" for f in c["required_facts"])
        missing = "".join(f'<li>{esc(u["requested_fact"])} — {esc(u["absence_rationale"])}</li>' for u in c["unsupported_facts"])
        candidates = " · ".join(f'<a href="corpus.html#{esc(d)}">{esc(d)}</a>' for d in c["review_candidate_faq_ids"])
        history = " / ".join(x["content"] for x in c["dialogue_history"])
        sections.append(f'''<article data-text="{esc(c['category']+' '+c['primary_purpose']+' '+c['case_id']+' '+c['query'])}">
<h2>{esc(c['case_id'])} · {esc(c['category'])} · {esc(c['primary_purpose'])}</h2>
<p class="badge">{esc(c['split'])} / {esc(c['corpus_status'])} / 독립 검수 대기</p>
<p><strong>질문:</strong> {esc(c['query'])}</p>
{('<p>이전 사용자 발화: '+esc(history)+'</p><p>고정 검색 입력: '+esc(c['model_query'])+'</p>') if history else ''}
<details><summary>사실별 근거·대체 정답 ({len(c['required_facts'])}개 사실)</summary><ol>{evidence}</ol></details>
{('<p>없는 것으로 제안한 사실 — 전체 코퍼스 의미 검수 필요</p><ul>'+missing+'</ul>') if missing else ''}
<details><summary>문자 유사도 기반 검수 후보 — 정답·부재 판정 아님</summary><p>{candidates}</p></details>
<p class="small">{esc(c.get('contrast_note',''))}</p></article>''')
    css = "body{max-width:1050px;margin:32px auto;padding:0 20px;font:16px/1.7 system-ui;color:#172738;background:#f6f8fb}article{background:white;border:1px solid #ccd5df;padding:20px;margin:16px 0;border-radius:10px}h2{font-size:18px}summary{cursor:pointer}input{width:95%;padding:14px;font-size:16px}.badge,.small{color:#526274;font-size:14px}a{color:#0758a8}header{background:#eaf2fb;padding:20px}li{margin:8px 0}"
    top = f"<h1>공정성 보완 평가셋 v2 — 검수본</h1><p>{len(cases)}문항 · {len(facts)}개 사실 · {len(audit['category_counts'])}분야. 모델 점수는 사용하지 않았습니다.</p><p>의미·대체 정답·NO_FAQ 부재의 독립 검수는 아직 없습니다. 아래 라벨은 작성자의 제안입니다. <a href='corpus.html'>전체 FAQ 1,024건</a>을 함께 확인해야 합니다.</p><p>development / calibration / holdout은 정책·사실·대체 정답·대조쌍 연결 단위로 분리했습니다. 이 페이지는 작성자에게 holdout을 숨기는 장치가 아닙니다.</p>"
    script = "<script>document.querySelector('input').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('article').forEach(a=>a.hidden=!a.dataset.text.toLowerCase().includes(q))})</script>"
    page = f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>UBot fair eval v2</title><style>{css}</style><header>{top}</header><p><input aria-label="문항 검색" placeholder="분야·목적·ID·질문으로 찾기"></p>'+"".join(sections)+script+"</html>"
    (ROOT / "review/index.html").write_text(page, encoding="utf-8")
    docs = "".join(f'<article id="{esc(fid)}" data-text="{esc(d["question"]+" "+d["answer"])}"><h2>{esc(fid)} · {esc(d["category"])}</h2><p>{esc(d["question"])}</p><p>{esc(d["answer"])}</p></article>' for fid, d in corpus.items())
    (ROOT / "review/corpus.html").write_text(f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>원천 FAQ</title><style>{css}</style><h1>고정 원천 FAQ 1,024건</h1><p><a href="index.html">검수 문항으로</a></p><input aria-label="코퍼스 검색" placeholder="질문·답변 검색">'+docs+script+"</html>", encoding="utf-8")


if __name__ == "__main__":
    build()
