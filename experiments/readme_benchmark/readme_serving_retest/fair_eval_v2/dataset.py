"""Strict v2 dataset loader and fact-based retrieval scoring.

This module does not call a serving model. Official loading requires independent
decisions for every selected row; provisional diagnostics require an explicit flag.
It deliberately cannot be substituted for V1's flat binary-GT loader silently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class ReviewRequired(ValueError):
    pass


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_artifacts(root=ROOT):
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    for path, expected in {**manifest["input_hashes"], **manifest["artifact_hashes"]}.items():
        if digest(root / path) != expected:
            raise ValueError(f"Frozen input/artifact changed: {path}")
    if digest(root.parent / "data/corpus.jsonl") != manifest["corpus_sha256"]:
        raise ValueError("Source corpus changed")
    return manifest


def decision_valid(case, decision, manifest, cases_hash):
    if not decision:
        return False
    common = (
        decision.get("case_id") == case["case_id"]
        and decision.get("decision") == "approve"
        and decision.get("independent_of_author") is True
        and bool(decision.get("reviewer", "").strip())
        and decision.get("reviewer") not in {"REPLACE_WITH_INDEPENDENT_REVIEWER", "dataset_author", "assistant"}
        and decision.get("semantic_check") is True
        and decision.get("atomic_fact_check") is True
        and decision.get("corpus_sha256") == manifest["corpus_sha256"]
        and decision.get("cases_sha256") == cases_hash
        and bool(decision.get("notes", "").strip())
    )
    if not common:
        return False
    if case["required_facts"] and decision.get("alternative_support_check") is not True:
        return False
    if case["unsupported_facts"] and decision.get("full_corpus_absence_check") is not True:
        return False
    return True


def load_cases(split, *, allow_unreviewed=False, root=ROOT):
    if split not in {"development", "calibration", "holdout"}:
        raise ValueError("An explicit development/calibration/holdout split is required")
    manifest = verify_artifacts(root)
    cases = read_jsonl(root / f"data/splits/{split}.jsonl")
    if allow_unreviewed:
        return cases
    decisions = read_jsonl(root / "review/decisions.jsonl")
    ids = [d.get("case_id") for d in decisions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate review decisions; resolve the adjudication record")
    by_id = {d["case_id"]: d for d in decisions}
    cases_hash = digest(root / "data/cases.jsonl")
    missing = [c["case_id"] for c in cases if not decision_valid(c, by_id.get(c["case_id"]), manifest, cases_hash)]
    if missing:
        raise ReviewRequired(f"{len(missing)} rows in {split} lack valid independent review; first: {missing[:5]}. "
                             "Use allow_unreviewed=True only for explicitly labelled provisional diagnostics.")
    for case in cases:
        case["eligible_for_final_benchmark"] = True
        case["review_status"] = "independent_review_approved"
    return cases


def inference_payload(case):
    """Same fixed input for every engine. Never include GT, category or purpose."""
    return {"case_id": case["case_id"], "query": case["model_query"]}


def score_case(case, ranked_faq_ids, *, k=10):
    if k <= 0:
        raise ValueError("k must be positive")
    if len(ranked_faq_ids) != len(set(ranked_faq_ids)):
        raise ValueError("Duplicate FAQ candidates inflate document budgets")
    required = case["required_facts"]
    if not required:
        return {"hit": None, "fact_recall": None, "all_facts_hit": None, "mrr": None,
                "required_fact_count": 0, "k": k, "corpus_status": case["corpus_status"],
                "retrieved_context_status": "CLARIFY" if case["corpus_status"] == "UNDERSPECIFIED" else "UNSUPPORTED"}
    selected = set(ranked_faq_ids[:k])
    hits = [bool(selected.intersection(f["acceptable_faq_ids"])) for f in required]
    relevant = set(case["expected_faq_ids"])
    ranks = [i for i, fid in enumerate(ranked_faq_ids[:k], 1) if fid in relevant]
    all_supported = all(hits)
    # A fully retrieved PARTIAL case still has unsupported requested information.
    context_status = ("FULL" if all_supported and case["corpus_status"] == "FAQ_EXISTS" else
                      "PARTIAL" if any(hits) else "UNSUPPORTED")
    return {"hit": float(any(hits)), "fact_recall": sum(hits) / len(hits),
            "all_facts_hit": float(all_supported), "mrr": 1 / min(ranks) if ranks else 0.0,
            "required_fact_count": len(hits), "k": k, "corpus_status": case["corpus_status"],
            "retrieved_context_status": context_status}


def binary_calibration_target(case, *, retrieved_faq_ids=None, target="corpus_presence"):
    """Never conflate a FAQ existing with this particular context supporting it.

    PARTIAL and ambiguous rows are outside the binary task. Report them separately.
    context_support targets are architecture-specific and need the frozen top-k.
    """
    if case["corpus_status"] not in {"FAQ_EXISTS", "NO_FAQ"}:
        return None
    if target == "corpus_presence":
        return int(case["corpus_status"] == "FAQ_EXISTS")
    if target == "context_support":
        if retrieved_faq_ids is None:
            raise ValueError("Actual final context IDs required")
        return int(score_case(case, retrieved_faq_ids, k=max(1, len(retrieved_faq_ids)))["retrieved_context_status"] == "FULL")
    raise ValueError("target must be corpus_presence or context_support")


def aggregate_by_track(cases, scores, metric="fact_recall"):
    """Return row micro plus purpose/category macro, with every denominator.

    A single pooled headline across core/context/controls/answerability is forbidden.
    Family macro prevents linked language/typo/long variants dominating a family.
    Confidence intervals should resample the reported leakage_group_id jointly.
    """
    tracks = defaultdict(list)
    for case in cases:
        score = scores[case["case_id"]].get(metric)
        if score is not None:
            tracks[case["evaluation_track"]].append((case, score))
    result = {}
    for track, entries in tracks.items():
        strata = {}
        for key in ("primary_purpose", "category", "leakage_group_id"):
            groups = defaultdict(list)
            for case, value in entries:
                groups[case[key]].append(value)
            strata[key] = {g: {"n": len(v), "mean": sum(v) / len(v)} for g, v in sorted(groups.items())}
        result[track] = {
            "n": len(entries), "row_micro": sum(v for _, v in entries) / len(entries),
            "macro": {key: sum(v["mean"] for v in groups.values()) / len(groups) for key, groups in strata.items()},
            "strata": strata,
        }
    return result


def calibration_examples(cases, *, target="corpus_presence", contexts=None, allow_unreviewed=False):
    """Produce weighted calibration rows without fitting or reading any scores."""
    if any(c["split"] != "calibration" for c in cases):
        raise ValueError("Threshold fitting accepts only the calibration split")
    if not allow_unreviewed and any(not c["eligible_for_final_benchmark"] for c in cases):
        raise ReviewRequired("Calibration labels require independent review or explicit provisional opt-in")
    rows = []
    groups = defaultdict(list)
    for c in cases:
        if not c["binary_fit_eligible"]:
            continue
        label = binary_calibration_target(c, target=target,
                    retrieved_faq_ids=None if contexts is None else contexts[c["case_id"]])
        if label is None:
            continue
        row = {"case_id": c["case_id"], "target": label, "leakage_group_id": c["leakage_group_id"]}
        rows.append(row)
        groups[(c["leakage_group_id"], label)].append(row)
    for group in groups.values():
        for row in group:
            row["sample_weight"] = 1 / len(group)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "calibration", "holdout"], required=True)
    parser.add_argument("--allow-unreviewed", action="store_true", help="Provisional diagnostics only")
    args = parser.parse_args()
    try:
        cases = load_cases(args.split, allow_unreviewed=args.allow_unreviewed)
    except ReviewRequired as e:
        parser.exit(2, str(e) + "\n")
    print(json.dumps({"rows": len(cases), "split": args.split,
                      "status": "provisional" if args.allow_unreviewed else "reviewed",
                      "corpus_status": {s: sum(c["corpus_status"] == s for c in cases)
                                        for s in sorted({c["corpus_status"] for c in cases})}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
