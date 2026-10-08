"""Model-free contracts for the UBot retrieval benchmark.

Scores are per case. Aggregation must retain reporting_cohort and family_id.
Empty gold is not a failed retrieval: it has no ranking denominator.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime


def stable_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def unique_ranking(ids):
    return list(dict.fromkeys(ids))


def fact_metrics(ranked_ids, fact_groups, k):
    """AND across facts, OR across equivalent evidence within each fact."""
    if k < 1:
        raise ValueError("k must be positive")
    ids = unique_ranking(ranked_ids)[:k]
    if not fact_groups:
        return {"strict_hit": None, "semantic_hit": None, "fact_recall": None,
                "all_facts_hit": None, "semantic_mrr": None, "fact_mrr": None,
                "fact_ranks": {}, "actual_count": len(ids)}
    ranks = {faq_id: rank for rank, faq_id in enumerate(ids, 1)}
    fact_ranks = {}
    strict_hit = False
    for group in fact_groups:
        relevant = set(group["primary_ids"]) | set(group["acceptable_ids"])
        found = [ranks[fid] for fid in relevant if fid in ranks]
        fact_ranks[group["fact_id"]] = min(found) if found else None
        strict_hit |= bool(set(group["primary_ids"]) & ranks.keys())
    values = list(fact_ranks.values())
    hit_count = sum(rank is not None for rank in values)
    first_rank = min((rank for rank in values if rank is not None), default=None)
    return {"strict_hit": int(strict_hit), "semantic_hit": int(hit_count > 0),
            "fact_recall": hit_count / len(values), "all_facts_hit": int(hit_count == len(values)),
            "semantic_mrr": 1 / first_rank if first_rank else 0.0,
            "fact_mrr": sum(1 / rank for rank in values if rank) / len(values),
            "fact_ranks": fact_ranks, "actual_count": len(ids)}


def merge_dense_views(question_scores, content_scores, per_view_k, budget):
    """Raw top-k union; rank by max Q/QA cosine, deterministic ID tie break.

    Inputs contain the FULL scores over the SAME corpus. Do not substitute
    top-k lists, which would lose the other view's score for a candidate.
    """
    if per_view_k < 1 or budget < 1:
        raise ValueError("candidate counts must be positive")
    if question_scores.keys() != content_scores.keys():
        raise ValueError("both views must score the same corpus")
    if any(not math.isfinite(float(v)) for v in [*question_scores.values(), *content_scores.values()]):
        raise ValueError("non-finite embedding score")
    q = sorted(question_scores, key=lambda fid: (-question_scores[fid], fid))[:per_view_k]
    qa = sorted(content_scores, key=lambda fid: (-content_scores[fid], fid))[:per_view_k]
    union = set(q) | set(qa)
    ranked = sorted(union, key=lambda fid: (-max(question_scores[fid], content_scores[fid]), fid))
    return {"raw_union_ids": ranked, "top_m_ids": ranked[:budget], "actual_u": len(union),
            "channels": {fid: [name for name, ids in [("question", q), ("question_answer", qa)] if fid in ids]
                         for fid in ranked}}


def no_faq_metrics(labels, accepted):
    """NO_FAQ is positive; unsafe acceptance is its false-negative rate."""
    if len(labels) != len(accepted):
        raise ValueError("labels and decisions must have the same length")
    if any(not isinstance(decision, bool) for decision in accepted):
        raise ValueError("accepted decisions must be booleans")
    if any(label not in {"full", "none"} for label in labels):
        raise ValueError("binary calibration accepts only full and none labels")
    tp = sum(label == "none" and not keep for label, keep in zip(labels, accepted))
    fp = sum(label == "full" and not keep for label, keep in zip(labels, accepted))
    fn = sum(label == "none" and keep for label, keep in zip(labels, accepted))
    tn = sum(label == "full" and keep for label, keep in zip(labels, accepted))
    ratio = lambda num, den: num / den if den else None
    return {"positive_class": "NO_FAQ", "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "detector_false_positive_rate": ratio(fp, fp + tn),
            "unsafe_accept_rate": ratio(fn, tp + fn)}


def _aware_timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("temporal timestamps require a timezone")
    return parsed


def active_policy_ids(temporal_context):
    as_of = _aware_timestamp(temporal_context["as_of"])
    result = []
    for version in temporal_context["versions"]:
        start = _aware_timestamp(version["valid_from"])
        end = _aware_timestamp(version["valid_until"]) if version["valid_until"] else None
        if end is not None and end <= start:
            raise ValueError("valid_until must be after valid_from")
        if start <= as_of and (end is None or as_of < end):
            result.append(version["faq_id"])
    return sorted(result)


def semantic_contract_errors(cases, catalog=None):
    """Checks beyond JSON Schema. Call after structural validation."""
    errors = []
    case_ids = set()
    split_keys = {}
    pair_groups = defaultdict(list)
    contrast_groups = defaultdict(list)
    targeted = {"EXACT_ENTITY", "HARD_NEGATIVE", "KR_EN_EQUIVALENT", "NEGATION", "CONDITION_SCOPE", "NUMERIC_CONDITION"}
    for c in cases:
        cid = c["case_id"]
        fail = lambda msg: errors.append(f"{cid}: {msg}")
        if cid in case_ids:
            fail("duplicate case_id")
        case_ids.add(cid)
        groups = c["fact_groups"]
        fact_ids = [g["fact_id"] for g in groups]
        if len(set(fact_ids)) != len(fact_ids):
            fail("duplicate fact_id")
        gold = {fid for g in groups for fid in g["primary_ids"] + g["acceptable_ids"]}
        if gold & set(c["hard_negative_ids"]):
            fail("GT overlaps hard negatives")
        if c["primary_type"] != "STANDARD" and c["primary_type"] not in c["tags"]:
            fail("primary_type must be present in tags")
        if c["primary_type"] in targeted and c["track"] == "core" and c["reporting_cohort"] != "targeted_condition":
            fail("targeted condition must not enter the core_general average")
        cohorts = {"contextual": "contextual", "no_faq": "no_faq", "special": "special_corpus", "repeat": "repeat", "clarify": "clarify"}
        if c["track"] in cohorts and c["reporting_cohort"] != cohorts[c["track"]]:
            fail("track and reporting_cohort disagree")
        if c["answerability"] == "ambiguous" and "NO_FAQ" in c["tags"]:
            fail("ambiguous input is not NO_FAQ")
        if c["track"] == "repeat" and c["repeat"]["index"] > c["repeat"]["total"]:
            fail("repeat index exceeds total")
        if catalog is not None:
            corpus = catalog.get(c["corpus_version"])
            if corpus is None:
                fail("unknown corpus_version")
            elif (gold | set(c["hard_negative_ids"])) - set(corpus):
                fail("gold or hard negative missing from this corpus")
        keys = [("family", c["family_id"]), ("split_group", c["split_group_id"])]
        if c.get("semantic_pair_id"):
            pair_groups[c["semantic_pair_id"]].append(c)
            keys.append(("semantic_pair", c["semantic_pair_id"]))
        if c.get("contrast_pair_id"):
            contrast_groups[c["contrast_pair_id"]].append(c)
            keys.append(("contrast_pair", c["contrast_pair_id"]))
            if not c.get("contrast_axis"):
                fail("contrast_pair_id requires contrast_axis")
        if c["source"].get("original_question_id"):
            keys.append(("original", c["source"]["source_key"], c["source"]["original_question_id"]))
        if c["source"].get("scenario_group"):
            keys.append(("scenario", c["source"]["source_key"], c["source"]["scenario_group"]))
        if c.get("repeat"):
            keys.append(("repeat", c["repeat"]["group_id"]))
        for key in keys:
            previous = split_keys.setdefault(key, c["split"])
            if previous != c["split"]:
                fail(f"split leakage for {key}")
        subrequests = c.get("subrequests", [])
        if subrequests:
            if len({r["request_id"] for r in subrequests}) != len(subrequests):
                fail("duplicate subrequest ID")
            if set(r["route"] for r in subrequests) != set(c["expected_routes"]):
                fail("subrequest routes disagree with expected_routes")
            referenced = {fid for r in subrequests for fid in r["retrieval_fact_ids"]}
            if referenced != set(fact_ids):
                fail("subrequests must partition or share all retrieval fact IDs")
        if c.get("temporal_context"):
            try:
                active = active_policy_ids(c["temporal_context"])
                if active != sorted(c["temporal_context"]["expected_policy_ids"]):
                    fail("expected policy disagrees with validity intervals")
                if catalog is not None and c["corpus_version"] in catalog:
                    corpus = catalog[c["corpus_version"]]
                    for version in c["temporal_context"]["versions"]:
                        source_doc = corpus.get(version["faq_id"], {})
                        if any(version[k] != source_doc.get(k) for k in ["valid_from", "valid_until"]):
                            fail("temporal metadata differs from source fixture")
            except (ValueError, KeyError, TypeError) as exc:
                fail(f"invalid temporal metadata: {exc}")
        if c.get("attack_context"):
            if not set(c["attack_context"]["fixture_document_ids"]) <= gold:
                fail("attack fixture must be represented in exposure GT")
    conserved = ["fact_groups", "expected_routes", "answerability", "retrieval_status", "expected_generation_status",
                 "execution_condition_id", "meaning_constraints", "family_id", "split_group_id", "split", "corpus_version"]
    for pid, group in pair_groups.items():
        variants = [c.get("language_variant") for c in group]
        if not {"ko_anchor", "en_term"} <= set(variants):
            errors.append(f"{pid}: ko_anchor and en_term variants required")
        if len(set(variants)) != len(variants):
            errors.append(f"{pid}: duplicate language_variant")
        for field in conserved:
            if any(c.get(field) != group[0].get(field) for c in group):
                errors.append(f"{pid}: language pair changed {field}")
    for pid, group in contrast_groups.items():
        for field in ["family_id", "split_group_id", "split", "corpus_version", "contrast_axis"]:
            if any(c.get(field) != group[0].get(field) for c in group):
                errors.append(f"{pid}: contrast pair changed {field}")
    return errors
