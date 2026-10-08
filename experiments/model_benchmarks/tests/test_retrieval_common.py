from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from retrieval_common import (active_policy_ids, fact_metrics, merge_dense_views,
                              no_faq_metrics, semantic_contract_errors, stable_hash)
from validate_retrieval_dataset import validate


def fact(fid, primary, acceptable=()):
    return {"fact_id": fid, "requirement": fid, "primary_ids": list(primary), "acceptable_ids": list(acceptable)}


class MetricsTests(unittest.TestCase):
    def test_and_between_facts_or_within_fact(self):
        gold = [fact("one", ["FAQ-001"], ["FAQ-002"]), fact("two", ["FAQ-003"])]
        partial = fact_metrics(["FAQ-002", "FAQ-004"], gold, 2)
        self.assertEqual(partial["semantic_hit"], 1)
        self.assertEqual(partial["strict_hit"], 0)
        self.assertEqual(partial["fact_recall"], 0.5)
        self.assertEqual(partial["all_facts_hit"], 0)
        self.assertEqual(fact_metrics(["FAQ-002", "FAQ-003"], gold, 2)["all_facts_hit"], 1)

    def test_one_document_can_cover_multiple_facts(self):
        gold = [fact("two_lines", ["FAQ-865"]), fact("three_lines", ["FAQ-865"])]
        self.assertEqual(fact_metrics(["FAQ-865"], gold, 1)["fact_recall"], 1.0)

    def test_candidate_cutoff_and_missing_fact_rr(self):
        gold = [fact("one", ["FAQ-003"]), fact("two", ["FAQ-010"])]
        self.assertEqual(fact_metrics(["FAQ-001", "FAQ-003"], gold, 1)["semantic_mrr"], 0.0)
        self.assertEqual(fact_metrics(["FAQ-001", "FAQ-003"], gold, 2)["fact_mrr"], 0.25)

    def test_no_gold_is_na_not_zero_or_vacuous_success(self):
        metrics = fact_metrics(["FAQ-001"], [], 3)
        for key in ["semantic_hit", "strict_hit", "fact_recall", "all_facts_hit", "semantic_mrr"]:
            self.assertIsNone(metrics[key])

    def test_dedupe_before_rank_and_cutoff(self):
        metric = fact_metrics(["FAQ-001", "FAQ-001", "FAQ-003"], [fact("f", ["FAQ-003"])], 2)
        self.assertEqual(metric["semantic_mrr"], 0.5)

    def test_dual_union_and_budget_are_distinct(self):
        q = {"FAQ-001": 0.9, "FAQ-002": 0.7, "FAQ-003": 0.1}
        qa = {"FAQ-001": 0.1, "FAQ-002": 0.7, "FAQ-003": 0.95}
        merge = merge_dense_views(q, qa, 1, 1)
        self.assertEqual(merge["actual_u"], 2)
        self.assertEqual(merge["raw_union_ids"], ["FAQ-003", "FAQ-001"])
        self.assertEqual(merge["top_m_ids"], ["FAQ-003"])

    def test_dense_merge_tie_is_deterministic_and_not_duplicated(self):
        scores = {"FAQ-002": 0.9, "FAQ-001": 0.9}
        self.assertEqual(merge_dense_views(scores, scores, 2, 2)["top_m_ids"], ["FAQ-001", "FAQ-002"])

    def test_dense_merge_rejects_mixed_corpus_and_nan(self):
        with self.assertRaises(ValueError):
            merge_dense_views({"FAQ-001": 0.5}, {"FAQ-002": 0.5}, 1, 1)
        with self.assertRaises(ValueError):
            merge_dense_views({"FAQ-001": float("nan")}, {"FAQ-001": 0.5}, 1, 1)

    def test_no_faq_positive_orientation(self):
        # Of four absent-answer cases, three are dangerously accepted.
        metric = no_faq_metrics(["none"] * 4 + ["full"] * 4,
                                [True, True, True, False, False, True, True, True])
        self.assertEqual(metric["recall"], 0.25)
        self.assertEqual(metric["unsafe_accept_rate"], 0.75)
        self.assertEqual(metric["detector_false_positive_rate"], 0.25)

    def test_no_faq_metrics_do_not_force_partial_or_ambiguous_into_binary(self):
        for label in ["partial", "ambiguous", "route_only"]:
            with self.assertRaises(ValueError):
                no_faq_metrics([label], [False])
        self.assertIsNone(no_faq_metrics([], [])["recall"])
        with self.assertRaises(ValueError):
            no_faq_metrics(["none"], ["false"])

    def test_stable_hash_keeps_semantics_and_rejects_nan(self):
        self.assertEqual(stable_hash({"a": 1, "b": 2}), stable_hash({"b": 2, "a": 1}))
        self.assertNotEqual(stable_hash(["a", "b"]), stable_hash(["b", "a"]))
        with self.assertRaises(ValueError):
            stable_hash(float("nan"))


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = [json.loads(line) for line in (ROOT / "faq/retrieval_design_examples.jsonl").read_text(encoding="utf-8").splitlines()]
        cls.schema = json.loads((ROOT / "schemas/retrieval_case.schema.json").read_text(encoding="utf-8"))

    def test_design_examples_are_structurally_valid(self):
        self.assertEqual(validate(copy.deepcopy(self.examples), self.schema, None, True), [])

    def test_design_samples_cannot_silently_enter_real_evaluation(self):
        errors = validate(copy.deepcopy(self.examples), self.schema, None, False)
        self.assertEqual(sum("unapproved" in err for err in errors), len(self.examples))

    def test_language_variant_cannot_change_gold_or_number_constraints(self):
        cases = copy.deepcopy(self.examples)
        en = next(c for c in cases if c.get("language_variant") == "en_term")
        en["meaning_constraints"]["numbers"] = ["100"]
        self.assertTrue(any("language pair changed meaning_constraints" in err for err in semantic_contract_errors(cases)))

    def test_family_and_language_split_leakage_is_detected(self):
        cases = copy.deepcopy(self.examples)
        cases[5]["split"] = "holdout"
        self.assertTrue(any("split leakage" in err for err in semantic_contract_errors(cases)))

    def test_same_source_scenario_cannot_leak_through_renamed_families(self):
        first = copy.deepcopy(self.examples[0])
        second = copy.deepcopy(first)
        second.update(case_id="copy", family_id="renamed", split_group_id="renamed", split="holdout")
        self.assertTrue(any("split leakage" in err for err in semantic_contract_errors([first, second])))

    def test_gold_and_hard_negative_overlap_is_rejected(self):
        case = copy.deepcopy(self.examples[0])
        case["hard_negative_ids"] = case["fact_groups"][0]["primary_ids"]
        self.assertTrue(any("overlaps" in err for err in semantic_contract_errors([case])))

    def test_ambiguity_cannot_be_relabelled_no_faq(self):
        case = copy.deepcopy(next(c for c in self.examples if c["answerability"] == "ambiguous"))
        case["tags"].append("NO_FAQ")
        self.assertTrue(any("ambiguous input is not NO_FAQ" in err for err in semantic_contract_errors([case])))
        case["retrieval_status"] = "NO_FAQ"
        self.assertTrue(validate([case], self.schema, None, True))

    def test_mult_intent_requires_subrequests_and_existing_fact_references(self):
        case = copy.deepcopy(next(c for c in self.examples if c["primary_type"] == "MULTI_INTENT"))
        case["subrequests"][0]["retrieval_fact_ids"] = ["invented_fact"]
        self.assertTrue(any("retrieval fact IDs" in err for err in semantic_contract_errors([case])))
        case.pop("subrequests")
        self.assertTrue(validate([case], self.schema, None, True))

    def test_temporal_boundary_uses_start_inclusive_end_exclusive(self):
        context = copy.deepcopy(next(c for c in self.examples if c.get("temporal_context"))["temporal_context"])
        context["as_of"] = "2026-10-01T00:00:00+09:00"
        self.assertEqual(active_policy_ids(context), ["TEST-CF-0061-B"])
        context["as_of"] = "2026-09-30T23:59:59+09:00"
        self.assertEqual(active_policy_ids(context), ["TEST-CF-0061-A"])
        context["as_of"] = "2026-10-01T00:00:00"
        with self.assertRaises(ValueError):
            active_policy_ids(context)

    def test_exact_conditions_do_not_enter_general_average(self):
        case = copy.deepcopy(self.examples[1])
        case["reporting_cohort"] = "core_general"
        self.assertTrue(any("core_general average" in err for err in semantic_contract_errors([case])))

    def test_fixture_ids_do_not_exist_in_normal_corpus(self):
        case = copy.deepcopy(self.examples[0])
        case["fact_groups"][0]["primary_ids"] = ["TEST-AD-0141"]
        self.assertTrue(any("missing from this corpus" in err for err in semantic_contract_errors([case], {"current-faq-1024-v1": {"FAQ-291": {}}})))


if __name__ == "__main__":
    unittest.main()
