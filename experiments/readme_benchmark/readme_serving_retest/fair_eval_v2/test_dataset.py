"""Invariants that catch contamination, stale labels and invalid scoring."""
import copy
import json
import unittest
from unittest import mock
from collections import Counter, defaultdict

import build_dataset as builder
import dataset

ROOT = dataset.ROOT


class FairDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = dataset.read_jsonl(ROOT / "data/cases.jsonl")
        cls.by_id = {c["case_id"]: c for c in cls.cases}
        cls.facts = {f["fact_id"]: f for f in dataset.read_jsonl(ROOT / "data/facts.jsonl")}
        cls.corpus = {d["faq_id"]: d for d in dataset.read_jsonl(ROOT.parent / "data/corpus.jsonl")}
        cls.manifest = dataset.verify_artifacts()

    def test_old_data_untouched(self):
        for name, value in builder.BASELINE_HASHES.items():
            self.assertEqual(dataset.digest(ROOT.parent / "data" / name), value)

    def test_artifact_hashes_and_source(self):
        self.assertEqual(self.manifest["corpus_sha256"], dataset.digest(ROOT.parent / "data/corpus.jsonl"))
        self.assertEqual(len(self.by_id), len(self.cases))

    def test_split_partition(self):
        rows = sum((dataset.read_jsonl(ROOT / f"data/splits/{s}.jsonl") for s in builder.SPLITS), [])
        self.assertEqual(Counter(c["case_id"] for c in rows), Counter(self.by_id.keys()))
        for c in rows:
            self.assertEqual(c, self.by_id[c["case_id"]])

    def test_no_split_leakage(self):
        where = defaultdict(set)
        for c in self.cases:
            keys = (["faq:" + d for d in c["expected_faq_ids"]]
                    + ["fact:" + f["fact_id"] for f in c["required_facts"]]
                    + ["family:" + f for f in c["policy_families"]]
                    + ["pair:" + p for p in c["pair_ids"]]
                    + ["missing:" + f["missing_fact_id"] for f in c["unsupported_facts"]]
                    + ["group:" + c["leakage_group_id"]])
            for key in keys:
                where[key].add(c["split"])
        self.assertTrue(where)
        self.assertFalse({k: v for k, v in where.items() if len(v) != 1})

    def test_evidence_and_fact_level_aliases(self):
        for fact in self.facts.values():
            self.assertEqual(set(fact["acceptable_faq_ids"]), {s["faq_id"] for s in fact["support"]})
            for evidence in fact["support"]:
                self.assertIn(evidence["quote"], self.corpus[evidence["faq_id"]]["answer"])
        for c in self.cases:
            self.assertEqual(set(c["expected_faq_ids"]), {d for f in c["required_facts"] for d in f["acceptable_faq_ids"]})
            for f in c["required_facts"]:
                self.assertEqual(f["acceptable_faq_ids"], self.facts[f["fact_id"]]["acceptable_faq_ids"])

    def test_status_contract(self):
        for c in self.cases:
            status = c["corpus_status"]
            if status == "FAQ_EXISTS":
                self.assertTrue(c["required_facts"])
                self.assertFalse(c["unsupported_facts"])
            elif status == "NO_FAQ":
                self.assertFalse(c["required_facts"])
                self.assertTrue(c["unsupported_facts"])
            elif status == "PARTIAL_FACTS_EXIST":
                self.assertTrue(c["required_facts"])
                self.assertTrue(c["unsupported_facts"])
                self.assertIsNone(dataset.binary_calibration_target(c))
            elif status == "UNDERSPECIFIED":
                self.assertEqual(c["expected_action"], "CLARIFY")
                self.assertFalse(c["required_facts"])
                self.assertIsNone(dataset.binary_calibration_target(c))
            else:
                self.fail(status)

    def test_model_input_has_no_gt(self):
        for s in builder.SPLITS:
            exports = dataset.read_jsonl(ROOT / f"data/inputs/{s}.jsonl")
            for row in exports:
                self.assertEqual(set(row), {"case_id", "query"})
                self.assertEqual(row, dataset.inference_payload(self.by_id[row["case_id"]]))

    def test_pairs_preserve_facts_and_context_ambiguity(self):
        for c in self.cases:
            if "paired_base_id" in c:
                base = self.by_id[c["paired_base_id"]]
                self.assertEqual(c["required_facts"], base["required_facts"])
                self.assertEqual(c["unsupported_facts"], base["unsupported_facts"])
                self.assertEqual(c["split"], base["split"])
                self.assertNotEqual(c["model_query"], base["model_query"])
            if c["primary_purpose"] == "CONTEXT_RETRIEVAL":
                ambiguous = self.by_id[c["case_id"].replace("T-", "A-")]
                self.assertEqual(c["query"], ambiguous["query"])
                self.assertTrue(c["dialogue_history"])
                self.assertNotEqual(c["model_query"], c["query"])
                self.assertFalse(ambiguous["dialogue_history"])

    def test_no_source_question_pasting_or_reused_v1_query(self):
        audit = json.loads((ROOT / "data/audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["source_question_or_old_query_overlap"], [])
        self.assertEqual(audit["unique_model_queries"], len(self.cases))

    def test_review_gate_and_stale_hash(self):
        original_reader = dataset.read_jsonl
        def without_decisions(path):
            return [] if path.name == "decisions.jsonl" else original_reader(path)
        for s in builder.SPLITS:
            with mock.patch.object(dataset, "read_jsonl", side_effect=without_decisions):
                with self.assertRaises(dataset.ReviewRequired):
                    dataset.load_cases(s)
            self.assertTrue(dataset.load_cases(s, allow_unreviewed=True))
        c = self.by_id["N-001"]
        d = {"case_id": c["case_id"], "reviewer": "test-fixture-only", "independent_of_author": True,
             "decision": "approve", "semantic_check": True, "atomic_fact_check": True,
             "alternative_support_check": True, "full_corpus_absence_check": False,
             "corpus_sha256": self.manifest["corpus_sha256"],
             "cases_sha256": dataset.digest(ROOT / "data/cases.jsonl"), "notes": "unit test; never written as a review"}
        self.assertFalse(dataset.decision_valid(c, d, self.manifest, d["cases_sha256"]))
        d["full_corpus_absence_check"] = True
        self.assertTrue(dataset.decision_valid(c, d, self.manifest, d["cases_sha256"]))
        d["cases_sha256"] = "stale"
        self.assertFalse(dataset.decision_valid(c, d, self.manifest, dataset.digest(ROOT / "data/cases.jsonl")))

    def test_fact_scoring_distinguishes_any_and_all(self):
        case = self.by_id["B-018"]  # parental ownership and senior eligibility
        self.assertGreaterEqual(len(case["required_facts"]), 2)
        one_doc = [case["required_facts"][0]["acceptable_faq_ids"][0]]
        score = dataset.score_case(case, one_doc)
        self.assertEqual(score["hit"], 1)
        self.assertGreater(score["fact_recall"], 0)
        self.assertLess(score["fact_recall"], 1)
        self.assertEqual(score["all_facts_hit"], 0)
        self.assertEqual(score["retrieved_context_status"], "PARTIAL")
        with self.assertRaises(ValueError):
            dataset.score_case(case, one_doc * 2)

    def test_corpus_presence_is_not_context_support(self):
        case = self.by_id["Q-103-1"]
        self.assertEqual(dataset.binary_calibration_target(case), 1)
        self.assertEqual(dataset.binary_calibration_target(case, target="context_support", retrieved_faq_ids=["FAQ-001"]), 0)
        self.assertEqual(dataset.binary_calibration_target(case, target="context_support", retrieved_faq_ids=["FAQ-103"]), 1)
        partial = self.by_id["P-001"]
        score = dataset.score_case(partial, partial["expected_faq_ids"])
        self.assertEqual(score["all_facts_hit"], 1)
        self.assertEqual(score["retrieved_context_status"], "PARTIAL")

    def test_deterministic_group_allocation(self):
        duplicate = copy.deepcopy(self.cases)
        protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
        builder.connected_split(duplicate, protocol["split"]["ratios"], protocol["split"]["seed"])
        self.assertEqual([(c["split"], c["leakage_group_id"]) for c in duplicate],
                         [(c["split"], c["leakage_group_id"]) for c in self.cases])

    def test_calibration_excludes_unmatched_controls_and_weights_families(self):
        selected = [c for c in self.cases if c['split'] == 'calibration']
        with self.assertRaises(dataset.ReviewRequired):
            dataset.calibration_examples(selected)
        with self.assertRaises(ValueError):
            dataset.calibration_examples(self.cases, allow_unreviewed=True)
        examples = dataset.calibration_examples(selected, allow_unreviewed=True)
        totals = defaultdict(float)
        for row in examples:
            case = self.by_id[row['case_id']]
            self.assertNotIn(case['primary_purpose'], {'LONG_QUERY','TYPO','CONTEXT_RETRIEVAL','PARTIAL','AMBIGUOUS_QUERY'})
            totals[(row['leakage_group_id'], row['target'])] += row['sample_weight']
        for total in totals.values():
            self.assertAlmostEqual(total, 1.0)

    def test_balanced_domain_and_disjoint_reporting_tracks(self):
        counts = Counter(c["category"] for c in self.cases)
        self.assertEqual(len(counts), 17)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 6)
        scores = {c["case_id"]: dataset.score_case(c, []) for c in self.cases}
        groups = dataset.aggregate_by_track(self.cases, scores)
        self.assertIn("core", groups)
        self.assertIn("context", groups)
        self.assertNotIn("all", groups)
        for s in builder.SPLITS:
            selected = [c for c in self.cases if c["split"] == s]
            self.assertGreaterEqual(len({u['missing_fact_id'] for c in selected
                                        if c['corpus_status']=='NO_FAQ' for u in c['unsupported_facts']}), 8)
            self.assertEqual({c['primary_purpose'] for c in selected}, {c['primary_purpose'] for c in self.cases})
            self.assertEqual(len({c['category'] for c in selected if c['corpus_status']=='FAQ_EXISTS'}), 17)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FairDatasetTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"test_count": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "passed": result.wasSuccessful(), "cases_sha256": dataset.digest(ROOT / "data/cases.jsonl"),
              "semantics_independently_reviewed": False,
              "scope": "structural, evidence-substring, split, review-gate and metric contracts; not independent semantic approval"}
    (ROOT / "data/validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
