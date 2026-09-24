"""Tests for the machine-readable comparison and readable report."""
import tempfile
import unittest
from pathlib import Path

from stage1 import report


TRAINING_METRICS = {
    "schema_version": "stage1-training-metrics-v1",
    "gate": "smoke-1000",
    "model_id": "openbmb/MiniCPM5-2B",
    "optimizer_steps": 50,
    "supervised_tokens_seen": 120000,
    "final_eval_loss": 0.42,
    "training": {
        "global_step": 50,
        "supervised_tokens_seen": 120000,
        "final_eval_loss": 0.42,
        "model_id": "openbmb/MiniCPM5-2B",
    },
}

EVAL_METRICS = {
    "schema_version": "stage1-eval-v1",
    "name": "sft-test",
    "source_kind": "checkpoint",
    "training_method": "full",
    "model_id": "openbmb/MiniCPM5-2B",
    "data_identity_sha256": "1" * 64,
    "max_seq_len": 4096,
    "evaluation_set_sha256": "2" * 64,
    "generation": {"examples": 100, "completed": 80, "truncated": 20,
                   "errors": 0, "empty": 2},
    "compilation": {"attempted": 100, "success": 75, "success_rate": 0.75,
                    "categories": {"latex_error": 25}},
    "latency": {"mean": 1.5, "tokens_per_second": 200.0},
    "duplicates": {"duplicate_rows": 3},
    "memorization": {"exact_matches": 1, "near_matches": 2},
}


class SummarizeTests(unittest.TestCase):
    def test_training_metrics_summary(self):
        row = report.summarize_metrics("minicpm-smoke", TRAINING_METRICS)
        self.assertEqual(row["label"], "minicpm-smoke")
        self.assertEqual(row["supervised_tokens"], 120000)
        self.assertEqual(row["optimizer_steps"], 50)
        self.assertEqual(row["eval_loss"], 0.42)
        self.assertIsNone(row["compile_success_rate"])

    def test_eval_metrics_summary(self):
        row = report.summarize_metrics("minicpm-eval", EVAL_METRICS)
        self.assertEqual(row["examples"], 100)
        self.assertEqual(row["completion_rate"], 0.8)
        self.assertEqual(row["compile_success_rate"], 0.75)
        self.assertEqual(row["mean_latency_s"], 1.5)
        self.assertEqual(row["memorized"], 3)
        self.assertIsNone(row["supervised_tokens"])

    def test_missing_fields_stay_none(self):
        row = report.summarize_metrics("empty", {})
        self.assertIsNone(row["eval_loss"])
        self.assertIsNone(row["compile_success_rate"])


class ComparisonTests(unittest.TestCase):
    def test_comparison_rows_and_markdown(self):
        comparison = report.compare_metrics([
            ("base", dict(EVAL_METRICS, artifact={"kind": "base"})),
            ("trained", dict(EVAL_METRICS, artifact={"kind": "final"})),
            ("smoke", TRAINING_METRICS),
        ])
        self.assertEqual(len(comparison["rows"]), 3)
        markdown = report.render_comparison(comparison)
        self.assertIn("| run |", markdown)
        self.assertIn("base", markdown)
        self.assertIn("120000", markdown)
        self.assertIn("supervised tokens", markdown)

    def test_comparison_is_structurally_sound(self):
        comparison = report.compare_metrics([("base", EVAL_METRICS)])
        self.assertEqual(comparison["schema_version"], "stage1-comparison-v1")
        row = comparison["rows"][0]
        for key in ("label", "supervised_tokens", "compile_success_rate",
                    "data_identity_sha256"):
            self.assertIn(key, row)

    def test_experiment_readiness_requires_comparable_artifacts(self):
        base = dict(EVAL_METRICS, artifact={"kind": "base"},
                    data_identity_sha256="1" * 64, split="test")
        trained = dict(EVAL_METRICS, artifact={"kind": "final"},
                       data_identity_sha256="1" * 64, split="test")
        only_trained = report.experiment_readiness([("trained", trained)])
        self.assertFalse(only_trained["ready"])
        self.assertIn("no base-model evaluation artifact", only_trained["problems"])
        ready = report.experiment_readiness([("base", base), ("trained", trained)])
        self.assertTrue(ready["ready"])
        self.assertEqual(ready["comparable_pairs"], [["base", "trained"]])

    def test_experiment_readiness_rejects_mismatched_identity_or_split(self):
        base = dict(EVAL_METRICS, artifact={"kind": "base"},
                    data_identity_sha256="1" * 64, split="test")
        trained = dict(EVAL_METRICS, artifact={"kind": "final"},
                       data_identity_sha256="2" * 64, split="test")
        readiness = report.experiment_readiness([("base", base), ("trained", trained)])
        self.assertFalse(readiness["ready"])
        self.assertIn("do not share a data identity", readiness["problems"][0])
        trained = dict(trained, data_identity_sha256="1" * 64, split="validation")
        readiness = report.experiment_readiness([("base", base), ("trained", trained)])
        self.assertFalse(readiness["ready"])

    def test_comparability_refuses_mixed_training_methods(self):
        base = dict(EVAL_METRICS, artifact={"kind": "base"},
                    training_method="base")
        full = dict(EVAL_METRICS, artifact={"kind": "final"},
                    training_method="full")
        lora = dict(EVAL_METRICS, artifact={"kind": "final"},
                    training_method="lora")
        problems = report.comparison_problems([("base", base), ("lora", lora)])
        self.assertEqual(problems, [])
        problems = report.comparison_problems([("full", full), ("lora", lora)])
        self.assertTrue(any("mixes training methods" in item for item in problems))
        readiness = report.experiment_readiness(
            [("base", base), ("full", full), ("lora", lora)])
        self.assertFalse(readiness["ready"])
        # Non-comparable is allowed only for an otherwise complete set.
        allowed = report.experiment_readiness(
            [("base", base), ("full", full), ("lora", lora)],
            allow_non_comparable=True)
        self.assertTrue(allowed["ready"])
        self.assertTrue(allowed["comparability_allowed"])

    def test_comparability_refuses_different_sequence_limit_or_eval_set(self):
        first = dict(EVAL_METRICS, artifact={"kind": "final"},
                     evaluation_set_sha256="2" * 64, max_seq_len=4096)
        second = dict(EVAL_METRICS, artifact={"kind": "final"},
                      evaluation_set_sha256="3" * 64, max_seq_len=4096)
        problems = report.comparison_problems([("a", first), ("b", second)])
        self.assertTrue(any("evaluation set" in item for item in problems))
        second = dict(EVAL_METRICS, artifact={"kind": "final"},
                      evaluation_set_sha256="2" * 64, max_seq_len=8192)
        problems = report.comparison_problems([("a", first), ("b", second)])
        self.assertTrue(any("sequence limit" in item for item in problems))

    def test_comparison_report_shows_method_and_comparability(self):
        base = dict(EVAL_METRICS, artifact={"kind": "base"},
                    training_method="base")
        lora = dict(EVAL_METRICS, artifact={"kind": "final"},
                    training_method="lora")
        comparison = report.compare_metrics([("base", base), ("lora", lora)])
        markdown = report.render_comparison(comparison)
        self.assertIn("method", markdown)
        self.assertIn("lora", markdown)
        self.assertIn("Comparability: OK", markdown)

    def test_training_metrics_expose_exact_token_accounting(self):
        metrics = dict(TRAINING_METRICS)
        metrics["supervised_tokens"] = {
            "supervised_tokens_committed": 120000,
            "supervised_tokens_exact": True,
        }
        row = report.summarize_metrics("smoke", metrics)
        self.assertEqual(row["supervised_tokens"], 120000)
        self.assertTrue(row["supervised_tokens_exact"])

    def test_write_comparison_writes_both_files(self):
        with tempfile.TemporaryDirectory() as directory:
            comparison = report.compare_metrics([("base", EVAL_METRICS)])
            written = report.write_comparison(Path(directory), comparison)
            self.assertTrue(Path(written["json"]).is_file())
            self.assertTrue(Path(written["markdown"]).is_file())
            loaded = report.load_metrics_file(written["json"])
            self.assertEqual(loaded["rows"][0]["label"], "base")

    def test_load_missing_metrics_file(self):
        with self.assertRaises(Exception):
            report.load_metrics_file("/nonexistent/metrics.json")


if __name__ == "__main__":
    unittest.main()
