"""Tests for evaluation metrics: completion, compilation, duplicates, similarity."""
import tempfile
import unittest
from pathlib import Path

from tests import support

from stage1 import evaluate
from stage1 import generate
from stage1.compile_tikz import CompileResult
from stage1 import schema


def record(row_id, output, *, finish="stop", tokens=10, latency=1.0, error=None,
           index=0):
    return generate.GenerationRecord(
        row_id=row_id, source_row_index=index,
        instruction_sha256="0" * 64, raw_output=output, input_tokens=20,
        output_tokens=tokens, latency_s=latency, batch_size=1,
        finish_reason=finish, error=error)


def compile_result(status, category, *, render_path=None, message=None):
    return CompileResult(
        status=status, category=category, extraction_kind="picture",
        extraction_note="", returncode=0 if status == "success" else 1,
        duration_s=0.5, error_line=None, error_message=message, log_tail="",
        artifacts_dir=None, pdf_path=None, render_path=render_path,
        render_error=None)


class MetricTests(unittest.TestCase):
    def evaluation_config(self, **overrides):
        config = support.make_config(**({"evaluation": overrides} if overrides else {}))
        return config.evaluation

    def test_generation_extraction_and_compile_counts(self):
        records = [
            record("r1", "\\begin{tikzpicture}\\draw (0,0)--(1,1);\\end{tikzpicture}"),
            record("r2", "", finish="length", tokens=1024),
            record("r3", "no code", finish="length"),
            record("r4", "\\begin{tikzpicture}\\draw broken;\\end{tikzpicture}"),
        ]
        results = iter([
            compile_result("success", "success"),
            compile_result("no_tikz", "no_tikz"),
            compile_result("no_tikz", "no_tikz"),
            compile_result("latex_error", "latex_error",
                           message="Undefined control sequence"),
        ])
        result = evaluate.evaluate_records(
            records, evaluation=self.evaluation_config(),
            compile_fn=lambda text: next(results))
        metrics = result["metrics"]
        self.assertEqual(metrics["generation"]["examples"], 4)
        self.assertEqual(metrics["generation"]["completed"], 2)
        self.assertEqual(metrics["generation"]["truncated"], 2)
        self.assertEqual(metrics["generation"]["empty"], 1)
        self.assertEqual(metrics["extraction"]["picture"], 2)
        self.assertEqual(metrics["extraction"]["none"], 2)
        self.assertEqual(metrics["compilation"]["attempted"], 4)
        self.assertEqual(metrics["compilation"]["success"], 1)
        self.assertEqual(metrics["compilation"]["categories"]["latex_error"], 1)
        self.assertEqual(metrics["compilation"]["categories"]["no_tikz"], 2)
        self.assertEqual(len(metrics["compilation"]["first_errors"]), 1)

    def test_error_records_are_counted_and_not_compiled(self):
        records = [record("r1", "", error="boom", finish="error")]
        result = evaluate.evaluate_records(
            records, evaluation=self.evaluation_config(),
            compile_fn=lambda text: self.fail("must not compile error records"))
        metrics = result["metrics"]
        self.assertEqual(metrics["generation"]["errors"], 1)
        self.assertEqual(metrics["compilation"]["attempted"], 0)

    def test_compile_disabled_marks_category(self):
        records = [record("r1", "\\draw (0,0) -- (1,1);")]
        result = evaluate.evaluate_records(
            records, evaluation=self.evaluation_config(), compile_enabled=False)
        metrics = result["metrics"]
        self.assertEqual(metrics["compilation"]["attempted"], 0)
        self.assertEqual(result["rows"][0]["compile_category"], "disabled")

    def test_duplicate_outputs_are_detected(self):
        text = "\\begin{tikzpicture}\\draw (0,0)--(1,1);\\end{tikzpicture}"
        records = [record("r1", text), record("r2", text),
                   record("r3", text + "\n"), record("r4", "other")]
        result = evaluate.evaluate_records(
            records, evaluation=self.evaluation_config(),
            compile_fn=lambda value: compile_result("no_tikz", "no_tikz"))
        metrics = result["metrics"]
        self.assertEqual(metrics["duplicates"]["duplicate_groups"], 1)
        self.assertEqual(metrics["duplicates"]["duplicate_rows"], 2)
        self.assertEqual(metrics["duplicates"]["largest_group"], 3)

    def test_exact_memorization_is_detected(self):
        target = "\\begin{tikzpicture}\\draw (0,0)--(2,2);\\end{tikzpicture}"
        records = [record("r1", target), record("r2", "fresh output")]
        result = evaluate.evaluate_records(
            records, evaluation=self.evaluation_config(),
            compile_fn=lambda value: compile_result("no_tikz", "no_tikz"),
            train_targets=[("train-1", target), ("train-2", "unrelated")])
        metrics = result["metrics"]
        self.assertEqual(metrics["memorization"]["exact_matches"], 1)
        self.assertEqual(result["rows"][0]["memorized"], "exact")

    def test_near_memorization_is_detected(self):
        target = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
        near = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda nu"
        records = [record("r1", near)]
        evaluation = support.make_config(
            evaluation={"duplicate_ngram_size": 3,
                        "duplicate_overlap_threshold": 0.5}).evaluation
        result = evaluate.evaluate_records(
            records, evaluation=evaluation,
            compile_fn=lambda value: compile_result("no_tikz", "no_tikz"),
            train_targets=[("train-1", target)])
        metrics = result["metrics"]
        self.assertEqual(metrics["memorization"]["near_matches"], 1)
        self.assertGreater(result["rows"][0]["near_duplicate_overlap"], 0.5)

    def test_latency_and_length_statistics(self):
        records = [record("r1", "a", tokens=5, latency=1.0),
                   record("r2", "bb", tokens=15, latency=3.0)]
        result = evaluate.evaluate_records(
            records, evaluation=self.evaluation_config(),
            compile_fn=lambda value: compile_result("no_tikz", "no_tikz"))
        metrics = result["metrics"]
        self.assertEqual(metrics["latency"]["mean"], 2.0)
        self.assertEqual(metrics["latency"]["tokens_per_second"], 5.0)
        self.assertEqual(metrics["output_length"]["tokens"]["total"], 20)

    def test_metrics_match_schema(self):
        records = [record("r1", "\\draw (0,0) -- (1,1);")]
        result = evaluate.evaluate_records(
            records, evaluation=self.evaluation_config(),
            compile_fn=lambda value: compile_result("no_tikz", "no_tikz"))
        metrics = result["metrics"]
        metrics.update({"name": "x", "source_kind": "base", "split": "test"})
        schema.validate_artifact(metrics, schema.load_schema("eval.schema.json"),
                                 "eval metrics")


@support.requires_pillow
class SimilarityTests(unittest.TestCase):
    def test_identical_images_score_high(self):
        with tempfile.TemporaryDirectory() as directory:
            render = Path(directory) / "render.png"
            render.write_bytes(support.make_png_bytes(seed=3))
            result = evaluate.image_similarity(
                render, support.make_png_bytes(seed=3), size=16)
            self.assertEqual(result.status, "ok")
            self.assertGreater(result.similarity, 0.99)

    def test_different_images_score_lower(self):
        with tempfile.TemporaryDirectory() as directory:
            render = Path(directory) / "render.png"
            render.write_bytes(support.make_png_bytes(seed=0))
            result = evaluate.image_similarity(
                render, support.make_png_bytes(seed=7), size=16)
            self.assertEqual(result.status, "ok")
            self.assertLess(result.similarity, 0.9)

    def test_similarity_flows_into_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            render = Path(directory) / "render.png"
            render.write_bytes(support.make_png_bytes(seed=1))
            records = [record("r1", "\\draw (0,0) -- (1,1);")]
            evaluation = support.make_config().evaluation
            result = evaluate.evaluate_records(
                records, evaluation=evaluation,
                compile_fn=lambda value: compile_result(
                    "success", "success", render_path=str(render)),
                reference_images={"r1": support.make_png_bytes(seed=1)})
            metrics = result["metrics"]
            self.assertEqual(metrics["render_similarity"]["status"], "ok")
            self.assertEqual(metrics["render_similarity"]["compared"], 1)
            self.assertGreater(metrics["render_similarity"]["mean"], 0.99)

    def test_missing_reference_is_reported_not_silent(self):
        with tempfile.TemporaryDirectory() as directory:
            render = Path(directory) / "render.png"
            render.write_bytes(support.make_png_bytes(seed=1))
            records = [record("r1", "\\draw (0,0) -- (1,1);")]
            result = evaluate.evaluate_records(
                records, evaluation=support.make_config().evaluation,
                compile_fn=lambda value: compile_result(
                    "success", "success", render_path=str(render)))
            metrics = result["metrics"]
            self.assertEqual(metrics["render_similarity"]["status"], "not_available")
            self.assertIn("no reference image",
                          metrics["render_similarity"]["reasons"])


@support.requires_pyarrow
class EvaluationSelectionTests(unittest.TestCase):
    """Quarantined rows must never be evaluated or sampled for memorization."""

    def test_quarantined_rows_are_never_selected(self):
        from stage1 import data as data_module
        with tempfile.TemporaryDirectory() as directory:
            export, prepared, report, eligibility = (
                support.mixed_quarantine_export_and_dir(directory))
            manifest = data_module.load_split_manifest(prepared)
            quarantined = set(eligibility["quarantined"])
            self.assertTrue(quarantined)
            for split in ("validation", "test"):
                ids = evaluate.select_evaluation_ids(
                    manifest, split, limit=None, seed=0, eligibility=eligibility)
                self.assertFalse(set(ids) & quarantined, split)
            sampled = evaluate.select_memorization_ids(
                manifest, limit=None, seed=0, eligibility=eligibility)
            self.assertFalse(set(sampled) & quarantined)
            # Eligible rows are still reachable.
            selected_total = len(evaluate.select_memorization_ids(
                manifest, limit=None, seed=0, eligibility=eligibility))
            self.assertEqual(selected_total,
                             eligibility["by_split"]["train"]["eligible"])


class NormalizationTests(unittest.TestCase):
    def test_normalize_text_collapses_whitespace(self):
        self.assertEqual(evaluate.normalize_text(" a\n b\tc "), "a b c")

    def test_ngram_hashes_short_text(self):
        self.assertEqual(len(evaluate.ngram_hashes("one two", 5)), 1)
        self.assertEqual(evaluate.ngram_hashes("", 5), set())

    def test_near_duplicate_index_no_overlap(self):
        index = evaluate.NearDuplicateIndex([("t1", "alpha beta gamma")], 2)
        overlap, match = index.query("completely different words")
        self.assertEqual(overlap, 0.0)
        self.assertIsNone(match)


if __name__ == "__main__":
    unittest.main()
