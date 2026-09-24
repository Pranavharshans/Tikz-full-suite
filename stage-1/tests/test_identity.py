"""Tests for run identities, resume refusal and the explicit gate chain."""
import json
import tempfile
import unittest
from pathlib import Path

from tests import support

from stage1 import identity
from stage1.errors import GateError, IdentityMismatch

DATA_IDENTITY = {
    "schema_version": "stage1-prepared-v1",
    "dataset_logical_sha256": "1" * 64,
    "split_manifest_sha256": "2" * 64,
    "seed": 7,
    "row_count": 100,
    "complete_rows": 100,
    "source_dataset": "nllg/DaTikZ-V4",
    "source_revision": "b" * 40,
    "sha256": "3" * 64,
}
FINGERPRINT = {
    "model_id": "test/Model-A",
    "revision": "a" * 40,
    "adapter": "minicpm5",
    "tokenizer_class": "FakeTokenizer",
    "vocab_size": 512,
    "pad_token_id": 0,
    "chat_template_sha256": "4" * 64,
    "chat_template_source": "tokenizer",
    "files": {},
}
DEPENDENCIES = {"torch": "2.12.1", "python": "3.12.0"}


def build_identity(config=None, **overrides):
    config = config or support.make_config()
    kwargs = dict(
        data_identity=DATA_IDENTITY, config=config,
        model_fingerprint=FINGERPRINT, dependencies=DEPENDENCIES,
        code_sha256="5" * 64, repo_commit="6" * 40,
        dataset_report_sha256="a" * 64)
    kwargs.update(overrides)
    return identity.build_run_identity(**kwargs)


class IdentityTests(unittest.TestCase):
    def test_identity_is_deterministic(self):
        self.assertEqual(build_identity()["sha256"], build_identity()["sha256"])

    def test_seed_changes_identity(self):
        self.assertNotEqual(
            build_identity()["sha256"],
            build_identity(support.make_config(seed=1))["sha256"])

    def test_dependency_versions_change_identity(self):
        self.assertNotEqual(
            build_identity()["sha256"],
            build_identity(dependencies={"torch": "9.9.9", "python": "3.12.0"})["sha256"])

    def test_code_hash_and_commit_change_identity(self):
        self.assertNotEqual(build_identity()["sha256"],
                            build_identity(code_sha256="7" * 64)["sha256"])
        self.assertNotEqual(build_identity()["sha256"],
                            build_identity(repo_commit="8" * 40)["sha256"])

    def test_template_source_is_not_identity_bearing_but_template_hash_is(self):
        first = build_identity()
        fingerprint = dict(FINGERPRINT, chat_template_source="/some/path")
        self.assertEqual(build_identity(model_fingerprint=fingerprint)["sha256"],
                         first["sha256"])
        fingerprint = dict(FINGERPRINT, chat_template_sha256="9" * 64)
        self.assertNotEqual(build_identity(model_fingerprint=fingerprint)["sha256"],
                            first["sha256"])

    def test_identity_requires_data_identity_sha(self):
        with self.assertRaises(Exception):
            build_identity(data_identity={"dataset_logical_sha256": "1" * 64})

    def test_dataset_report_hash_changes_identity(self):
        first = build_identity()
        second = build_identity(dataset_report_sha256="b" * 64)
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["data"]["dataset_report_sha256"], "a" * 64)
        self.assertEqual(second["data"]["dataset_report_sha256"], "b" * 64)

    def test_missing_dataset_report_hash_is_recorded_as_null(self):
        identity_payload = build_identity(dataset_report_sha256=None)
        self.assertIsNone(identity_payload["data"]["dataset_report_sha256"])


class IdentityDiffTests(unittest.TestCase):
    def test_diff_reports_nested_paths(self):
        left = {"a": {"b": 1}, "c": 2}
        right = {"a": {"b": 3}, "c": 2, "d": 4}
        differences = identity.identity_diff(left, right)
        self.assertTrue(any("a.b" in line for line in differences))
        self.assertTrue(any("d" in line for line in differences))

    def test_diff_ignores_equal(self):
        self.assertEqual(identity.identity_diff({"a": 1}, {"a": 1}), [])


class RunRecordTests(unittest.TestCase):
    def test_run_record_written_once_and_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            run_identity = build_identity()
            record = identity.ensure_run_record(run_dir, run_identity)
            self.assertEqual(record["identity_sha256"], run_identity["sha256"])
            self.assertTrue(identity.run_record_path(run_dir).is_file())
            again = identity.ensure_run_record(run_dir, run_identity)
            self.assertEqual(again["identity_sha256"], run_identity["sha256"])

    def test_resume_with_different_identity_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            identity.ensure_run_record(run_dir, build_identity())
            changed = build_identity(support.make_config(seed=99))
            with self.assertRaises(IdentityMismatch) as context:
                identity.ensure_run_record(run_dir, changed)
            message = str(context.exception)
            self.assertIn("seed", message)

    def test_orphan_artifacts_without_run_json_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "checkpoints").mkdir(parents=True)
            (run_dir / "checkpoints" / "checkpoint-10").mkdir()
            with self.assertRaises(IdentityMismatch):
                identity.ensure_run_record(run_dir, build_identity())

    def test_run_record_matches_schema(self):
        from stage1 import schema
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            identity.ensure_run_record(run_dir, build_identity())
            schema.validate_artifact(
                json.loads(identity.run_record_path(run_dir).read_text()),
                schema.load_schema("run.schema.json"), "run record")


class GateChainTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name)
        self.identity_sha = build_identity()["sha256"]

    def tearDown(self):
        self.temporary.cleanup()

    def write_preflight(self, status="passed", complete=True, sha=None):
        identity.write_preflight_report(
            self.run_dir, identity_sha256=sha or self.identity_sha,
            checks=[{"name": "gpu.identity", "status": "pass", "detail": "ok"}],
            status=status, complete=complete, payload={})

    def test_overfit_requires_preflight(self):
        with self.assertRaises(GateError) as context:
            identity.require_prerequisites(self.run_dir, "overfit-100",
                                           self.identity_sha)
        self.assertIn("preflight: not run", str(context.exception))

    def test_smoke_requires_overfit_after_preflight(self):
        self.write_preflight()
        identity.require_prerequisites(self.run_dir, "overfit-100",
                                       self.identity_sha)
        with self.assertRaises(GateError) as context:
            identity.require_prerequisites(self.run_dir, "smoke-1000",
                                           self.identity_sha)
        self.assertIn("overfit-100: not run", str(context.exception))

    def test_full_requires_smoke(self):
        self.write_preflight()
        identity.write_gate_evidence(
            self.run_dir, "overfit-100", status="passed",
            identity_sha256=self.identity_sha, payload={})
        identity.write_gate_evidence(
            self.run_dir, "smoke-1000", status="passed",
            identity_sha256=self.identity_sha, payload={})
        identity.require_prerequisites(self.run_dir, "full", self.identity_sha)

    def test_failed_gate_blocks_next_gate(self):
        self.write_preflight()
        identity.write_gate_evidence(
            self.run_dir, "overfit-100", status="failed",
            identity_sha256=self.identity_sha, payload={"reason": "loss"})
        with self.assertRaises(GateError) as context:
            identity.require_prerequisites(self.run_dir, "smoke-1000",
                                           self.identity_sha)
        self.assertIn("status failed", str(context.exception))

    def test_incomplete_preflight_blocks_gates(self):
        self.write_preflight(complete=False)
        with self.assertRaises(GateError) as context:
            identity.require_prerequisites(self.run_dir, "overfit-100",
                                           self.identity_sha)
        self.assertIn("incomplete", str(context.exception))

    def test_foreign_preflight_identity_blocks(self):
        self.write_preflight(sha="f" * 64)
        with self.assertRaises(GateError) as context:
            identity.require_prerequisites(self.run_dir, "overfit-100",
                                           self.identity_sha)
        self.assertIn("identity mismatch", str(context.exception))

    def test_passed_gate_refuses_silent_rerun(self):
        identity.write_gate_evidence(
            self.run_dir, "overfit-100", status="passed",
            identity_sha256=self.identity_sha, payload={})
        with self.assertRaises(GateError) as context:
            identity.require_gate_start(self.run_dir, "overfit-100",
                                        self.identity_sha)
        self.assertIn("--rerun-gate", str(context.exception))
        evidence = identity.require_gate_start(
            self.run_dir, "overfit-100", self.identity_sha, allow_rerun=True)
        self.assertEqual(evidence["status"], "passed")

    def test_failed_gate_can_rerun(self):
        identity.write_gate_evidence(
            self.run_dir, "overfit-100", status="failed",
            identity_sha256=self.identity_sha, payload={})
        evidence = identity.require_gate_start(self.run_dir, "overfit-100",
                                               self.identity_sha)
        self.assertEqual(evidence["status"], "failed")

    def test_gate_evidence_foreign_identity_refuses(self):
        identity.write_gate_evidence(
            self.run_dir, "overfit-100", status="passed",
            identity_sha256="f" * 64, payload={})
        with self.assertRaises(IdentityMismatch):
            identity.require_gate_start(self.run_dir, "overfit-100",
                                        self.identity_sha)

    def test_gate_evidence_matches_schema(self):
        from stage1 import schema
        identity.write_gate_evidence(
            self.run_dir, "overfit-100", status="passed",
            identity_sha256=self.identity_sha,
            payload={"reason": "ok", "supervised_tokens_seen": 10,
                     "optimizer_steps": 1, "final_eval_loss": 0.01})
        schema.validate_artifact(
            json.loads(identity.gate_evidence_path(self.run_dir, "overfit-100").read_text()),
            schema.load_schema("gate-evidence.schema.json"), "gate evidence")


if __name__ == "__main__":
    unittest.main()
