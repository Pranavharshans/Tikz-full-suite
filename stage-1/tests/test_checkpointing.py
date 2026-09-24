"""Regression tests for gate isolation, provenance and resume verification."""
import functools
import json
import math
import tempfile
import unittest
from pathlib import Path

from tests import support

from stage1 import checkpointing
from stage1.errors import CheckpointError

IDENTITY = "a" * 64
MODEL = dict(model_id="test/Model-A", model_revision="a" * 40, adapter="minicpm5")


def make_gate_checkpoint(run_dir, gate, step, **overrides):
    values = {"kind": "checkpoint", "identity_sha256": IDENTITY, "gate": gate,
              "global_step": step, **MODEL}
    values.update(overrides)
    return support.write_artifact(
        Path(run_dir) / "checkpoints" / gate / f"checkpoint-{step}", **values)


class GateNamespaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_overfit_checkpoint_is_invisible_to_smoke_and_full(self):
        make_gate_checkpoint(self.run_dir, "overfit-100", 200)
        for gate in ("smoke-1000", "full"):
            path, meta = checkpointing.find_latest_checkpoint(
                self.run_dir, gate, IDENTITY, **MODEL)
            self.assertIsNone(path, f"{gate} must not see the overfit checkpoint")
            self.assertEqual(meta, {})
            self.assertEqual(checkpointing.list_checkpoints(self.run_dir, gate), [])

    def test_smoke_checkpoint_is_invisible_to_full(self):
        make_gate_checkpoint(self.run_dir, "smoke-1000", 50)
        path, _ = checkpointing.find_latest_checkpoint(
            self.run_dir, "full", IDENTITY, **MODEL)
        self.assertIsNone(path)

    def test_same_gate_checkpoint_is_resumed(self):
        make_gate_checkpoint(self.run_dir, "smoke-1000", 50)
        make_gate_checkpoint(self.run_dir, "smoke-1000", 100)
        path, meta = checkpointing.find_latest_checkpoint(
            self.run_dir, "smoke-1000", IDENTITY, **MODEL)
        self.assertEqual(path.name, "checkpoint-100")
        self.assertEqual(meta["gate"], "smoke-1000")

    def test_copied_checkpoint_into_wrong_namespace_is_refused(self):
        make_gate_checkpoint(self.run_dir, "overfit-100", 200)
        source = self.run_dir / "checkpoints" / "overfit-100" / "checkpoint-200"
        target = self.run_dir / "checkpoints" / "smoke-1000" / "checkpoint-200"
        target.parent.mkdir(parents=True)
        target.mkdir()
        for item in source.iterdir():
            target.joinpath(item.name).write_bytes(item.read_bytes())
        with self.assertRaisesRegex(CheckpointError, "gate"):
            checkpointing.find_latest_checkpoint(
                self.run_dir, "smoke-1000", IDENTITY, **MODEL)

    def test_checkpoint_meta_records_gate_and_model(self):
        path = make_gate_checkpoint(self.run_dir, "full", 7)
        meta = json.loads(
            (path / checkpointing.CHECKPOINT_META_NAME).read_text())
        self.assertEqual(meta["gate"], "full")
        self.assertEqual(meta["kind"], "checkpoint")
        self.assertEqual(meta["model_id"], "test/Model-A")
        self.assertEqual(meta["model_revision"], "a" * 40)
        self.assertEqual(meta["adapter"], "minicpm5")

    def test_final_artifacts_are_namespaced_per_gate(self):
        for gate in ("overfit-100", "smoke-1000", "full"):
            directory = checkpointing.final_dir(self.run_dir, gate)
            self.assertEqual(directory, self.run_dir / "final" / gate)
            support.write_artifact(directory, kind="final", gate=gate,
                                   identity_sha256=IDENTITY, **MODEL)
            meta = checkpointing.verify_final_artifact(
                directory, identity_sha256=IDENTITY, gate=gate, **MODEL)
            self.assertEqual(meta["kind"], "final")
            self.assertEqual(meta["gate"], gate)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_foreign_identity_is_refused(self):
        path = make_gate_checkpoint(self.run_dir, "smoke-1000", 10,
                                    identity_sha256="b" * 64)
        with self.assertRaisesRegex(CheckpointError, "identity_sha256"):
            checkpointing.verify_checkpoint(path, IDENTITY, gate="smoke-1000",
                                            **MODEL)

    def test_wrong_model_is_refused(self):
        path = make_gate_checkpoint(self.run_dir, "smoke-1000", 10,
                                    model_id="other/Model-B")
        with self.assertRaisesRegex(CheckpointError, "model_id"):
            checkpointing.verify_checkpoint(path, IDENTITY, gate="smoke-1000",
                                            **MODEL)

    def test_wrong_revision_is_refused(self):
        path = make_gate_checkpoint(self.run_dir, "smoke-1000", 10,
                                    model_revision="c" * 40)
        with self.assertRaisesRegex(CheckpointError, "model_revision"):
            checkpointing.verify_checkpoint(path, IDENTITY, gate="smoke-1000",
                                            **MODEL)

    def test_wrong_adapter_is_refused(self):
        path = make_gate_checkpoint(self.run_dir, "smoke-1000", 10,
                                    adapter="qwen3.5")
        with self.assertRaisesRegex(CheckpointError, "adapter"):
            checkpointing.verify_checkpoint(path, IDENTITY, gate="smoke-1000",
                                            **MODEL)

    def test_missing_metadata_is_refused(self):
        path = make_gate_checkpoint(self.run_dir, "smoke-1000", 10, meta=False)
        with self.assertRaisesRegex(CheckpointError, "no Stage 1 artifact metadata"):
            checkpointing.verify_checkpoint(path, IDENTITY, gate="smoke-1000",
                                            **MODEL)

    def test_incomplete_checkpoint_is_refused(self):
        path = make_gate_checkpoint(self.run_dir, "smoke-1000", 10, complete=False)
        with self.assertRaisesRegex(CheckpointError, "incomplete"):
            checkpointing.verify_checkpoint(path, IDENTITY, gate="smoke-1000",
                                            **MODEL)

    def test_evaluation_artifact_rejects_wrong_gate(self):
        path = make_gate_checkpoint(self.run_dir, "smoke-1000", 10)
        with self.assertRaisesRegex(CheckpointError, "gate"):
            checkpointing.verify_evaluation_artifact(
                path, identity_sha256=IDENTITY, gate="full", **MODEL)

    def test_evaluation_artifact_accepts_checkpoint_and_final(self):
        checkpoint = make_gate_checkpoint(self.run_dir, "smoke-1000", 10)
        final = support.write_artifact(
            self.run_dir / "final" / "smoke-1000", kind="final",
            gate="smoke-1000", identity_sha256=IDENTITY, **MODEL)
        for path in (checkpoint, final):
            meta = checkpointing.verify_evaluation_artifact(
                path, identity_sha256=IDENTITY, **MODEL)
            self.assertIn(meta["kind"], ("checkpoint", "final"))

    def test_evaluation_artifact_rejects_arbitrary_directory(self):
        directory = self.run_dir / "not-an-artifact"
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        with self.assertRaisesRegex(CheckpointError, "no Stage 1 artifact metadata"):
            checkpointing.verify_evaluation_artifact(
                directory, identity_sha256=IDENTITY, **MODEL)

    def test_evaluation_artifact_rejects_cross_model_final(self):
        final = support.write_artifact(
            self.run_dir / "final" / "smoke-1000", kind="final",
            gate="smoke-1000", identity_sha256=IDENTITY,
            model_id="other/Model-B")
        with self.assertRaisesRegex(CheckpointError, "model_id"):
            checkpointing.verify_evaluation_artifact(
                final, identity_sha256=IDENTITY, **MODEL)

    def test_artifact_meta_matches_schema(self):
        from stage1 import schema
        path = make_gate_checkpoint(self.run_dir, "smoke-1000", 10)
        meta = json.loads(
            (path / checkpointing.CHECKPOINT_META_NAME).read_text())
        schema.validate_artifact(meta, schema.load_schema("artifact.schema.json"),
                                 "artifact meta")


def fake_cosine_lambda(step, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """A stand-in for Transformers' cosine lambda (same shape, pure Python)."""
    if step < num_warmup_steps:
        return step / max(1, num_warmup_steps)
    progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))


class SchedulerRestoreTests(unittest.TestCase):
    """Production scheduler states are LambdaLR (get_scheduler("cosine"))."""

    def production_state(self, *, num_warmup_steps=2, num_training_steps=10,
                         base_lrs=(1e-5,), last_epoch=3, step_count=4):
        return {
            "base_lrs": list(base_lrs),
            "last_epoch": last_epoch,
            "_step_count": step_count,
            "_last_lr": list(base_lrs),
            "lr_lambdas": [functools.partial(
                fake_cosine_lambda, num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps)],
        }

    def test_lambda_parameters_are_extracted_from_partials(self):
        parameters = checkpointing.lambda_schedule_parameters(
            self.production_state())
        self.assertEqual(parameters["num_warmup_steps"], 2)
        self.assertEqual(parameters["num_training_steps"], 10)

    def test_lambda_parameters_are_none_when_not_introspectable(self):
        state = self.production_state()
        state["lr_lambdas"] = [lambda step: 1.0]
        self.assertIsNone(checkpointing.lambda_schedule_parameters(state))
        self.assertIsNone(checkpointing.lambda_schedule_parameters({}))

    def test_restore_plan_accepts_the_production_schedule(self):
        plan = checkpointing.scheduler_restore_plan(
            self.production_state(), num_training_steps=10, num_warmup_steps=2)
        self.assertEqual(plan["scheduler"], "cosine")
        self.assertTrue(plan["schedule_parameters_verified"])

    def test_restore_plan_refuses_a_cosine_annealing_state(self):
        # Regression: production is LambdaLR; a CosineAnnealingLR state has
        # T_max and must not be silently accepted.
        with self.assertRaisesRegex(CheckpointError, "not a Transformers cosine"):
            checkpointing.scheduler_restore_plan(
                {"T_max": 10, "base_lrs": [1e-5], "last_epoch": 1},
                num_training_steps=10, num_warmup_steps=2)

    def test_restore_plan_refuses_mismatched_schedule_parameters(self):
        state = self.production_state(num_warmup_steps=2, num_training_steps=10)
        with self.assertRaisesRegex(CheckpointError, "num_training_steps"):
            checkpointing.scheduler_restore_plan(
                state, num_training_steps=20, num_warmup_steps=2)
        with self.assertRaisesRegex(CheckpointError, "num_warmup_steps"):
            checkpointing.scheduler_restore_plan(
                state, num_training_steps=10, num_warmup_steps=5)

    def test_restore_plan_validates_arguments(self):
        state = self.production_state()
        with self.assertRaises(CheckpointError):
            checkpointing.scheduler_restore_plan(
                state, num_training_steps=0, num_warmup_steps=0)
        with self.assertRaises(CheckpointError):
            checkpointing.scheduler_restore_plan(
                state, num_training_steps=10, num_warmup_steps=-1)

    def test_initial_lr_is_compared_before_loading(self):
        state = self.production_state(base_lrs=(1e-5,))
        report = checkpointing.check_saved_initial_lr(state, 1e-5)
        self.assertTrue(report["initial_lr_verified"])
        with self.assertRaisesRegex(CheckpointError, "base_lrs"):
            checkpointing.check_saved_initial_lr(state, 1e-4)

    def test_missing_base_lrs_is_reported_as_unverified(self):
        report = checkpointing.check_saved_initial_lr({}, 1e-5)
        self.assertFalse(report["initial_lr_verified"])

    def test_verify_scheduler_state_probes_the_reconstruction(self):
        saved = self.production_state()
        fresh = self.production_state()
        report = checkpointing.verify_scheduler_state(
            saved, dict(saved), fresh_state=fresh,
            expected_base_lrs=[1e-5], expected_warmup_steps=2,
            expected_training_steps=10)
        self.assertTrue(report["lambda_schedule_verified"])
        self.assertEqual(report["kind"], "lambda")
        self.assertEqual(report["last_epoch"], 3)

    def test_verify_scheduler_state_detects_a_different_schedule(self):
        saved = self.production_state(num_training_steps=10)
        fresh = self.production_state(num_training_steps=40)
        with self.assertRaisesRegex(CheckpointError, "lambda schedule values"):
            checkpointing.verify_scheduler_state(
                saved, dict(saved), fresh_state=fresh,
                expected_training_steps=10, expected_warmup_steps=2)

    def test_verify_scheduler_state_detects_field_mismatch(self):
        saved = self.production_state()
        restored = dict(saved, last_epoch=2)
        with self.assertRaisesRegex(CheckpointError, "last_epoch"):
            checkpointing.verify_scheduler_state(saved, restored)

    def test_verify_scheduler_state_does_not_claim_unverifiable_lambdas(self):
        # Non-callable lambdas cannot be probed and carry no keywords: the
        # report must say so instead of claiming verification.
        saved = self.production_state()
        saved["lr_lambdas"] = ["not-callable"]
        report = checkpointing.verify_scheduler_state(
            saved, dict(saved), fresh_state=dict(saved),
            expected_training_steps=10, expected_warmup_steps=2)
        self.assertFalse(report["lambda_schedule_verified"])
        self.assertFalse(report["schedule_parameters_verified"])
        self.assertFalse(report["lambda_probes"]["compared"])


@support.requires_transformers
@support.requires_torch
class ProductionSchedulerIntegrationTests(unittest.TestCase):
    """The real Transformers cosine schedule (LambdaLR), dependency-gated.

    These run only where torch and transformers are installed; the gate stays
    explicitly unverified in dependency-free environments.
    """

    def setUp(self):
        import torch
        from transformers import get_scheduler

        self.torch = torch
        self.get_scheduler = get_scheduler
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name)

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(4))

        self.model = Tiny()

    def tearDown(self):
        self.temporary.cleanup()

    def write_checkpoint(self, *, num_warmup_steps=2, num_training_steps=10,
                         learning_rate=1e-5, scheduler=None):
        torch = self.torch
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.model.weight.sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler = scheduler or self.get_scheduler(
            "cosine", optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps)
        for _ in range(3):
            scheduler.step()
        directory = self.run_dir / "checkpoints" / "smoke-1000" / "checkpoint-1"
        directory.mkdir(parents=True)
        torch.save(optimizer.state_dict(), directory / "optimizer.pt")
        torch.save(scheduler.state_dict(), directory / "scheduler.pt")
        (directory / "trainer_state.json").write_text(json.dumps(
            {"global_step": 1, "epoch": 0.1}))
        return directory, optimizer, scheduler

    def verify(self, directory, **overrides):
        values = dict(torch=self.torch, model=self.model, learning_rate=1e-5,
                      fused=False, num_training_steps=10, num_warmup_steps=2,
                      device="cpu")
        values.update(overrides)
        return checkpointing.verify_resume_state(directory, **values)

    def test_real_transformers_cosine_schedule_round_trip(self):
        directory, optimizer, scheduler = self.write_checkpoint()
        report = self.verify(directory,
                             expected_optimizer_state=optimizer.state_dict())
        self.assertTrue(report["verified"])
        self.assertEqual(report["scheduler"]["kind"], "lambda")
        self.assertTrue(report["scheduler"]["lambda_schedule_verified"])
        self.assertEqual(report["scheduler"]["last_epoch"],
                         scheduler.last_epoch)
        self.assertEqual(report["scheduler_plan"]["num_training_steps"], 10)
        self.assertEqual(report["scheduler_plan"]["num_warmup_steps"], 2)
        self.assertTrue(report["initial_lr"]["initial_lr_verified"])

    def test_wrong_training_steps_are_refused(self):
        directory, _, _ = self.write_checkpoint(num_training_steps=10)
        with self.assertRaisesRegex(CheckpointError, "num_training_steps"):
            self.verify(directory, num_training_steps=20)

    def test_wrong_warmup_steps_are_refused(self):
        directory, _, _ = self.write_checkpoint(num_warmup_steps=2)
        with self.assertRaisesRegex(CheckpointError, "num_warmup_steps"):
            self.verify(directory, num_warmup_steps=5)

    def test_configured_initial_lr_mismatch_is_refused(self):
        directory, _, _ = self.write_checkpoint(learning_rate=1e-5)
        with self.assertRaisesRegex(CheckpointError, "base_lrs"):
            self.verify(directory, learning_rate=1e-4)

    def test_cosine_annealing_state_is_refused(self):
        torch = self.torch
        directory, _, _ = self.write_checkpoint(scheduler=(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                torch.optim.AdamW(self.model.parameters(), lr=1e-5), T_max=10)))
        with self.assertRaisesRegex(CheckpointError, "not a Transformers cosine"):
            self.verify(directory)


class ResumeStateTests(unittest.TestCase):
    """Scheduler-independent resume checks (the production scheduler round trip
    lives in ProductionSchedulerIntegrationTests, dependency-gated)."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    @support.requires_torch
    def test_resume_state_detects_missing_state_file(self):
        import torch

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2))

        directory = self.run_dir / "broken"
        directory.mkdir()
        (directory / "trainer_state.json").write_text("{}")
        with self.assertRaisesRegex(CheckpointError, "missing"):
            checkpointing.verify_resume_state(
                directory, torch=torch, model=Tiny(), learning_rate=1e-5,
                fused=False, num_training_steps=10, num_warmup_steps=2,
                device="cpu")

    @support.requires_torch
    def test_resume_state_rejects_mismatched_optimizer(self):
        import torch

        class Tiny(torch.nn.Module):
            def __init__(self, size):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(size))

        source = Tiny(4)
        optimizer = torch.optim.AdamW(source.parameters(), lr=1e-5)
        source.weight.sum().backward()
        optimizer.step()
        directory = self.run_dir / "checkpoints" / "smoke-1000" / "checkpoint-2"
        directory.mkdir(parents=True)
        torch.save(optimizer.state_dict(), directory / "optimizer.pt")
        torch.save({}, directory / "scheduler.pt")
        (directory / "trainer_state.json").write_text(json.dumps({"global_step": 2}))
        with self.assertRaises(CheckpointError):
            checkpointing.verify_resume_state(
                directory, torch=torch, model=Tiny(8), learning_rate=1e-5,
                fused=False, num_training_steps=10, num_warmup_steps=2,
                device="cpu")


if __name__ == "__main__":
    unittest.main()
