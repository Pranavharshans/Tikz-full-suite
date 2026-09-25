"""Tests for token accounting, trainer wiring, emission guards and gate criteria.

Everything here is CPU-only: the Trainer subclass is built on a fake base class
and the gate criteria are pure functions over metrics documents.
"""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from tests import support

from stage1 import checkpointing, collator, train
from stage1.errors import CheckpointError, DataError

HAS_DATASETS = importlib.util.find_spec("datasets") is not None


class FakeTrainerCallback:
    pass


class FakeTrainerBase:
    """Minimal stand-in for transformers.Trainer used by the subclass tests."""

    def __init__(self, calls=None):
        self.calls = calls if calls is not None else []
        self.state = types.SimpleNamespace(global_step=0, max_steps=0, epoch=0.0)
        self._next_loss = 0.5

    def training_step(self, model, inputs, *args, **kwargs):
        self.calls.append(("training_step", len(inputs.get("labels", []))))
        return self._next_loss

    def prediction_step(self, model, inputs, *args, **kwargs):
        self.calls.append(("prediction_step",))
        return 0.0


FAKE_TRANSFORMERS = types.SimpleNamespace(Trainer=FakeTrainerBase,
                                          TrainerCallback=FakeTrainerCallback)


def monitor(**overrides):
    values = dict(planned_per_epoch=100, epochs=1,
                  gradient_accumulation_steps=2, check_gradients=True)
    values.update(overrides)
    return train.TrainingMonitor(**values)


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeMask:
    def __init__(self, values):
        self.values = list(values)

    def sum(self):
        return FakeScalar(sum(self.values))


class FakeTensor1D:
    """Torch-like 1-D tensor: vectorised comparisons, no iteration."""

    def __init__(self, values):
        self.values = list(values)

    def numel(self):
        return len(self.values)

    def __ne__(self, other):
        return FakeMask(1 if value != other else 0 for value in self.values)

    def __iter__(self):
        raise RuntimeError("iteration over a tensor is not supported")


class FakeTensor2D:
    """Torch-like 2-D tensor whose iteration raises, like real torch tensors."""

    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    def numel(self):
        return sum(len(row) for row in self.rows)

    def __ne__(self, other):
        return FakeMask(1 if value != other else 0
                        for row in self.rows for value in row)

    def __iter__(self):
        raise RuntimeError("iteration over a 2D tensor is not supported")


class TensorAccountingTests(unittest.TestCase):
    """Regression: 2-D labels must count tokens, not rows, and never iterate."""

    def test_flat_list_counts_tokens(self):
        self.assertEqual(train.count_supervised_labels([-100, 1, 2]), 2)

    def test_nested_batch_list_counts_tokens_not_rows(self):
        labels = [[-100, 1, 2], [-100, -100, 3]]
        self.assertEqual(train.count_supervised_labels(labels), 3)

    def test_1d_tensor_counts_tokens(self):
        self.assertEqual(train.count_supervised_labels(FakeTensor1D([-100, 1])), 1)

    def test_2d_tensor_counts_tokens_without_iterating(self):
        labels = FakeTensor2D([[-100, 1, 2], [-100, -100, 3]])
        self.assertEqual(train.count_supervised_labels(labels), 3)

    def test_monitor_counts_a_2d_batch(self):
        accounting = monitor()
        accounting.record_micro_batch(
            FakeTensor2D([[-100, 1, 2], [-100, -100, 3]]))
        self.assertEqual(accounting.pending_supervised_tokens, 3)

    def test_none_labels_count_zero(self):
        self.assertEqual(train.count_supervised_labels(None), 0)


class AccountingTests(unittest.TestCase):
    def test_counts_only_supervised_labels(self):
        accounting = monitor()
        accounting.record_micro_batch([-100, -100, 5, 6, 7])
        self.assertEqual(accounting.pending_supervised_tokens, 3)
        self.assertEqual(accounting.committed_supervised_tokens, 0)

    def test_commit_only_happens_at_step_end(self):
        accounting = monitor(gradient_accumulation_steps=2)
        accounting.record_micro_batch([-100, 1])          # micro 1
        self.assertEqual(accounting.committed_supervised_tokens, 0)
        accounting.record_micro_batch([-100, 1, 2])       # micro 2
        self.assertEqual(accounting.committed_supervised_tokens, 0)
        accounting.commit_step()
        self.assertEqual(accounting.committed_supervised_tokens, 3)
        self.assertEqual(accounting.committed_steps, 1)
        self.assertEqual(accounting.pending_supervised_tokens, 0)

    def test_evaluation_batches_never_reach_the_monitor(self):
        accounting = monitor()
        trainer = train.make_trainer_class(
            FAKE_TRANSFORMERS, accounting, isfinite=lambda loss: True,
            grad_norm=lambda model: 1.0)(calls=[])
        trainer.training_step(object(), {"labels": [-100, 1, 2]})
        trainer.prediction_step(object(), {"labels": [-100, 9, 9, 9]})
        self.assertEqual(accounting.micro_batches, 1)
        self.assertEqual(accounting.pending_supervised_tokens, 2)

    def test_trainer_subclass_does_not_override_evaluation_paths(self):
        accounting = monitor()
        klass = train.make_trainer_class(FAKE_TRANSFORMERS, accounting,
                                         isfinite=lambda loss: True,
                                         grad_norm=lambda model: 1.0)
        self.assertIn("training_step", klass.__dict__)
        self.assertNotIn("prediction_step", klass.__dict__)
        self.assertNotIn("evaluation_loop", klass.__dict__)

    def test_prefetched_but_unconsumed_batches_do_not_count(self):
        # The monitor is fed from training_step, so a batch that was collated
        # (for example by a prefetching dataloader) but never passed to the
        # model cannot contribute. Collating alone must not touch accounting.
        accounting = monitor()
        plan = collator.pad_batch(
            [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}], pad_token_id=0)
        self.assertEqual(plan.supervised_tokens, 2)
        self.assertEqual(accounting.micro_batches, 0)
        self.assertEqual(accounting.pending_supervised_tokens, 0)

    def test_gradient_checks_at_accumulation_boundaries_only(self):
        accounting = monitor(gradient_accumulation_steps=3)
        trainer = train.make_trainer_class(
            FAKE_TRANSFORMERS, accounting, isfinite=lambda loss: True,
            grad_norm=lambda model: 0.5)(calls=[])
        for _ in range(3):
            trainer.training_step(object(), {"labels": [-100, 1]})
        self.assertEqual(accounting.gradient_checks, 1)
        self.assertEqual(accounting.max_gradient_norm, 0.5)
        self.assertEqual(accounting.gradient_failures, 0)

    def test_non_finite_losses_and_gradients_are_recorded(self):
        accounting = monitor(gradient_accumulation_steps=1)
        trainer = train.make_trainer_class(
            FAKE_TRANSFORMERS, accounting, isfinite=lambda loss: False,
            grad_norm=lambda model: float("nan"))(calls=[])
        trainer.training_step(object(), {"labels": [-100, 1]})
        self.assertEqual(accounting.non_finite_losses, 1)
        self.assertEqual(accounting.gradient_failures, 1)

    def test_resume_does_not_double_count(self):
        accounting = monitor()
        accounting.resume_from({"supervised_tokens_seen": 500, "global_step": 4})
        self.assertEqual(accounting.committed_supervised_tokens, 500)
        accounting.record_micro_batch([-100, 1, 2])
        accounting.commit_step()
        self.assertEqual(accounting.committed_supervised_tokens, 502)
        self.assertEqual(accounting.committed_steps, 5)

    def test_finalize_commits_only_completed_steps(self):
        accounting = monitor()
        accounting.record_micro_batch([-100, 1])
        # No optimizer step happened: the in-flight tokens stay uncommitted.
        self.assertFalse(accounting.finalize(global_step=0))
        self.assertEqual(accounting.committed_supervised_tokens, 0)
        self.assertEqual(accounting.pending_supervised_tokens, 1)
        # One completed step: the trailing cycle is committed once.
        self.assertTrue(accounting.finalize(global_step=1))
        self.assertFalse(accounting.finalize(global_step=1))
        self.assertEqual(accounting.committed_supervised_tokens, 1)

    def test_snapshot_is_explicit_about_exactness(self):
        accounting = monitor(planned_per_epoch=10, epochs=3)
        snapshot = accounting.snapshot()
        self.assertEqual(snapshot["planned_supervised_tokens"], 30)
        self.assertTrue(snapshot["supervised_tokens_exact"])
        self.assertIn("completed", snapshot["accounting_method"])

    def test_accounting_beyond_the_plan_is_refused(self):
        accounting = monitor(planned_per_epoch=2, epochs=1)
        accounting.committed_supervised_tokens = 2
        accounting.assert_within_plan()
        accounting.committed_supervised_tokens = 3
        with self.assertRaisesRegex(DataError, "exceed the deterministic plan"):
            accounting.assert_within_plan()

    def test_accounting_may_be_below_the_plan(self):
        # An interrupted or max_steps-limited run commits fewer tokens.
        accounting = monitor(planned_per_epoch=100, epochs=2)
        accounting.committed_supervised_tokens = 50
        accounting.assert_within_plan()


class CallbackTests(unittest.TestCase):
    def make_callbacks(self, run_dir, accounting):
        return train.make_callbacks(
            FAKE_TRANSFORMERS, monitor=accounting,
            checkpointing_module=checkpointing, identity_sha256="a" * 64,
            gate_name="smoke-1000", model_id="test/Model-A",
            model_revision="b" * 40, adapter="minicpm5", run_id="run1",
            run_dir=run_dir)

    def test_step_end_commits_and_log_records(self):
        with tempfile.TemporaryDirectory() as directory:
            accounting = monitor()
            accounting.record_micro_batch([-100, 1, 2])
            callback, _ = self.make_callbacks(Path(directory), accounting)
            state = types.SimpleNamespace(global_step=1, epoch=0.1)
            callback.on_step_end(args=None, state=state, control=None)
            self.assertEqual(accounting.committed_supervised_tokens, 2)
            logs = {"loss": 1.0}
            callback.on_log(args=None, state=state, control=None, logs=logs)
            self.assertEqual(logs["supervised_tokens_committed"], 2)
            record = json.loads(
                (Path(directory) / "logs" / "smoke-1000.jsonl").read_text().splitlines()[0])
            self.assertEqual(record["logs"]["loss"], 1.0)

    def test_train_end_refuses_zero_committed_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            accounting = monitor()
            callback, _ = self.make_callbacks(Path(directory), accounting)
            with self.assertRaisesRegex(DataError, "zero committed supervised"):
                callback.on_train_end(
                    args=None, state=types.SimpleNamespace(global_step=0),
                    control=None)

    def test_save_without_model_or_optimizer_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            accounting = monitor()
            _, save_callback = self.make_callbacks(Path(directory), accounting)
            checkpoint_dir = Path(directory) / "checkpoints" / "smoke-1000" / "checkpoint-3"
            checkpoint_dir.mkdir(parents=True)
            args = types.SimpleNamespace(
                output_dir=str(Path(directory) / "checkpoints" / "smoke-1000"),
                get_warmup_steps=lambda steps: 2)
            state = types.SimpleNamespace(global_step=3, epoch=0.2, max_steps=10)
            with self.assertRaisesRegex(DataError, "no model/optimizer"):
                save_callback.on_save(args=args, state=state, control=None)

    def test_checkpoint_meta_records_gate_model_and_committed_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            accounting = monitor()
            accounting.committed_supervised_tokens = 321
            _, save_callback = self.make_callbacks(Path(directory), accounting)
            checkpoint_dir = Path(directory) / "checkpoints" / "smoke-1000" / "checkpoint-3"
            checkpoint_dir.mkdir(parents=True)
            args = types.SimpleNamespace(
                output_dir=str(Path(directory) / "checkpoints" / "smoke-1000"),
                get_warmup_steps=lambda steps: 2)
            state = types.SimpleNamespace(global_step=3, epoch=0.2, max_steps=10)
            parameter = object()
            model = types.SimpleNamespace(named_parameters=lambda: [("weight", parameter)])
            optimizer = types.SimpleNamespace(param_groups=[{"params": [parameter]}])
            save_callback.on_save(args=args, state=state, control=None,
                                  model=model, optimizer=optimizer)
            meta = json.loads(
                (checkpoint_dir / checkpointing.CHECKPOINT_META_NAME).read_text())
            self.assertEqual(meta["gate"], "smoke-1000")
            self.assertEqual(meta["model_id"], "test/Model-A")
            self.assertEqual(meta["model_revision"], "b" * 40)
            self.assertEqual(meta["adapter"], "minicpm5")
            self.assertEqual(meta["supervised_tokens_seen"], 321)


class EmissionGuardTests(unittest.TestCase):
    def test_quarantined_row_is_refused_at_emission(self):
        with self.assertRaisesRegex(DataError, "Quarantined row"):
            train.check_emission_row("row-1", 10, max_seq_len=100,
                                     quarantined={"row-1"}, wanted={"row-1"})

    def test_overlength_row_is_refused_at_emission(self):
        with self.assertRaisesRegex(DataError, "above data.max_seq_len"):
            train.check_emission_row("row-1", 101, max_seq_len=100,
                                     quarantined=set(), wanted={"row-1"})

    def test_unselected_row_is_refused_at_emission(self):
        with self.assertRaisesRegex(DataError, "never selected"):
            train.check_emission_row("row-9", 10, max_seq_len=100,
                                     quarantined=set(), wanted={"row-1"})

    def test_missing_expected_id_is_refused(self):
        with self.assertRaisesRegex(DataError, "not emitted"):
            train.check_emitted_ids({"a", "b"}, ["a"], set())

    def test_extra_or_quarantined_id_is_refused(self):
        with self.assertRaisesRegex(DataError, "unexpected row"):
            train.check_emitted_ids({"a"}, ["a", "b"], set())
        with self.assertRaisesRegex(DataError, "Quarantined rows were emitted"):
            train.check_emitted_ids({"a"}, ["a"], {"a"})


class ExportIdentityTests(unittest.TestCase):
    """Training refuses an export that does not match the prepared manifest."""

    def prepared(self, logical):
        return {"manifest": {"data_identity": {"dataset_logical_sha256": logical}}}

    def test_matching_export_is_accepted(self):
        export = types.SimpleNamespace(
            root="/exports/right", dataset_logical_sha256="a" * 64)
        train.check_export_identity(export, self.prepared("a" * 64))

    def test_mismatched_export_is_refused_with_both_hashes(self):
        export = types.SimpleNamespace(
            root="/exports/wrong", dataset_logical_sha256="b" * 64)
        with self.assertRaisesRegex(DataError, "was built from"):
            train.check_export_identity(export, self.prepared("a" * 64))


class ScheduleBoundsTests(unittest.TestCase):
    def bounds(self, **overrides):
        values = dict(examples=1000, per_device_batch_size=2,
                      gradient_accumulation_steps=8, epochs=1, max_steps=-1)
        values.update(overrides)
        return train.optimizer_step_bounds(**values)

    def test_smoke_schedule_bounds(self):
        bounds = self.bounds()
        self.assertEqual(bounds["micro_batches_per_epoch"], 500)
        self.assertEqual(bounds["steps_per_epoch_lower"], 62)
        self.assertEqual(bounds["steps_per_epoch_upper"], 63)
        self.assertEqual(bounds["expected_steps_lower"], 62)
        self.assertEqual(bounds["expected_steps_upper"], 63)

    def test_overfit_schedule_bounds(self):
        bounds = self.bounds(examples=100, epochs=40)
        self.assertEqual(bounds["micro_batches_per_epoch"], 50)
        self.assertEqual(bounds["expected_steps_lower"], 240)
        self.assertEqual(bounds["expected_steps_upper"], 280)

    def test_max_steps_truncates_both_bounds(self):
        bounds = self.bounds(max_steps=30)
        self.assertEqual(bounds["expected_steps_lower"], 30)
        self.assertEqual(bounds["expected_steps_upper"], 30)

    def test_empty_dataset_and_bad_batch_are_refused(self):
        with self.assertRaisesRegex(DataError, "empty dataset"):
            self.bounds(examples=0)
        with self.assertRaisesRegex(DataError, "positive"):
            self.bounds(per_device_batch_size=0)

    def test_unreachable_save_interval_is_refused(self):
        config = support.make_config(gates={"smoke-1000": {"save_steps": 200}})
        gate = config.gate("smoke-1000")
        with self.assertRaisesRegex(DataError, "no checkpoint would ever be written"):
            train.validate_gate_intervals(gate, self.bounds(), has_eval=True)

    def test_eval_beyond_schedule_is_recorded_not_refused(self):
        config = support.make_config(gates={"smoke-1000": {
            "save_steps": 25, "eval_steps": 500}})
        report = train.validate_gate_intervals(
            config.gate("smoke-1000"), self.bounds(), has_eval=True)
        self.assertTrue(report["save_within_schedule"])
        self.assertFalse(report["eval_within_schedule"])

    def test_shipped_gate_defaults_have_reachable_intervals(self):
        config = support.make_config()
        examples = {"overfit-100": 100, "smoke-1000": 1000, "full": 100000}
        for name, count in examples.items():
            with self.subTest(gate=name):
                gate = config.gate(name)
                bounds = self.bounds(
                    examples=count, epochs=gate.epochs,
                    max_steps=gate.max_steps)
                report = train.validate_gate_intervals(gate, bounds, has_eval=True)
                self.assertTrue(report["save_within_schedule"])
                self.assertTrue(report["eval_within_schedule"])


class FakeTorch:
    """Minimal torch stand-in for the forward-pass helper."""

    long = "long"

    @staticmethod
    def tensor(values, dtype=None, device=None):
        return values

    @staticmethod
    def no_grad():
        import contextlib
        return contextlib.nullcontext()


class FakeModel:
    def __init__(self, loss=0.25, shape=(2, 4)):
        self.loss = loss
        self.shape = shape
        self.loaded = []

    def eval(self):
        return self

    def __call__(self, **batch):
        return types.SimpleNamespace(
            loss=self.loss, logits=types.SimpleNamespace(shape=self.shape))

    def load_state_dict(self, state, strict=False):
        self.loaded.append(state.get("marker"))
        return [], []


class ArtifactWeightTests(unittest.TestCase):
    """Checkpoint verification must not leave older weights live.

    Also covers the method-aware weight loading introduced for LoRA: adapters
    load only adapter tensors, never base weights.
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "checkpoint-25"
        self.final = self.root / "final"
        for directory in (self.checkpoint, self.final):
            directory.mkdir(parents=True)
            (directory / "model.safetensors").write_bytes(b"weights")
        self.loader = lambda path: {"marker": Path(path).parent.name}
        self.batch_plan = collator.pad_batch(
            [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}], pad_token_id=0)

    def tearDown(self):
        self.temporary.cleanup()

    def test_load_artifact_weights_loads_exactly(self):
        model = FakeModel()
        report = checkpointing.load_artifact_weights(
            model, self.checkpoint, loader=self.loader)
        self.assertEqual(model.loaded, ["checkpoint-25"])
        self.assertEqual(report["tensors"], 1)
        self.assertEqual(report["method"], "full")

    def test_load_artifact_weights_refuses_mismatch(self):
        class Mismatched(FakeModel):
            def load_state_dict(self, state, strict=False):
                return ["missing.weight"], []

        with self.assertRaisesRegex(CheckpointError, "does not match the model"):
            checkpointing.load_artifact_weights(
                Mismatched(), self.checkpoint, loader=self.loader)

    def test_load_artifact_weights_refuses_empty_directory(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(CheckpointError, "No weight file"):
            checkpointing.load_artifact_weights(FakeModel(), empty,
                                                loader=self.loader)

    def test_restore_final_weights_runs_after_checkpoint_weights(self):
        model = FakeModel(loss=0.125, shape=(2, 4))
        checkpointing.load_artifact_weights(model, self.checkpoint,
                                            loader=self.loader)
        report = train.restore_final_weights(
            model, self.final, method="full", batch_plan=self.batch_plan,
            torch=FakeTorch, loader=self.loader)
        self.assertEqual(model.loaded, ["checkpoint-25", "final"])
        self.assertTrue(report["restored_after_checkpoint_verification"])
        self.assertEqual(report["restore_method"], "full")
        self.assertEqual(report["restore_loss"], 0.125)
        self.assertEqual(report["restore_logits_shape"], [2, 4])

    def test_restore_accepts_unsloth_lazy_logits(self):
        model = FakeModel(loss=0.125, shape=lambda: None)
        report = train.restore_final_weights(
            model, self.final, method="full", batch_plan=self.batch_plan,
            torch=FakeTorch, loader=self.loader)
        self.assertIsNone(report["restore_logits_shape"])
        self.assertFalse(report["restore_logits_materialized"])

    def test_non_finite_forward_loss_is_refused(self):
        model = FakeModel(loss=float("nan"))
        with self.assertRaisesRegex(DataError, "non-finite loss"):
            train.restore_final_weights(
                model, self.final, method="full", batch_plan=self.batch_plan,
                torch=FakeTorch, loader=self.loader)

class RestoreMask:
    def __init__(self, ok):
        self.ok = ok

    def all(self):
        return self.ok


class RestoreTensor:
    def __init__(self, value, *, dtype="float32", shape=(1,)):
        self.value = value
        self.dtype = dtype
        self.shape = shape

    def __eq__(self, other):
        return RestoreMask(self.value == getattr(other, "value", other))


class NoStateDictModel:
    """Lora restore must not consult model.state_dict() at all."""

    def state_dict(self):
        raise AssertionError("model.state_dict() must not be used for LoRA restore")

    def load_state_dict(self, state, strict=False):
        raise AssertionError("model.load_state_dict() must not be used for LoRA restore")


class LoraRestoreTests(unittest.TestCase):
    """PEFT round trip: set via the adapter API, read back via the getter.

    Regression: the read-back must not compare raw ``model.state_dict()`` keys
    and must not reject setter ``missing_keys`` that are base-model weights.
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.adapter = self.root / "adapter"
        self.adapter.mkdir()
        (self.adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        self.saved = {
            "base_model.model.base.layers.0.q_proj.lora_A.weight":
                RestoreTensor(1, shape=(4, 8)),
            "base_model.model.base.layers.0.q_proj.lora_B.weight":
                RestoreTensor(2, shape=(8, 4)),
        }
        self.loader = lambda path: dict(self.saved)

    def tearDown(self):
        self.temporary.cleanup()

    def restore(self, *, restored=None, setter_result=None, model=None,
                loader=None, setter=None, getter=None):
        if restored is None:
            restored = {key.replace("base_model.model.", ""): value
                        for key, value in self.saved.items()}
        if setter is None:
            calls = []

            def setter(model_arg, state_dict, adapter_name="default"):
                calls.append({"state_dict": dict(state_dict),
                              "adapter_name": adapter_name})
                return setter_result if setter_result is not None else types.SimpleNamespace(
                    missing_keys=["base_model.model.base.embed_tokens.weight"],
                    unexpected_keys=[])
            self.setter_calls = calls
        if getter is None:
            def getter(model_arg, adapter_name="default"):
                return dict(restored)
        return checkpointing.load_artifact_weights(
            model or NoStateDictModel(), self.adapter, method="lora",
            loader=loader or self.loader, adapter_state_setter=setter,
            adapter_state_getter=getter)

    def test_restore_uses_both_peft_apis_and_ignores_base_missing_keys(self):
        report = self.restore()
        self.assertEqual(report["restore_api"], "peft.set_peft_model_state_dict")
        self.assertEqual(report["readback_api"], "peft.get_peft_model_state_dict")
        self.assertTrue(report["readback_verified"])
        self.assertEqual(report["adapter_name"], "default")
        self.assertEqual(report["compared_tensors"], 2)
        self.assertEqual(report["setter_missing_keys"], 1)  # base weight: ignored
        self.assertEqual(self.setter_calls[0]["adapter_name"], "default")
        self.assertEqual(set(self.setter_calls[0]["state_dict"]), set(self.saved))

    def test_readback_normalizes_wrapper_prefixes(self):
        # The live adapter is spelled without the prefix, the file with it.
        restored = {key.replace("base_model.model.", ""): value
                    for key, value in self.saved.items()}
        report = self.restore(restored=restored)
        self.assertTrue(report["readback_verified"])

    def test_adapter_parameter_missing_from_the_artifact_fails(self):
        restored = dict(self.saved)
        restored["base_model.model.base.layers.0.gate_proj.lora_A.weight"] = (
            RestoreTensor(3, shape=(4, 8)))
        with self.assertRaisesRegex(CheckpointError,
                                    "missing from the saved artifact"):
            self.restore(restored=restored)

    def test_extra_saved_tensor_fails(self):
        restored = {key.replace("base_model.model.", ""): value
                    for key, value in self.saved.items()
                    if "lora_B" in key}
        with self.assertRaisesRegex(CheckpointError,
                                    "not part of this model's adapter"):
            self.restore(restored=restored)

    def test_shape_dtype_and_value_mismatches_fail(self):
        for restored_change, pattern in (
                ({"base_model.model.base.layers.0.q_proj.lora_A.weight":
                  RestoreTensor(1, shape=(8, 8))}, "shape mismatch"),
                ({"base_model.model.base.layers.0.q_proj.lora_A.weight":
                  RestoreTensor(1, dtype="bfloat16", shape=(4, 8))}, "dtype mismatch"),
                ({"base_model.model.base.layers.0.q_proj.lora_A.weight":
                  RestoreTensor(9, shape=(4, 8))}, "values differ"),
        ):
            with self.subTest(pattern=pattern):
                restored = dict(self.saved)
                restored.update(restored_change)
                with self.assertRaisesRegex(CheckpointError, pattern):
                    self.restore(restored=restored)

    def test_non_adapter_artifact_is_refused(self):
        loader = lambda path: {"base.weight": RestoreTensor(1)}
        with self.assertRaisesRegex(CheckpointError, "no adapter tensors"):
            self.restore(loader=loader)

    def test_restore_requires_peft_without_a_getter_seam(self):
        # Simulate an environment without peft: the real getter must not be
        # used, and the missing dependency must be reported as such.
        with mock.patch.dict(sys.modules, {"peft": None}):
            with self.assertRaisesRegex(CheckpointError, "peft is required"):
                checkpointing.load_artifact_weights(
                    NoStateDictModel(), self.adapter, method="lora",
                    loader=self.loader,
                    adapter_state_setter=lambda model, state, adapter_name="default": None)

    def test_setter_failure_is_reported_as_adapter_incompatibility(self):
        def setter(model, state_dict, adapter_name="default"):
            raise ValueError("size mismatch for lora_A")

        with self.assertRaisesRegex(CheckpointError, "PEFT failed to restore"):
            checkpointing.load_artifact_weights(
                NoStateDictModel(), self.adapter, method="lora",
                loader=self.loader, adapter_state_setter=setter,
                adapter_state_getter=lambda model, adapter_name="default": dict(self.saved))


def metrics_document(**overrides):
    """A minimal metrics document that satisfies the smoke gate by default."""
    document = {
        "gate": "smoke-1000",
        "final_eval_loss": 0.4,
        "optimizer_steps": 10,
        "expected_optimizer_steps": 10,
        "epochs_completed": 1.0,
        "losses": {"count": 10, "finite": 10, "non_finite": 0,
                   "first": 2.0, "last": 0.4},
        "gradients": {"gradient_checks": 5, "gradient_failures": 0,
                      "non_finite_losses": 0},
        "checkpoint": {"verified": True, "reload_verified": True,
                       "resume_verified": True},
        "final_artifact": {"verified": True},
        "generation": {"examples": 2, "completed": 2, "truncated": 0,
                       "empty": 0, "errors": 0},
        "compilation": {"attempted": 2, "success": 1, "categories": {}},
    }
    document.update(overrides)
    return document


class GateCriteriaTests(unittest.TestCase):
    def gate(self, name):
        return support.make_config().gate(name)

    def outcome(self, name, document):
        return train.gate_outcome(self.gate(name), document)

    def test_smoke_passes_with_complete_evidence(self):
        status, outcome = self.outcome("smoke-1000", metrics_document())
        self.assertEqual(status, "passed", outcome)

    def test_smoke_fails_without_gradient_checks(self):
        status, outcome = self.outcome("smoke-1000", metrics_document(
            gradients={"gradient_checks": 0, "gradient_failures": 0,
                       "non_finite_losses": 0}))
        self.assertEqual(status, "failed")
        self.assertIn("gradients_checked", outcome["failed_criteria"])

    def test_smoke_fails_on_non_finite_values(self):
        status, outcome = self.outcome("smoke-1000", metrics_document(
            losses={"count": 3, "finite": 2, "non_finite": 1,
                    "first": 1.0, "last": float("nan")}))
        self.assertEqual(status, "failed")
        self.assertIn("no_nan_or_inf", outcome["failed_criteria"])

    def test_smoke_fails_on_incomplete_steps(self):
        status, outcome = self.outcome("smoke-1000", metrics_document(
            optimizer_steps=9, expected_optimizer_steps=10))
        self.assertEqual(status, "failed")
        self.assertIn("optimizer_steps_completed", outcome["failed_criteria"])

    def test_smoke_fails_without_same_gate_resume_verification(self):
        status, outcome = self.outcome("smoke-1000", metrics_document(
            checkpoint={"verified": True, "reload_verified": True,
                        "resume_verified": False}))
        self.assertEqual(status, "failed")
        self.assertIn("same_gate_resume_verified", outcome["failed_criteria"])

    def test_smoke_fails_when_generation_errored(self):
        status, outcome = self.outcome("smoke-1000", metrics_document(
            generation={"examples": 2, "completed": 1, "truncated": 0,
                        "empty": 1, "errors": 1}))
        self.assertEqual(status, "failed")
        self.assertIn("generation_completed", outcome["failed_criteria"])

    def test_smoke_requires_compile_results_but_not_improvement(self):
        status, outcome = self.outcome("smoke-1000", metrics_document(
            compilation={"attempted": 0, "success": 0, "categories": {}}))
        self.assertEqual(status, "failed")
        self.assertIn("compilation_recorded", outcome["failed_criteria"])
        status, _ = self.outcome("smoke-1000", metrics_document(
            compilation={"attempted": 2, "success": 0,
                         "categories": {"latex_error": 2}}))
        self.assertEqual(status, "passed")

    def test_overfit_requires_threshold_and_reload(self):
        status, outcome = self.outcome("overfit-100", metrics_document(
            gate="overfit-100", final_eval_loss=0.04))
        self.assertEqual(status, "passed", outcome)
        status, outcome = self.outcome("overfit-100", metrics_document(
            gate="overfit-100", final_eval_loss=0.2))
        self.assertEqual(status, "failed")
        self.assertIn("eval_loss_threshold", outcome["failed_criteria"])
        status, outcome = self.outcome("overfit-100", metrics_document(
            gate="overfit-100",
            checkpoint={"verified": True, "reload_verified": False,
                        "resume_verified": True}))
        self.assertEqual(status, "failed")
        self.assertIn("checkpoint_reload_verified", outcome["failed_criteria"])

    def test_overfit_requires_loss_reduction_when_configured(self):
        status, outcome = self.outcome("overfit-100", metrics_document(
            gate="overfit-100", final_eval_loss=0.04,
            losses={"count": 5, "finite": 5, "non_finite": 0,
                    "first": 0.05, "last": 0.04}))
        self.assertEqual(status, "failed")
        self.assertIn("loss_reduction", outcome["failed_criteria"])

    def test_full_requires_steps_or_epochs(self):
        status, outcome = self.outcome("full", metrics_document(
            gate="full", optimizer_steps=5, expected_optimizer_steps=10,
            epochs_completed=0.5))
        self.assertEqual(status, "failed")
        self.assertIn("steps_or_epochs_completed", outcome["failed_criteria"])
        status, outcome = self.outcome("full", metrics_document(
            gate="full", optimizer_steps=10, expected_optimizer_steps=10,
            epochs_completed=1.0))
        self.assertEqual(status, "passed", outcome)

    def test_full_requires_final_artifact_and_finite_validation_loss(self):
        status, outcome = self.outcome("full", metrics_document(
            gate="full", final_artifact={"verified": False}))
        self.assertEqual(status, "failed")
        self.assertIn("final_checkpoint_verified", outcome["failed_criteria"])
        status, outcome = self.outcome("full", metrics_document(
            gate="full", final_eval_loss=None))
        self.assertEqual(status, "failed")
        self.assertIn("validation_loss_finite", outcome["failed_criteria"])


@unittest.skipUnless(HAS_DATASETS, "datasets is not installed")
@support.requires_pyarrow
class DatasetConstructionTests(unittest.TestCase):
    def test_tokenized_dataset_excludes_quarantined_rows(self):
        from stage1 import data as data_module, formatting
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = support.default_rows(8, pad_tikz=30)
            export = support.make_export(root / "export", rows=rows)
            prepared_dir = root / "prepared"
            config = support.make_config(data={
                "max_seq_len": 128, "max_quarantined_fraction": 1.0,
                "splits": {"validation_fraction": 0.25, "test_fraction": 0.25}})
            data_module.prepare_dataset(
                [config], export_dir=export["root"], prepared_dir=prepared_dir,
                tokenizer_factory=support.fake_tokenizer_factory())
            manifest = data_module.load_split_manifest(prepared_dir)
            eligibility = data_module.load_eligibility(prepared_dir, "minicpm5")
            export_info = data_module.verify_export(export["root"], quick=True)
            tokenizer = support.FakeTokenizer()
            template = formatting.resolve_chat_template(
                tokenizer, support.config_module.TokenizerConfig())
            # Every row is quarantined: training must refuse rather than build
            # an empty dataset or fall back to ineligible rows.
            with self.assertRaisesRegex(DataError, "No eligible rows"):
                train.build_tokenized_dataset(
                    config, tokenizer, template, export_info, manifest, "train",
                    eligibility)


if __name__ == "__main__":
    unittest.main()
