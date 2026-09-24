"""Tests for preflight orchestration, individual checks and the no-train rule."""
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock
import sys

from tests import support

from stage1 import identity, preflight
from stage1.errors import DataError

STAGE1 = Path(__file__).resolve().parents[1]


def context(tmpdir=None, config=None, **scratch):
    run_dir = Path(tmpdir) if tmpdir else Path("/tmp/stage1-test-run")
    ctx = preflight.PreflightContext(
        config=config or support.make_config(), run_dir=run_dir,
        identity={"data": {"data_identity_sha256": "0" * 64}},
        prepared={"data_identity_sha256": "0" * 64, "manifest": {
            "data_identity": {"dataset_logical_sha256": "1" * 64}}})
    ctx.scratch.update(scratch)
    return ctx


def ok(detail="ok"):
    return {"status": "pass", "detail": detail, "data": {}}


def bad(detail="bad"):
    return {"status": "fail", "detail": detail, "data": {}}


class OrchestrationTests(unittest.TestCase):
    def test_prerequisite_failure_skips_dependents(self):
        calls = []

        def first(ctx):
            calls.append("first")
            return bad("first failed")

        def second(ctx):
            calls.append("second")
            return ok()

        report = preflight.run_preflight(context(), checks=(
            ("first", first, ()),
            ("second", second, ("first",)),
        ))
        self.assertEqual(calls, ["first"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failed"], ["first"])
        self.assertEqual(report["skipped"], ["second"])
        self.assertFalse(report["complete"])

    def test_all_passing_is_complete(self):
        report = preflight.run_preflight(context(), checks=(
            ("a", lambda ctx: ok(), ()),
            ("b", lambda ctx: ok(), ("a",)),
        ))
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["complete"])

    def test_explicit_skip_marks_incomplete(self):
        report = preflight.run_preflight(context(), checks=(
            ("a", lambda ctx: ok(), ()),
            ("b", lambda ctx: ok(), ()),
        ), skip=("b",))
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["complete"])
        self.assertEqual(report["skipped"], ["b"])

    def test_unknown_skip_is_refused(self):
        with self.assertRaisesRegex(DataError, "Unknown --skip-check"):
            preflight.run_preflight(context(), skip=("nope",))

    def test_check_exception_becomes_failure(self):
        def explode(ctx):
            raise RuntimeError("kaboom")

        report = preflight.run_preflight(context(), checks=(("a", explode, ()),))
        self.assertEqual(report["status"], "failed")
        self.assertIn("RuntimeError", report["checks"][0]["detail"])

    def test_report_never_claims_to_start_training(self):
        report = preflight.run_preflight(context(), checks=(
            ("a", lambda ctx: ok(), ()),))
        self.assertFalse(report["started_training"])


class DatasetIdentityTests(unittest.TestCase):
    def test_cli_export_path_is_used_when_config_path_is_null(self):
        ctx = context()
        ctx.config.data.export_dir = None
        ctx.export_dir = Path("/resolved/from-cli/export")
        export = types.SimpleNamespace(
            rows=98450, dataset_logical_sha256="1" * 64)

        with mock.patch("stage1.data.verify_export", return_value=export) as verify:
            result = preflight.check_dataset_identity(ctx)

        self.assertEqual(result["status"], "pass")
        verify.assert_called_once_with(
            ctx.export_dir,
            require_complete=ctx.config.data.require_complete_export,
            quick=True)

    def test_specs_are_structurally_sound(self):
        names = [name for name, _, _ in preflight.CHECK_SPECS]
        self.assertEqual(len(names), len(set(names)))
        for name, function_name, requires in preflight.CHECK_SPECS:
            self.assertTrue(callable(getattr(preflight, function_name, None)),
                            function_name)
            for prerequisite in requires:
                self.assertIn(prerequisite, names)


class GpuProfileTests(unittest.TestCase):
    """Hardware profiles: A40 LoRA versus RTX PRO 6000 full SFT."""

    def device(self, name, total_gib):
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda index: types.SimpleNamespace(
                name=name, total_memory=int(total_gib * 1024 ** 3)))
        return types.SimpleNamespace(cuda=cuda)

    def run_check(self, name, total_gib, **config_overrides):
        ctx = context(config=support.make_config(**config_overrides))
        ctx.torch = self.device(name, total_gib)
        return preflight.check_gpu_identity(ctx)

    def test_a40_profile_accepts_an_a40_with_48_gib(self):
        outcome = self.run_check("NVIDIA A40", 48,
                                 hardware={"profile": "a40-lora"})
        self.assertEqual(outcome["status"], "pass", outcome["detail"])
        self.assertAlmostEqual(outcome["data"]["vram_gib"], 48.0, places=1)

    def test_a40_profile_rejects_a40_below_the_threshold(self):
        outcome = self.run_check("NVIDIA A40", 40,
                                 hardware={"profile": "a40-lora"})
        self.assertEqual(outcome["status"], "fail")
        self.assertIn("min_vram_gib", outcome["detail"])

    def test_a40_profile_rejects_an_rtx_pro_6000(self):
        outcome = self.run_check("NVIDIA RTX PRO 6000 Blackwell Server Edition",
                                 96, hardware={"profile": "a40-lora"})
        self.assertEqual(outcome["status"], "fail")
        self.assertIn("does not match", outcome["detail"])

    def test_rtx_profile_accepts_an_rtx_pro_6000(self):
        outcome = self.run_check("NVIDIA RTX PRO 6000 Blackwell Server Edition",
                                 96)
        self.assertEqual(outcome["status"], "pass", outcome["detail"])

    def test_rtx_profile_rejects_an_a40(self):
        outcome = self.run_check("NVIDIA A40", 48)
        self.assertEqual(outcome["status"], "fail")
        self.assertIn("does not match", outcome["detail"])

    def test_rtx_lora_profile_accepts_an_rtx_pro_6000_at_60_gib(self):
        outcome = self.run_check("NVIDIA RTX PRO 6000 Blackwell Server Edition",
                                 96, hardware={"profile": "rtxpro6000-lora"})
        self.assertEqual(outcome["status"], "pass", outcome["detail"])

    def test_rtx_lora_profile_rejects_an_a40(self):
        outcome = self.run_check("NVIDIA A40", 48,
                                 hardware={"profile": "rtxpro6000-lora"})
        self.assertEqual(outcome["status"], "fail")
        self.assertIn("does not match", outcome["detail"])

    def test_rtx_lora_profile_threshold_is_sixty_gib(self):
        outcome = self.run_check("NVIDIA RTX PRO 6000 Blackwell Server Edition",
                                 60, hardware={"profile": "rtxpro6000-lora"})
        self.assertEqual(outcome["status"], "pass", outcome["detail"])
        outcome = self.run_check("NVIDIA RTX PRO 6000 Blackwell Server Edition",
                                 59, hardware={"profile": "rtxpro6000-lora"})
        self.assertEqual(outcome["status"], "fail")
        self.assertIn("60.0", outcome["detail"])

    def test_a40_profile_explicit_vram_override(self):
        outcome = self.run_check("NVIDIA A40", 46,
                                 hardware={"profile": "a40-lora",
                                           "min_vram_gib": 47.0})
        self.assertEqual(outcome["status"], "fail")
        self.assertIn("47.0", outcome["detail"])


class IndividualCheckTests(unittest.TestCase):
    def test_lazy_unsloth_logits_are_valid_for_loss_only_forward(self):
        lazy_logits = types.SimpleNamespace(shape=lambda: None)
        outputs = types.SimpleNamespace(logits=lazy_logits)
        self.assertIsNone(preflight._materialized_logits_shape(outputs))

    def test_materialized_logits_shape_is_recorded(self):
        outputs = types.SimpleNamespace(
            logits=types.SimpleNamespace(shape=(2, 17, 130560)))
        self.assertEqual(
            preflight._materialized_logits_shape(outputs), (2, 17, 130560))

    def test_direct_forward_explicitly_disables_kv_cache(self):
        calls = []

        class Model:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return "output"

        result = preflight._forward_no_cache(Model(), {"input_ids": [1]})
        self.assertEqual(result, "output")
        self.assertFalse(calls[0]["use_cache"])

    def test_lora_serialization_uses_peft_state_not_raw_model_state(self):
        with tempfile.TemporaryDirectory() as directory:
            class Model:
                def save_pretrained(self, path, safe_serialization=True):
                    Path(path, "adapter_model.safetensors").write_bytes(b"fake")

                def state_dict(self):
                    raise AssertionError("raw state_dict must not verify LoRA")

            saved = {"base_model.model.layer.lora_A.weight": [1, 2]}
            live = {"base_model.model.layer.lora_A.weight": [1, 2]}
            getter_calls = []
            package = types.ModuleType("safetensors")
            package.__path__ = []
            torch_module = types.ModuleType("safetensors.torch")
            torch_module.load_file = lambda path: saved
            ctx = context(directory)
            ctx.model = Model()
            ctx.config.training.method = "lora"
            ctx.torch = types.SimpleNamespace()

            def getter(model):
                getter_calls.append(model)
                return live

            with mock.patch.dict(sys.modules, {
                    "safetensors": package,
                    "safetensors.torch": torch_module}):
                outcome = preflight.check_weight_serialization(
                    ctx, adapter_state_getter=getter)

            self.assertEqual(outcome["status"], "pass")
            self.assertEqual(getter_calls, [ctx.model])
            self.assertEqual(outcome["data"]["adapter_tensors"], 1)

    def test_python_version_check(self):
        import sys
        from unittest import mock
        actual = ".".join(str(part) for part in sys.version_info[:2])
        ctx = context()
        ctx.config.environment.python_version = actual
        with mock.patch.object(preflight, "python_version_pin", return_value=actual):
            self.assertEqual(preflight.check_python_version(ctx)["status"], "pass")
        ctx.config.environment.python_version = "3.4"
        with mock.patch.object(preflight, "python_version_pin", return_value="3.4"):
            self.assertEqual(preflight.check_python_version(ctx)["status"], "fail")

    def test_python_version_check_detects_repo_pin_disagreement(self):
        import sys
        from unittest import mock
        actual = ".".join(str(part) for part in sys.version_info[:2])
        ctx = context()
        ctx.config.environment.python_version = actual
        with mock.patch.object(preflight, "python_version_pin", return_value="3.12"):
            outcome = preflight.check_python_version(ctx)
        self.assertEqual(outcome["status"], "fail")
        self.assertIn(".python-version", outcome["detail"])

    def test_disk_space_check(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = context(directory)
            ctx.config.hardware.min_free_disk_gib = 0.0001
            self.assertEqual(preflight.check_disk_space(ctx)["status"], "pass")
            ctx.config.hardware.min_free_disk_gib = 10 ** 9
            outcome = preflight.check_disk_space(ctx)
            self.assertEqual(outcome["status"], "fail")
            self.assertIn("free", outcome["detail"])

    def test_disk_space_estimates_checkpoints_from_parameter_count(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = context(directory, parameter_count=4_000_000_000)
            ctx.config.hardware.min_free_disk_gib = 0.0001
            outcome = preflight.check_disk_space(ctx)
            self.assertIsNotNone(outcome["data"]["estimate_bytes"])
            # 4B params * 10 bytes * retention 3
            self.assertEqual(outcome["data"]["estimate_bytes"],
                             4_000_000_000 * 10 * 3)

    def test_environment_lock_check(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "test.lock"
            installed = {"pyarrow": "25.0.1", "yaml": "6.0.3", "PIL": "12.3.0"}
            lock.write_text("pyarrow==25.0.1\nPyYAML==6.0.3\npillow==12.3.0\n")
            ctx = context(directory)
            ctx.config.environment.lock_path = lock
            ctx.scratch["installed_versions"] = installed
            self.assertEqual(preflight.check_environment_lock(ctx)["status"], "pass")
            lock.write_text("torch==9.9.9\n")
            outcome = preflight.check_environment_lock(ctx)
            self.assertEqual(outcome["status"], "fail")
            self.assertIn("torch", outcome["data"]["differences"][0])

    def test_environment_lock_check_against_the_real_environment(self):
        # In any environment, an empty/unknown lock must not pass silently.
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "test.lock"
            lock.write_text("torch==9.9.9\n")
            ctx = context(directory)
            ctx.config.environment.lock_path = lock
            outcome = preflight.check_environment_lock(ctx)
            self.assertEqual(outcome["status"], "fail")

    def test_attention_backend_is_informational(self):
        outcome = preflight.check_attention_backend(context())
        self.assertEqual(outcome["status"], "pass")
        self.assertIn("flash_attn", outcome["data"])

    def test_checkpoint_reload_and_resume_fail_without_a_model(self):
        ctx = context()
        ctx.torch = types.SimpleNamespace()  # avoid importing the real torch
        for check in (preflight.check_model_reload,
                      preflight.check_trainer_resume):
            outcome = check(ctx)
            self.assertEqual(outcome["status"], "fail")
            self.assertIn("model", outcome["detail"])

    def test_checkpoint_checks_are_truthfully_named(self):
        names = [name for name, _, _ in preflight.CHECK_SPECS]
        self.assertIn("checkpoint.weight_serialization", names)
        self.assertIn("checkpoint.model_reload", names)
        self.assertIn("checkpoint.trainer_resume", names)
        self.assertNotIn("checkpoint.roundtrip", names)

    def test_missing_lock_path_fails(self):
        outcome = preflight.check_environment_lock(context())
        self.assertEqual(outcome["status"], "fail")


class NoTrainingRuleTests(unittest.TestCase):
    def test_preflight_module_never_imports_or_calls_the_trainer(self):
        source = (STAGE1 / "src" / "stage1" / "preflight.py").read_text()
        self.assertNotIn("run_training", source)
        self.assertNotIn("import stage1.train", source)
        self.assertNotIn("from stage1.train", source)
        self.assertNotIn("from .train", source)

    def test_preflight_script_never_imports_or_calls_the_trainer(self):
        source = (STAGE1 / "scripts" / "preflight.py").read_text()
        self.assertNotIn("run_training", source)
        self.assertNotIn("stage1.train", source)
        self.assertNotIn("train_module", source)


class ReportSchemaTests(unittest.TestCase):
    def test_written_preflight_report_matches_schema(self):
        from stage1 import schema
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            identity.write_preflight_report(
                run_dir, identity_sha256="a" * 64,
                checks=[{"name": "gpu.identity", "status": "pass", "detail": "ok",
                         "data": {}}],
                status="passed", complete=True,
                payload={"peak_vram_bytes": 1, "load_report": None,
                         "failed": [], "skipped": [],
                         "started_training": False})
            payload = json.loads(
                identity.preflight_report_path(run_dir).read_text())
            schema.validate_artifact(payload,
                                     schema.load_schema("preflight.schema.json"),
                                     "preflight report")


if __name__ == "__main__":
    unittest.main()
