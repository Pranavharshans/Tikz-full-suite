"""Dependency-free tests for the extensible native Unsloth path."""
import json
import tempfile
import types
import unittest
from pathlib import Path

from unsloth_native import cli, models, runner
from stage1.errors import DataError


class NativeRegistryTests(unittest.TestCase):
    def test_built_in_models_use_the_expected_unsloth_loaders(self):
        self.assertEqual(models.get_spec("minicpm5-2b").loader_name,
                         "FastLanguageModel")
        self.assertEqual(models.get_spec("qwen3.5-4b").loader_name,
                         "FastVisionModel")

    def test_every_method_config_exists(self):
        for spec in models.MODEL_SPECS.values():
            for method in ("lora", "full"):
                with self.subTest(model=spec.key, method=method):
                    self.assertTrue(spec.config_path(method).is_file())

    def test_unknown_model_is_refused(self):
        with self.assertRaisesRegex(DataError, "Unknown native model"):
            models.get_spec("missing")


class NativeCliTests(unittest.TestCase):
    def test_model_cli_defaults_to_lora_and_native_resume(self):
        args = cli.parser("minicpm5-2b").parse_args([
            "--export", "/export", "--prepared", "/prepared",
            "--run-dir", "/run", "--gate", "overfit-100"])
        self.assertEqual(args.method, "lora")
        self.assertEqual(args.resume, "auto")

    def test_full_method_selects_full_config(self):
        spec = models.get_spec("qwen3.5-4b")
        self.assertEqual(spec.config_path("full").name,
                         "qwen3.5-4b-full.yaml")


class NativeIdentityTests(unittest.TestCase):
    def test_raw_gate_overrides_are_resolved_to_typed_gate(self):
        typed = types.SimpleNamespace(name="overfit-100", max_rows=100)
        config = types.SimpleNamespace(
            gates={"overfit-100": {"max_rows": 100}},
            gate=lambda name: typed if name == "overfit-100" else None)
        self.assertIs(runner._resolve_gate(config, "overfit-100"), typed)

    def test_missing_gate_is_refused_before_resolution(self):
        config = types.SimpleNamespace(gates={}, gate=lambda name: None)
        with self.assertRaisesRegex(DataError, "has no gate"):
            runner._resolve_gate(config, "overfit-100")

    def identity(self):
        config = types.SimpleNamespace(
            model=types.SimpleNamespace(id="test/model", revision="a" * 40,
                                        adapter="minicpm5"),
            training=types.SimpleNamespace(method="lora"),
            data=types.SimpleNamespace(max_seq_len=8192), seed=7,
            to_jsonable=lambda: {"model": "test/model", "seed": 7})
        gate = types.SimpleNamespace(name="overfit-100")
        datasets = types.SimpleNamespace(
            prepared={"data_identity_sha256": "d" * 64,
                      "report_sha256": "r" * 64},
            export=types.SimpleNamespace(dataset_logical_sha256="e" * 64))
        fingerprint = {
            "model_id": "test/model", "revision": "a" * 40,
            "adapter": "minicpm5", "tokenizer_class": "Fast",
            "vocab_size": 42, "pad_token_id": 0,
            "chat_template_sha256": "c" * 64,
            "chat_template_source": "ignored", "files": {},
        }
        return runner._native_identity(
            models.get_spec("minicpm5-2b"), config, gate, datasets, fingerprint)

    def test_identity_is_stable_and_self_hashed(self):
        first, second = self.identity(), self.identity()
        self.assertEqual(first, second)
        self.assertEqual(len(first["sha256"]), 64)

    def test_existing_foreign_run_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner._ensure_identity(root, self.identity())
            written = json.loads((root / "native-run.json").read_text())
            self.assertEqual(written, self.identity())
            changed = dict(self.identity(), gate="full")
            with self.assertRaisesRegex(DataError, "identity mismatch"):
                runner._ensure_identity(root, changed)

    def test_explicit_resume_path_must_exist(self):
        with self.assertRaisesRegex(DataError, "does not exist"):
            runner._resume_checkpoint(Path("/tmp/run"), "/definitely/missing")
        self.assertIsNone(runner._resume_checkpoint(Path("/tmp/run"), "none"))

    def test_resume_is_confined_to_the_identified_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            foreign = Path(directory) / "foreign" / "checkpoint-1"
            foreign.mkdir(parents=True)
            with self.assertRaisesRegex(DataError, "must belong"):
                runner._resume_checkpoint(run, str(foreign))

    def test_clean_start_refuses_existing_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "checkpoints" / "checkpoint-1").mkdir(parents=True)
            with self.assertRaisesRegex(DataError, "use --resume auto"):
                runner._resume_checkpoint(run, "none")


if __name__ == "__main__":
    unittest.main()
