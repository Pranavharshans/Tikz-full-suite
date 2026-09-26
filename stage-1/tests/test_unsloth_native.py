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

    def test_text_tokenizer_accepts_a_tokenizer_directly(self):
        tokenizer = types.SimpleNamespace(__len__=lambda self: 42)
        self.assertIs(models._text_tokenizer(tokenizer), tokenizer)

    def test_text_tokenizer_unwraps_a_vision_processor(self):
        tokenizer = object()
        processor = types.SimpleNamespace(tokenizer=tokenizer)
        self.assertIs(models._text_tokenizer(processor), tokenizer)

    def test_text_tokenizer_refuses_an_unknown_loader_result(self):
        with self.assertRaisesRegex(DataError, "neither a tokenizer nor a processor"):
            models._text_tokenizer(object())


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
    def test_sft_arguments_allows_transformers_five_without_overwrite_flag(self):
        captured = {}

        def fake_sft_config(output_dir, do_train, do_eval, eval_strategy,
                            eval_steps, save_strategy, save_steps,
                            save_total_limit, logging_steps, logging_first_step,
                            learning_rate, lr_scheduler_type, warmup_ratio,
                            max_grad_norm, num_train_epochs, max_steps,
                            per_device_train_batch_size,
                            per_device_eval_batch_size,
                            gradient_accumulation_steps, optim, bf16,
                            bf16_full_eval, gradient_checkpointing,
                            gradient_checkpointing_kwargs, seed, data_seed,
                            report_to, run_name,
                            include_num_input_tokens_seen,
                            dataloader_num_workers, remove_unused_columns,
                            label_names, packing, dataset_kwargs, max_length,
                            assistant_only_loss):
            captured.update(locals())
            return captured

        config = types.SimpleNamespace(
            model=types.SimpleNamespace(adapter="qwen3.5"),
            data=types.SimpleNamespace(max_seq_len=8192),
        )
        gate = types.SimpleNamespace(name="overfit-100")
        values = {
            "output_dir": "/old", "overwrite_output_dir": False,
            "do_train": True, "do_eval": True, "eval_strategy": "steps",
            "eval_steps": 5, "save_strategy": "steps", "save_steps": 5,
            "save_total_limit": 2, "logging_steps": 1,
            "logging_first_step": True, "learning_rate": 1e-4,
            "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
            "max_grad_norm": 1.0, "num_train_epochs": 1,
            "max_steps": -1, "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 8, "optim": "adamw_torch_fused",
            "bf16": True, "bf16_full_eval": True,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "seed": 3407, "data_seed": 3407, "report_to": [],
            "run_name": "native", "include_num_input_tokens_seen": True,
            "dataloader_num_workers": 0, "remove_unused_columns": False,
            "label_names": ["labels"],
        }
        original = runner.config_module.training_arguments_kwargs
        runner.config_module.training_arguments_kwargs = lambda *a, **k: dict(values)
        try:
            result = runner._sft_arguments(
                fake_sft_config, config, gate, Path("/run"), has_eval=True)
        finally:
            runner.config_module.training_arguments_kwargs = original
        self.assertNotIn("overwrite_output_dir", result)
        self.assertEqual(result["output_dir"], "/run/checkpoints")

    def test_full_epoch_acceptance_requires_the_configured_epoch(self):
        gate = types.SimpleNamespace(epochs=1.0)
        self.assertFalse(runner._epoch_completed(
            types.SimpleNamespace(epoch=0.999), gate))
        self.assertTrue(runner._epoch_completed(
            types.SimpleNamespace(epoch=1.0), gate))

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

    def test_resume_refuses_a_foreign_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            foreign_root = Path(directory) / "foreign"
            foreign = foreign_root / "checkpoints" / "checkpoint-1"
            foreign.mkdir(parents=True)
            (foreign_root / "native-run.json").write_text(
                json.dumps(dict(self.identity(), gate="full")))
            with self.assertRaisesRegex(DataError, "identity does not match"):
                runner._resume_checkpoint(
                    run, str(foreign), identity=self.identity())

    def test_resume_allows_a_matching_external_native_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "resume-drill"
            source = Path(directory) / "smoke"
            checkpoint = source / "checkpoints" / "checkpoint-50"
            checkpoint.mkdir(parents=True)
            (source / "native-run.json").write_text(json.dumps(self.identity()))
            self.assertEqual(
                runner._resume_checkpoint(
                    run, str(checkpoint), identity=self.identity()),
                str(checkpoint.resolve()))

    def test_clean_start_refuses_existing_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "checkpoints" / "checkpoint-1").mkdir(parents=True)
            with self.assertRaisesRegex(DataError, "use --resume auto"):
                runner._resume_checkpoint(run, "none")


if __name__ == "__main__":
    unittest.main()
