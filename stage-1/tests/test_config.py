"""Tests for strict configuration parsing, gates and TrainingArguments mapping."""
import tempfile
import unittest
from pathlib import Path

from tests import support

from stage1 import config as config_module
from stage1.errors import ConfigError

REPO_STAGE1 = Path(__file__).resolve().parents[1]


class ParsingTests(unittest.TestCase):
    def test_defaults_are_the_documented_baseline(self):
        config = support.make_config()
        self.assertEqual(config.seed, 20260923)
        self.assertEqual(config.training.learning_rate, 1e-5)
        self.assertEqual(config.training.lr_scheduler_type, "cosine")
        self.assertEqual(config.training.warmup_ratio, 0.03)
        self.assertEqual(config.training.max_grad_norm, 1.0)
        self.assertTrue(config.training.bf16)
        self.assertTrue(config.training.gradient_checkpointing)
        self.assertEqual(config.training.optim, "adamw_torch_fused")
        self.assertEqual(config.training.epochs, 1)
        self.assertEqual(config.data.splits.validation_fraction, 0.01)
        self.assertFalse(hasattr(config.training, "packing"))

    def test_unknown_top_level_key_is_rejected(self):
        raw = support.config_dict()
        raw["surprise"] = 1
        with self.assertRaisesRegex(ConfigError, "unknown key"):
            config_module.parse_config(raw)

    def test_unknown_nested_key_is_rejected(self):
        raw = support.config_dict()
        raw["training"]["turbo"] = True
        with self.assertRaisesRegex(ConfigError, "training: unknown key"):
            config_module.parse_config(raw)

    def test_branch_revision_is_rejected(self):
        with self.assertRaisesRegex(ConfigError, "40-character commit SHA"):
            support.make_config(model={"revision": "main"})

    def test_model_id_must_be_owner_slash_name(self):
        with self.assertRaisesRegex(ConfigError, "owner/name"):
            support.make_config(model={"id": "not-a-repo"})

    def test_unknown_adapter_and_loader(self):
        with self.assertRaisesRegex(ConfigError, "adapter"):
            support.make_config(model={"adapter": "mystery"})
        with self.assertRaisesRegex(ConfigError, "loader"):
            support.make_config(model={"loader": "magic"})

    def test_relative_paths_are_rejected(self):
        with self.assertRaisesRegex(ConfigError, "absolute"):
            support.make_config(data={"export_dir": "relative/export"})
        with self.assertRaisesRegex(ConfigError, "absolute"):
            support.make_config(model={"local_path": "relative/model"})

    def test_split_fractions_must_sum_below_one(self):
        with self.assertRaisesRegex(ConfigError, "must be < 1"):
            support.make_config(data={"splits": {"validation_fraction": 0.5,
                                                 "test_fraction": 0.5}})

    def test_learning_rate_bounds(self):
        with self.assertRaises(ConfigError):
            support.make_config(training={"learning_rate": 0.0})
        with self.assertRaises(ConfigError):
            support.make_config(training={"learning_rate": 1.0})

    def test_scheduler_is_pinned_to_cosine(self):
        with self.assertRaisesRegex(ConfigError, "cosine"):
            support.make_config(training={"lr_scheduler_type": "linear"})

    def test_bf16_is_required(self):
        with self.assertRaisesRegex(ConfigError, "BF16"):
            support.make_config(training={"bf16": False})

    def test_optimizer_choices(self):
        with self.assertRaisesRegex(ConfigError, "optim"):
            support.make_config(training={"optim": "lomo"})
        config = support.make_config(training={"optim": "adamw_torch"})
        self.assertEqual(config.training.optim, "adamw_torch")

    def test_packing_true_is_rejected_with_clear_error(self):
        with self.assertRaisesRegex(ConfigError, "not supported"):
            support.make_config(training={"packing": True})
        with self.assertRaisesRegex(ConfigError, "right-padded only"):
            support.make_config(training={"packing": True})

    def test_packing_false_is_accepted(self):
        config = support.make_config(training={"packing": False})
        self.assertFalse(hasattr(config.training, "packing"))

    def test_max_seq_len_lower_bound(self):
        with self.assertRaises(ConfigError):
            support.make_config(data={"max_seq_len": 64})

    def test_unknown_gate_and_gate_override(self):
        with self.assertRaisesRegex(ConfigError, "unknown gate"):
            support.make_config(gates={"tiny": {"max_rows": 1}})
        with self.assertRaisesRegex(ConfigError, "unknown key"):
            support.make_config(gates={"full": {"learning_rate": 1.0}})

    def test_lock_file_is_required(self):
        raw = support.config_dict()
        raw["environment"] = {}
        with self.assertRaisesRegex(ConfigError, "lock_file"):
            config_module.parse_config(raw)

    def test_python_version_format(self):
        with self.assertRaisesRegex(ConfigError, "3.x"):
            support.make_config(environment={"python_version": "3"})


class GateTests(unittest.TestCase):
    def test_overfit_gate_defaults(self):
        gate = support.make_config().gate("overfit-100")
        self.assertEqual(gate.max_rows, 100)
        self.assertEqual(gate.epochs, 40)
        self.assertEqual(gate.success_eval_loss, 0.05)
        self.assertEqual(gate.min_loss_reduction, 0.9)
        self.assertEqual(gate.save_steps, 50)
        self.assertEqual(gate.eval_steps, 50)
        self.assertEqual(gate.generation_samples, 1)
        self.assertTrue(gate.check_gradients)

    def test_smoke_and_full_gates(self):
        config = support.make_config()
        smoke = config.gate("smoke-1000")
        self.assertEqual(smoke.max_rows, 1000)
        self.assertEqual(smoke.save_steps, 25)
        self.assertEqual(smoke.eval_steps, 25)
        self.assertTrue(smoke.check_gradients)
        self.assertEqual(smoke.generation_samples, 2)
        full = config.gate("full")
        self.assertIsNone(full.max_rows)
        self.assertEqual(full.epochs, 1)
        self.assertIsNone(full.success_eval_loss)
        self.assertFalse(full.check_gradients)

    def test_gate_override_beats_training_default(self):
        config = support.make_config(gates={"full": {"save_steps": 7}})
        self.assertEqual(config.gate("full").save_steps, 7)
        self.assertEqual(config.gate("full").eval_steps, 200)

    def test_new_gate_override_keys_validate(self):
        config = support.make_config(gates={"smoke-1000": {
            "min_loss_reduction": 0.5, "generation_samples": 3,
            "check_gradients": False}})
        gate = config.gate("smoke-1000")
        self.assertEqual(gate.min_loss_reduction, 0.5)
        self.assertEqual(gate.generation_samples, 3)
        self.assertFalse(gate.check_gradients)
        with self.assertRaises(ConfigError):
            support.make_config(gates={"smoke-1000": {
                "min_loss_reduction": 2.0}}).gate("smoke-1000")
        with self.assertRaises(ConfigError):
            support.make_config(gates={"smoke-1000": {
                "generation_samples": 0}}).gate("smoke-1000")

    def test_unknown_training_gate_is_rejected(self):
        with self.assertRaisesRegex(ConfigError, "Unknown gate"):
            support.make_config().gate("preflight")


class TrainingArgumentsTests(unittest.TestCase):
    def test_mapping_contains_the_required_optimizer_settings(self):
        config = support.make_config()
        gate = config.gate("full")
        kwargs = config_module.training_arguments_kwargs(
            config, gate, run_dir=Path("/tmp/run"), has_eval=True)
        self.assertEqual(kwargs["learning_rate"], 1e-5)
        self.assertEqual(kwargs["lr_scheduler_type"], "cosine")
        self.assertEqual(kwargs["warmup_ratio"], 0.03)
        self.assertEqual(kwargs["max_grad_norm"], 1.0)
        self.assertTrue(kwargs["bf16"])
        self.assertTrue(kwargs["gradient_checkpointing"])
        self.assertEqual(kwargs["optim"], "adamw_torch_fused")
        self.assertEqual(kwargs["num_train_epochs"], 1)
        self.assertEqual(kwargs["seed"], 20260923)
        self.assertEqual(kwargs["data_seed"], 20260923)
        self.assertEqual(kwargs["eval_strategy"], "steps")
        self.assertEqual(kwargs["save_steps"], 200)
        self.assertEqual(kwargs["eval_steps"], 200)
        self.assertFalse(kwargs["remove_unused_columns"])
        self.assertEqual(kwargs["label_names"], ["labels"])
        self.assertTrue(kwargs["include_num_input_tokens_seen"])
        self.assertEqual(kwargs["output_dir"], "/tmp/run/checkpoints/full")

    def test_mapping_reflects_gate_frequencies_and_no_eval(self):
        config = support.make_config()
        gate = config.gate("overfit-100")
        kwargs = config_module.training_arguments_kwargs(
            config, gate, run_dir=Path("/tmp/run"), has_eval=False)
        self.assertEqual(kwargs["num_train_epochs"], 40)
        self.assertEqual(kwargs["eval_steps"], 50)
        self.assertEqual(kwargs["logging_steps"], 5)
        self.assertEqual(kwargs["eval_strategy"], "no")
        self.assertFalse(kwargs["do_eval"])

    def test_checkpoint_namespace_is_per_gate(self):
        config = support.make_config()
        for gate_name in ("overfit-100", "smoke-1000", "full"):
            gate = config.gate(gate_name)
            kwargs = config_module.training_arguments_kwargs(
                config, gate, run_dir=Path("/tmp/run"), has_eval=True)
            self.assertEqual(kwargs["output_dir"],
                             f"/tmp/run/checkpoints/{gate_name}")


class IdentityPayloadTests(unittest.TestCase):
    def test_paths_are_excluded_from_the_identity(self):
        config = support.make_config(
            data={"export_dir": "/data/export", "prepared_dir": "/data/prepared"},
            model={"local_path": "/models/local"})
        payload = config.identity_payload()
        serialized = repr(payload)
        self.assertNotIn("/data/export", serialized)
        self.assertNotIn("/data/prepared", serialized)
        self.assertNotIn("/models/local", serialized)
        self.assertEqual(payload["model"]["revision"], "a" * 40)
        self.assertEqual(payload["training"]["learning_rate"], 1e-5)

    def test_identity_payload_requires_lock_file(self):
        config = support.make_config()
        config.environment.lock_file = None
        with self.assertRaises(ConfigError):
            config.identity_payload()


class ShippedConfigTests(unittest.TestCase):
    @support.requires_yaml
    def test_common_yaml_alone_requires_a_model(self):
        with self.assertRaisesRegex(ConfigError, "model.id"):
            config_module.load_config(REPO_STAGE1 / "configs" / "common.yaml")

    @support.requires_yaml
    def test_qwen_config_resolves(self):
        config = config_module.load_config(
            REPO_STAGE1 / "configs" / "qwen3.5-4b-full.yaml")
        self.assertEqual(config.model.id, "Qwen/Qwen3.5-4B")
        self.assertEqual(config.model.revision,
                         "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
        self.assertEqual(config.model.adapter, "qwen3.5")
        self.assertEqual(config.model.loader, "unsloth-vision-model")
        self.assertEqual(config.environment.lock_path.name, "qwen3.5-4b.lock")
        self.assertIsNotNone(config.environment.lock_sha256)
        self.assertEqual(config.data.max_seq_len, 4096)
        self.assertEqual(config.gate("full").max_rows, None)

    @support.requires_yaml
    def test_minicpm_config_resolves(self):
        config = config_module.load_config(
            REPO_STAGE1 / "configs" / "minicpm5-2b-full.yaml")
        self.assertEqual(config.model.id, "openbmb/MiniCPM5-2B")
        self.assertEqual(config.model.revision,
                         "12a3808a956f869c767195e9266b59c4d21d92e2")
        self.assertEqual(config.model.adapter, "minicpm5")
        self.assertEqual(config.model.loader, "unsloth-language-model")
        self.assertEqual(config.environment.lock_path.name, "minicpm5-2b.lock")

    @support.requires_yaml
    def test_adapter_context_length_covers_sequence_length(self):
        from stage1 import adapters
        for name in ("qwen3.5-4b-full.yaml", "minicpm5-2b-full.yaml"):
            config = config_module.load_config(REPO_STAGE1 / "configs" / name)
            adapter = adapters.validate_config_against_adapter(config)
            self.assertLessEqual(config.data.max_seq_len, adapter.context_length)

    @support.requires_yaml
    def test_shipped_configs_share_identical_identity_shape(self):
        first = config_module.load_config(
            REPO_STAGE1 / "configs" / "qwen3.5-4b-full.yaml").identity_payload()
        second = config_module.load_config(
            REPO_STAGE1 / "configs" / "minicpm5-2b-full.yaml").identity_payload()
        self.assertEqual(set(first), set(second))
        self.assertNotEqual(first["model"], second["model"])
        self.assertNotIn("packing", first["training"])

    @support.requires_yaml
    def test_shipped_gate_intervals_are_reachable(self):
        """Regression: the smoke gate must be able to write a checkpoint."""
        from stage1 import train as train_module
        for name in ("qwen3.5-4b-full.yaml", "minicpm5-2b-full.yaml"):
            config = config_module.load_config(REPO_STAGE1 / "configs" / name)
            examples = {"overfit-100": 100, "smoke-1000": 1000, "full": 100000}
            for gate_name, count in examples.items():
                with self.subTest(config=name, gate=gate_name):
                    gate = config.gate(gate_name)
                    bounds = train_module.optimizer_step_bounds(
                        examples=count,
                        per_device_batch_size=config.training.per_device_train_batch_size,
                        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                        epochs=gate.epochs, max_steps=gate.max_steps)
                    report = train_module.validate_gate_intervals(
                        gate, bounds, has_eval=True)
                    self.assertTrue(report["save_within_schedule"])
                    self.assertTrue(report["eval_within_schedule"])

    @support.requires_yaml
    def test_shipped_configs_reject_packing(self):
        raw = support.config_dict(training={"packing": True})
        with self.assertRaisesRegex(ConfigError, "not supported"):
            config_module.parse_config(raw)


class PathResolutionTests(unittest.TestCase):
    def test_run_paths_require_absolute_paths(self):
        config = support.make_config()
        with self.assertRaisesRegex(ConfigError, "absolute"):
            config_module.resolve_run_paths(config, export="rel", prepared="/p",
                                            run_dir="/r")

    def test_cli_overrides_config_paths(self):
        config = support.make_config(
            data={"export_dir": "/from/config/export",
                  "prepared_dir": "/from/config/prepared"})
        paths = config_module.resolve_run_paths(
            config, export="/cli/export", prepared=None, run_dir="/cli/run")
        self.assertEqual(str(paths.export_dir), "/cli/export")
        self.assertEqual(str(paths.prepared_dir), "/from/config/prepared")
        self.assertEqual(str(paths.run_dir), "/cli/run")

    def test_resolve_aux_file_prefers_config_dir_then_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "locks").mkdir()
            (root / "locks" / "a.lock").write_text("torch==1\n")
            config_path = root / "configs" / "model.yaml"
            config_path.parent.mkdir()
            config_path.write_text("model: {}\n")
            self.assertEqual(
                config_module.resolve_aux_file(config_path, "locks/a.lock"),
                (root / "locks" / "a.lock").resolve())
            with self.assertRaisesRegex(ConfigError, "Cannot resolve"):
                config_module.resolve_aux_file(config_path, "locks/missing.lock")


if __name__ == "__main__":
    unittest.main()
