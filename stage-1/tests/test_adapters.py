"""Dependency-free tests for the LoRA adapter layer.

The loading tests install fake ``torch``/``unsloth`` modules so the real
loader functions execute end to end without Unsloth: they prove the base-only
loader never calls ``get_peft_model``, the LoRA loader calls it exactly once,
and the adapter attachment refuses a second adapter. The pure helpers
(target validation, trainable assertions) are tested directly. Real PEFT round
trips live in ``test_lora_integration.py``.
"""
import sys
import tempfile
import types
import unittest
from unittest import mock

from tests import support

from stage1 import adapters
from stage1.errors import DataError


class FakeEmbedding:
    def __init__(self):
        self.weight = types.SimpleNamespace(dtype="bfloat16")


class FakeParameter:
    def __init__(self, numel, requires_grad):
        self._numel = numel
        self.requires_grad = requires_grad

    def numel(self):
        return self._numel


TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")


class FakeModel:
    """A tiny stand-in with the attributes the loaders and checks use."""

    def __init__(self, *, with_adapter=False):
        self.with_adapter = with_adapter
        self.named = [
            ("base.layers.0.self_attn.q_proj.weight", FakeParameter(64, False)),
            ("base.layers.0.self_attn.q_proj.bias", FakeParameter(8, False)),
        ]
        if with_adapter:
            for target in TARGETS:
                self.named.append(
                    (f"base.layers.0.{target}.lora_A.default.weight",
                     FakeParameter(4, True)))
                self.named.append(
                    (f"base.layers.0.{target}.lora_B.default.weight",
                     FakeParameter(4, True)))
        self.saved_pretrained = []

    def get_input_embeddings(self):
        return FakeEmbedding()

    def named_modules(self):
        yield "", self
        for target in TARGETS:
            yield f"base.layers.0.{target}", object()

    def parameters(self):
        for _, parameter in self.named:
            yield parameter

    def named_parameters(self):
        for name, parameter in self.named:
            yield name, parameter

    def state_dict(self):
        return {name: types.SimpleNamespace(shape=(2,)) for name, _ in self.named}

    def save_pretrained(self, directory, **kwargs):
        self.saved_pretrained.append(directory)

    def train(self):
        return self

    def eval(self):
        return self


class FakeFastLanguageModel:
    def __init__(self):
        self.load_calls = []
        self.peft_calls = []

    def from_pretrained(self, model_name, **kwargs):
        self.load_calls.append({"model_name": model_name, "kwargs": dict(kwargs)})
        return FakeModel(), types.SimpleNamespace(pad_token_id=0)

    def get_peft_model(self, model, r, lora_alpha, lora_dropout, bias,
                       target_modules, random_state=None,
                       use_gradient_checkpointing=None):
        self.peft_calls.append({
            "model": model, "r": r, "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout, "bias": bias,
            "target_modules": list(target_modules),
            "use_gradient_checkpointing": use_gradient_checkpointing})
        return FakeModel(with_adapter=True)

    def for_training(self, model):
        return model

    def for_inference(self, model):
        return model


class FakeLoaderModule:
    def __init__(self):
        self.FastLanguageModel = FakeFastLanguageModel()


def fake_modules():
    loader = FakeLoaderModule()
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    unsloth = types.ModuleType("unsloth")
    unsloth.FastLanguageModel = loader.FastLanguageModel
    unsloth.__version__ = "fake-1.0"
    return loader, {"torch": torch, "unsloth": unsloth}


class BaseOnlyLoaderTests(unittest.TestCase):
    def setUp(self):
        self.loader, self.modules = fake_modules()
        self.config = support.make_config(training={"method": "lora"}, lora={})

    def test_base_loader_never_creates_an_adapter(self):
        with mock.patch.dict(sys.modules, self.modules):
            model, tokenizer, report = adapters.load_base_model_and_tokenizer(
                self.config)
        self.assertEqual(self.loader.FastLanguageModel.peft_calls, [])
        self.assertEqual(len(self.loader.FastLanguageModel.load_calls), 1)
        self.assertEqual(report["training_method"], "base")
        self.assertFalse(report["adapter_applied"])
        self.assertEqual(report["quantization"], "none")
        self.assertFalse(report["load_in_4bit"])
        self.assertFalse(report["load_in_8bit"])

    def test_local_only_model_load_uses_resolved_snapshot_path(self):
        hub = types.ModuleType("huggingface_hub")
        calls = []

        def snapshot_download(**kwargs):
            calls.append(dict(kwargs))
            return "/cache/pinned-snapshot"

        hub.snapshot_download = snapshot_download
        modules = dict(self.modules)
        modules["huggingface_hub"] = hub
        with mock.patch.dict(sys.modules, modules):
            adapters.load_base_model_and_tokenizer(
                self.config, local_files_only=True, cache_dir="/cache")

        self.assertEqual(calls, [{
            "repo_id": self.config.model.id,
            "revision": self.config.model.revision,
            "cache_dir": "/cache",
            "local_files_only": True,
        }])
        load = self.loader.FastLanguageModel.load_calls[0]
        self.assertEqual(load["model_name"], "/cache/pinned-snapshot")
        self.assertNotIn("revision", load["kwargs"])
        self.assertTrue(load["kwargs"]["local_files_only"])

    def test_lora_loader_creates_exactly_one_adapter(self):
        with mock.patch.dict(sys.modules, self.modules):
            model, tokenizer, report = adapters.load_model_and_tokenizer(
                self.config)
        self.assertEqual(len(self.loader.FastLanguageModel.peft_calls), 1)
        call = self.loader.FastLanguageModel.peft_calls[0]
        self.assertEqual(call["r"], 64)
        self.assertEqual(call["lora_alpha"], 64)
        self.assertEqual(call["bias"], "none")
        self.assertEqual(call["target_modules"], list(TARGETS))
        self.assertEqual(call["use_gradient_checkpointing"], "unsloth")
        self.assertEqual(report["training_method"], "lora")
        self.assertTrue(report["adapter_applied"])
        self.assertEqual(report["target_modules"]["total_matched_modules"],
                         len(TARGETS))
        self.assertTrue(report["base_frozen"])

    def test_model_loader_applies_configured_padding_token(self):
        self.config.tokenizer.pad_token = "<unused_token_477>"
        with mock.patch.dict(sys.modules, self.modules):
            _, tokenizer, _ = adapters.load_base_model_and_tokenizer(self.config)
        self.assertEqual(tokenizer.pad_token, "<unused_token_477>")
        self.assertEqual(tokenizer.name_or_path, self.config.model.id)

    def test_lora_loader_honors_gradient_checkpointing_false(self):
        loader, modules = fake_modules()
        config = support.make_config(
            training={"method": "lora", "gradient_checkpointing": False}, lora={})
        with mock.patch.dict(sys.modules, modules):
            adapters.load_model_and_tokenizer(config)
        call = loader.FastLanguageModel.peft_calls[0]
        self.assertIs(call["use_gradient_checkpointing"], False)

    def test_base_loader_reports_missing_targets_before_adapting(self):
        loader, modules = fake_modules()

        class NoTargets(FakeModel):
            def named_modules(self):
                yield "", self
                yield "base.layers.0.attention.wq", object()

        loader.FastLanguageModel.from_pretrained = lambda model_name, **kwargs: (
            NoTargets(), types.SimpleNamespace(pad_token_id=0))
        modules["unsloth"].FastLanguageModel = loader.FastLanguageModel
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(DataError, "not found in the loaded model"):
                adapters.load_model_and_tokenizer(self.config)
        self.assertFalse(loader.FastLanguageModel.peft_calls)

    def test_attach_refuses_a_model_that_already_has_an_adapter(self):
        model = FakeModel()
        model.peft_config = {"default": object()}
        with self.assertRaisesRegex(DataError, "already carries a PEFT adapter"):
            adapters.attach_lora_adapter(model, "/tmp/does-not-matter")

    def test_attach_can_request_a_trainable_saved_adapter(self):
        calls = []

        class FakePeftModel:
            @classmethod
            def from_pretrained(cls, model, directory, **kwargs):
                calls.append((model, directory, kwargs))
                return "attached"

        peft = types.ModuleType("peft")
        peft.PeftModel = FakePeftModel
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(sys.modules, {"peft": peft}):
                attached = adapters.attach_lora_adapter(
                    FakeModel(), directory, is_trainable=True)
        self.assertEqual(attached, "attached")
        self.assertEqual(calls[0][2], {"is_trainable": True})


class CheckpointingModule:
    """Tiny gradient-checkpointing module stand-in."""

    def __init__(self, *, with_function=True):
        self.gradient_checkpointing = False
        if with_function:
            self._gradient_checkpointing_func = lambda *args, **kwargs: None


class TrainingLifecycleTests(unittest.TestCase):
    """The reloaded-adapter training lifecycle: Unsloth patch, for_training,
    gradient_checkpointing_enable, and a hard pre-forward diagnostic.

    These are the dependency-free twins of the real-PEFT and GPU/Unsloth
    regressions; they prove the call order and the refusal, not the kernels.
    """

    def setUp(self):
        self.config = support.make_config(training={"method": "lora"}, lora={})

    def fake_loader(self, calls, modules):
        class Loader:
            @staticmethod
            def patch_peft_model(model, use_gradient_checkpointing="unsloth"):
                calls.append(("patch_peft_model", use_gradient_checkpointing))
                for module in modules:
                    module.gradient_checkpointing = True
                    module._gradient_checkpointing_func = lambda *a, **k: None
                return model

            @staticmethod
            def for_training(model, use_gradient_checkpointing=True):
                calls.append(("for_training", use_gradient_checkpointing))
                for module in modules:
                    module.gradient_checkpointing = use_gradient_checkpointing
                return model

        return Loader

    def test_reload_restores_the_full_training_lifecycle_in_order(self):
        calls = []
        modules = [CheckpointingModule() for _ in range(2)]
        Loader = self.fake_loader(calls, modules)

        class Model:
            peft_config = {"default": object()}
            _require_grads_hook = None
            # Unsloth marks every model it loads; patch_peft_model relies on it.
            max_seq_length = 256

            def named_modules(self):
                yield "model.layers.0", modules[0]
                yield "model.layers.1", modules[1]

            def gradient_checkpointing_enable(
                    self, gradient_checkpointing_kwargs=None):
                calls.append(("gradient_checkpointing_enable",
                              dict(gradient_checkpointing_kwargs)))
                for module in modules:
                    module.gradient_checkpointing = True
                    module._gradient_checkpointing_func = lambda *a, **k: None

            def enable_input_require_grads(self):
                calls.append(("enable_input_require_grads",))
                self._require_grads_hook = object()

        model = Model()
        report = {}
        prepared = adapters.prepare_model_for_training(
            model, self.config, loader=Loader, report=report)
        self.assertIs(prepared, model)
        self.assertEqual(
            [call[0] for call in calls],
            ["patch_peft_model", "for_training",
             "gradient_checkpointing_enable", "enable_input_require_grads"])
        self.assertEqual(calls[0][1], "unsloth")
        self.assertIs(calls[1][1], True)
        self.assertEqual(calls[2][1], {"use_reentrant": False})
        self.assertEqual(report["training_state"]["peft_training_state_api"],
                         "FastLanguageModel.patch_peft_model")
        self.assertEqual(
            report["training_state"]["modules_with_gradient_checkpointing"], 2)
        self.assertEqual(
            report["training_state"]["modules_missing_checkpoint_function"], 0)

    def test_gradient_checkpointing_false_is_passed_through(self):
        config = support.make_config(
            training={"method": "lora", "gradient_checkpointing": False}, lora={})
        calls = []
        modules = [CheckpointingModule()]

        class Loader:
            @staticmethod
            def for_training(model, use_gradient_checkpointing=True):
                calls.append(("for_training", use_gradient_checkpointing))
                for module in modules:
                    module.gradient_checkpointing = use_gradient_checkpointing
                return model

        model = FakeModel(with_adapter=True)
        model.peft_config = {"default": object()}
        model.named_modules = lambda: iter([("model.layers.0", modules[0])])
        # No gradient_checkpointing_enable must be required when it is disabled.
        adapters.prepare_model_for_training(model, config, loader=Loader)
        self.assertEqual([call[0] for call in calls], ["for_training"])
        self.assertIs(calls[0][1], False)

    def test_plain_model_skips_patch_peft_model_but_restores_checkpointing(self):
        calls = []
        modules = [CheckpointingModule()]

        class Loader:
            @staticmethod
            def patch_peft_model(model, use_gradient_checkpointing="unsloth"):
                calls.append("patch_peft_model")
                return model

            @staticmethod
            def for_training(model, use_gradient_checkpointing=True):
                calls.append("for_training")
                for module in modules:
                    module.gradient_checkpointing = use_gradient_checkpointing
                return model

        class Model:
            peft_config = {"default": object()}
            _require_grads_hook = None

            def named_modules(self):
                yield "model.layers.0", modules[0]

            def gradient_checkpointing_enable(
                    self, gradient_checkpointing_kwargs=None):
                calls.append("gradient_checkpointing_enable")
                modules[0]._gradient_checkpointing_func = lambda *a, **k: None

            def enable_input_require_grads(self):
                self._require_grads_hook = object()

        report = {}
        adapters.prepare_model_for_training(
            Model(), self.config, loader=Loader, report=report)
        self.assertEqual(
            calls, ["for_training", "gradient_checkpointing_enable"])
        self.assertIsNone(report["training_state"]["peft_training_state_api"])
        self.assertIn("not loaded through the Unsloth Fast* loader",
                      report["training_state"]["peft_training_state_skipped"][0])

    def test_incomplete_checkpointing_state_is_refused_before_a_forward(self):
        module = CheckpointingModule(with_function=False)
        module.gradient_checkpointing = True

        class Model:
            def named_modules(self):
                yield "model.layers.0", module

        with self.assertRaisesRegex(DataError, "_gradient_checkpointing_func"):
            adapters.require_gradient_checkpointing_ready(
                Model(), context="the test training forward")

    def test_no_checkpointing_module_is_refused(self):
        class Model:
            def named_modules(self):
                yield "model.layers.0", object()

        with self.assertRaisesRegex(DataError, "no module reports"):
            adapters.require_gradient_checkpointing_ready(
                Model(), context="the test training forward")

    def test_diagnostic_accepts_a_complete_state(self):
        module = CheckpointingModule()
        module.gradient_checkpointing = True

        class Model:
            def named_modules(self):
                yield "model.layers.0", module

        state = adapters.require_gradient_checkpointing_ready(
            Model(), context="the test training forward")
        self.assertEqual(state["modules_with_gradient_checkpointing"], 1)
        self.assertEqual(state["modules_missing_checkpoint_function"], 0)


class EvaluationBaseBranchTests(unittest.TestCase):
    def test_base_evaluation_uses_the_base_only_loader(self):
        calls = []

        def loader(config, **kwargs):
            calls.append(kwargs)
            return FakeModel(), types.SimpleNamespace(pad_token_id=0), {
                "training_method": "base"}

        config = support.make_config(training={"method": "lora"}, lora={})
        model, tokenizer, report = adapters.load_model_for_evaluation(
            config, base_loader=loader)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["for_inference"])
        self.assertEqual(report["source_kind"], "base")


class TargetModuleTests(unittest.TestCase):
    def test_all_targets_must_exist(self):
        available = {"q_proj": 4, "k_proj": 4, "gate_proj": 4}
        report = adapters.validate_target_modules(
            available, ["q_proj", "gate_proj"])
        self.assertEqual(report["matched_modules"], {"q_proj": 4, "gate_proj": 4})
        self.assertEqual(report["total_matched_modules"], 8)

    def test_missing_target_fails_with_available_names(self):
        available = {"q_proj": 4, "linear_attn_out_proj": 32, "mlp_gate": 1}
        with self.assertRaisesRegex(DataError, "not found in the loaded model"):
            adapters.validate_target_modules(available, ["q_proj", "o_proj"])
        try:
            adapters.validate_target_modules(available, ["o_proj"])
        except DataError as exc:
            self.assertIn("o_proj", str(exc))
            self.assertIn("linear_attn_out_proj", str(exc))

    def test_zero_count_module_is_missing(self):
        with self.assertRaises(DataError):
            adapters.validate_target_modules({"q_proj": 0}, ["q_proj"])

    def test_module_short_names_counts_occurrences(self):
        class Fake:
            def named_modules(self):
                yield "", self
                yield "model.layers.0.self_attn.q_proj", object()
                yield "model.layers.1.self_attn.q_proj", object()
                yield "model.layers.0.mlp.gate_proj", object()

        counts = adapters.module_short_names(Fake())
        self.assertEqual(counts["q_proj"], 2)
        self.assertEqual(counts["gate_proj"], 1)
        self.assertIsNone(counts.get("o_proj"))


class TrainableAssertionTests(unittest.TestCase):
    def lora_config(self, **overrides):
        values = {"rank": 64, "alpha": 64, "dropout": 0.0, "bias": "none",
                  "target_modules": ("q_proj", "gate_proj")}
        values.update(overrides)
        from stage1.config import LoraConfig
        return LoraConfig(**values)

    def test_base_frozen_and_adapters_trainable(self):
        named = [
            ("base.weight", FakeParameter(1000, False)),
            ("base.layers.0.q_proj.lora_A.default.weight", FakeParameter(64, True)),
            ("base.layers.0.q_proj.lora_B.default.weight", FakeParameter(64, True)),
            ("base.layers.0.gate_proj.lora_A.default.weight", FakeParameter(64, True)),
            ("base.layers.0.gate_proj.lora_B.default.weight", FakeParameter(64, True)),
        ]
        report = adapters.verify_lora_trainables(named, self.lora_config())
        self.assertEqual(report["total_parameters"], 1256)
        self.assertEqual(report["trainable_parameters"], 256)
        self.assertEqual(report["base_parameters_trainable"], 0)
        self.assertTrue(report["base_frozen"])
        self.assertAlmostEqual(report["trainable_percentage"],
                               100.0 * 256 / 1256, places=4)
        self.assertEqual(report["adapter_targets"],
                         {"gate_proj": 2, "q_proj": 2})

    def test_base_trainable_parameter_is_refused(self):
        named = [
            ("base.weight", FakeParameter(10, True)),
            ("base.q_proj.lora_A.weight", FakeParameter(4, True)),
        ]
        with self.assertRaisesRegex(DataError, "base parameter"):
            adapters.verify_lora_trainables(named, self.lora_config())

    def test_no_adapters_is_refused(self):
        named = [("base.weight", FakeParameter(10, False))]
        with self.assertRaisesRegex(DataError, "No trainable adapter"):
            adapters.verify_lora_trainables(named, self.lora_config())

    def test_partial_target_coverage_is_refused(self):
        named = [
            ("base.q_proj.lora_A.weight", FakeParameter(4, True)),
            ("base.q_proj.lora_B.weight", FakeParameter(4, True)),
        ]
        with self.assertRaisesRegex(DataError, "without trainable adapter"):
            adapters.verify_lora_trainables(named, self.lora_config())

    def test_bias_training_is_allowed_when_configured(self):
        named = [
            ("base.q_proj.bias", FakeParameter(8, True)),
            ("base.q_proj.lora_A.weight", FakeParameter(4, True)),
            ("base.q_proj.lora_B.weight", FakeParameter(4, True)),
            ("base.gate_proj.lora_A.weight", FakeParameter(4, True)),
            ("base.gate_proj.lora_B.weight", FakeParameter(4, True)),
        ]
        report = adapters.verify_lora_trainables(
            named, self.lora_config(bias="all"))
        self.assertEqual(report["base_parameters_trainable"], 0)
        self.assertTrue(report["base_frozen"])


if __name__ == "__main__":
    unittest.main()
