"""Dependency-free tests for the LoRA adapter layer.

These exercise the pure helpers used before and after Unsloth attaches
adapters: target-module validation against the loaded model, trainable
parameter assertions, and the numbers recorded in the load report. The actual
PEFT save/load/resume/merge round trips are dependency-gated in
``test_lora_integration.py``.
"""
import unittest

from stage1 import adapters
from stage1.errors import DataError


class FakeParameter:
    def __init__(self, numel, requires_grad):
        self._numel = numel
        self.requires_grad = requires_grad

    def numel(self):
        return self._numel


class FakeModule:
    """Minimal named_modules() source: (name, object) pairs."""

    def __init__(self, names):
        self._names = names

    def named_modules(self):
        yield "", self
        for name in self._names:
            yield name, object()


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
        model = FakeModule([
            "model.layers.0.self_attn.q_proj",
            "model.layers.1.self_attn.q_proj",
            "model.layers.0.mlp.gate_proj",
        ])
        counts = adapters.module_short_names(model)
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


class EvaluationLoadTests(unittest.TestCase):
    def test_base_and_adapter_loading_is_method_aware(self):
        # load_model_for_evaluation must never load a LoRA adapter as if it
        # were a complete model; the dispatch is by artifact metadata.
        import inspect
        signature = inspect.signature(adapters.load_model_for_evaluation)
        self.assertIn("artifact_meta", signature.parameters)
        self.assertIn("attach_lora_adapter", dir(adapters))
        source = inspect.getsource(adapters.load_model_for_evaluation)
        self.assertIn('method == "lora"', source)
        self.assertIn("attach_lora_adapter", source)


if __name__ == "__main__":
    unittest.main()
