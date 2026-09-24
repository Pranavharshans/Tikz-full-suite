"""Dependency-gated LoRA integration tests (torch + transformers + peft).

These exercise the real PEFT round trips that the dependency-free suite can
only fake:

- applying LoRA through PEFT and asserting the trainable-parameter contract,
- native adapter save -> fresh-base attach -> identical logits,
- the **production restore function** recovering corrupted adapter weights,
- the production evaluation, preflight-reload and merge paths loading the base
  exactly once and attaching the saved adapter exactly once,
- merge equivalence (base+adapter vs merged logits),
- artifact verification of a real adapter directory,
- a fresh transformers Trainer resuming from a native checkpoint written by a
  previous Trainer (the production ``resume_from_checkpoint`` path, without
  manually reconstructing optimizer or scheduler objects).

Everything here is skipped where the dependencies are missing; the run report
states that the LoRA GPU/resume gates remain unverified in such environments.
"""
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

from tests import support

from stage1 import adapters, checkpointing, collator, merge, preflight
from stage1.errors import CheckpointError, DataError

HAS_PEFT = importlib.util.find_spec("peft") is not None
HAS_DATASETS = importlib.util.find_spec("datasets") is not None
requires_peft = unittest.skipUnless(HAS_PEFT, "peft is not installed")


class StubTokenizer:
    """Minimal tokenizer stand-in for save/attach paths."""

    def __init__(self):
        self.pad_token_id = 0

    def save_pretrained(self, directory):
        Path(directory, "tokenizer_config.json").write_text("{}")


def tiny_llama():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2)
    return LlamaForCausalLM(config)


def apply_peft_lora(model, targets=("q_proj", "k_proj", "v_proj", "o_proj"),
                    *, randomize=True):
    import torch
    from peft import LoraConfig, get_peft_model

    wrapped = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=4, lora_dropout=0.0, bias="none",
        target_modules=list(targets), task_type="CAUSAL_LM"))
    if randomize:
        # Adapters start as identity; randomize so logit comparisons are real.
        torch.manual_seed(1)
        with torch.no_grad():
            for name, parameter in wrapped.named_parameters():
                if "lora_" in name:
                    parameter.normal_(0.0, 0.02)
    return wrapped


def state_copy(model):
    return {key: value.detach().clone()
            for key, value in model.state_dict().items()}


@support.requires_transformers
@support.requires_torch
@requires_peft
class AdapterRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = support.make_config(
            training={"method": "lora", "learning_rate": 1e-4},
            lora={"rank": 4, "alpha": 4, "dropout": 0.0, "bias": "none",
                  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]})

    def tearDown(self):
        self.temporary.cleanup()

    def stage1_lora(self):
        from stage1.config import LoraConfig
        return LoraConfig(rank=4, alpha=4, dropout=0.0, bias="none",
                          target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))

    def test_trainable_assertions_on_a_real_peft_model(self):
        model = apply_peft_lora(tiny_llama())
        report = adapters.verify_lora_trainables(model.named_parameters(),
                                                 self.stage1_lora())
        self.assertEqual(report["base_parameters_trainable"], 0)
        self.assertGreater(report["trainable_parameters"], 0)
        # A tiny two-layer model makes rank 4 a larger share than a real 2B/4B
        # checkpoint; the contract is "adapters are a small minority".
        self.assertLess(report["trainable_percentage"], 15.0)
        self.assertEqual(set(report["adapter_targets"]),
                         {"q_proj", "k_proj", "v_proj", "o_proj"})

    def test_adapter_save_attach_and_identical_logits(self):
        import torch

        model = apply_peft_lora(tiny_llama())
        inputs = torch.tensor([[1, 2, 3, 4, 5]])
        model.eval()
        with torch.no_grad():
            before = model(inputs).logits.detach().clone()
        directory = self.root / "adapter"
        model.save_pretrained(str(directory))
        self.assertTrue((directory / "adapter_config.json").is_file())
        self.assertTrue((directory / "adapter_model.safetensors").is_file())

        attached = adapters.attach_lora_adapter(tiny_llama(), directory)
        attached.eval()
        with torch.no_grad():
            after = attached(inputs).logits.detach()
        self.assertTrue(torch.allclose(before, after, atol=1e-6),
                        f"max diff {(before - after).abs().max().item()}")

    def test_production_restore_recovers_a_mutated_adapter(self):
        """The production restore uses PEFT's API and is fully reversible."""
        import torch

        model = apply_peft_lora(tiny_llama())
        inputs = torch.tensor([[1, 2, 3, 4]])
        model.eval()
        with torch.no_grad():
            expected = model(inputs).logits.detach().clone()
        directory = self.root / "adapter"
        model.save_pretrained(str(directory))

        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_" in name:
                    parameter.add_(0.5)
        with torch.no_grad():
            corrupted = model(inputs).logits.detach().clone()
        self.assertFalse(torch.allclose(expected, corrupted, atol=1e-6))

        report = checkpointing.load_artifact_weights(model, directory,
                                                     method="lora")
        self.assertEqual(report["restore_api"],
                         "peft.set_peft_model_state_dict")
        self.assertEqual(report["readback_api"],
                         "peft.get_peft_model_state_dict")
        self.assertTrue(report["readback_verified"])
        # The real PEFT setter reports base weights as "missing" because
        # adapter files never contain them; that must not fail the restore.
        self.assertGreater(report["setter_missing_keys"], 0)
        self.assertEqual(report["adapter_name"], "default")
        with torch.no_grad():
            restored = model(inputs).logits.detach().clone()
        self.assertTrue(torch.allclose(expected, restored, atol=1e-6),
                        f"max diff {(expected - restored).abs().max().item()}")

    def test_production_restore_rejects_a_genuine_adapter_incompatibility(self):
        """A rank-mismatched adapter is a real incompatibility and must fail."""
        from peft import LoraConfig, get_peft_model

        small = apply_peft_lora(tiny_llama())
        adapter_dir = self.root / "adapter-r4"
        small.save_pretrained(str(adapter_dir))
        big = get_peft_model(tiny_llama(), LoraConfig(
            r=8, lora_alpha=8, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM"))
        with self.assertRaises(CheckpointError):
            checkpointing.load_artifact_weights(big, adapter_dir, method="lora")

    def test_evaluation_path_loads_base_only_then_attaches_once(self):
        """Production evaluation path: one base load, one adapter attach."""
        import torch

        base = tiny_llama()
        base_state = state_copy(base)
        reference = apply_peft_lora(base)
        inputs = torch.tensor([[1, 2, 3, 4]])
        reference.eval()
        with torch.no_grad():
            expected = reference(inputs).logits.detach().clone()
        adapter_dir = self.root / "adapter"
        reference.save_pretrained(str(adapter_dir))

        calls = []

        def base_loader(config, **kwargs):
            calls.append(kwargs)
            fresh = tiny_llama()
            fresh.load_state_dict(base_state, strict=False)
            return fresh, StubTokenizer(), {"training_method": "base"}

        model, tokenizer, report = adapters.load_model_for_evaluation(
            self.config, checkpoint_dir=adapter_dir,
            artifact_meta={"training_method": "lora",
                           "base_model_id": self.config.model.id,
                           "base_model_revision": self.config.model.revision},
            base_loader=base_loader)
        self.assertEqual(len(calls), 1)
        self.assertEqual(report["source_kind"], "lora_adapter")
        self.assertEqual(report["adapter_attached"], "once")
        self.assertIsNotNone(getattr(model, "peft_config", None))
        model.eval()
        with torch.no_grad():
            actual = model(inputs).logits.detach().clone()
        self.assertTrue(torch.allclose(expected, actual, atol=1e-5),
                        f"max diff {(expected - actual).abs().max().item()}")

    def test_preflight_reload_path_loads_base_only_then_attaches_once(self):
        """Production preflight reload: base-only load, one attach, forward check."""
        import torch

        base = tiny_llama()
        base_state = state_copy(base)
        model = apply_peft_lora(base)
        plan = collator.pad_batch(
            [{"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]}],
            pad_token_id=0)
        batch = {
            "input_ids": torch.tensor(plan.input_ids, dtype=torch.long),
            "labels": torch.tensor(plan.labels, dtype=torch.long),
            "attention_mask": torch.tensor(plan.attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(plan.position_ids, dtype=torch.long),
        }
        model.eval()
        with torch.no_grad():
            outputs = model(**batch)
            eval_loss = float(outputs.loss)
        config = support.make_config(
            training={"method": "lora", "learning_rate": 1e-4,
                      "optim": "adamw_torch"},
            lora={"rank": 4, "alpha": 4, "dropout": 0.0,
                  # The saved adapter is q/k/v/o only; the config must match or
                  # the trainable check correctly refuses a partial attachment.
                  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]})
        ctx = preflight.PreflightContext(
            config=config, run_dir=self.root / "run", identity={}, prepared={})
        ctx.model = model
        ctx.tokenizer = StubTokenizer()
        ctx.scratch.update({"batch_plan": plan, "eval_loss": eval_loss,
                            "logits_shape": tuple(outputs.logits.shape),
                            "device": "cpu"})
        calls = []

        def base_loader(config, **kwargs):
            calls.append(kwargs)
            fresh = tiny_llama()
            fresh.load_state_dict(base_state, strict=False)
            return fresh, StubTokenizer(), {}

        outcome = preflight.check_model_reload(ctx, base_loader=base_loader)
        self.assertEqual(outcome["status"], "pass", outcome["detail"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(ctx.load_report["adapter_attached"], "once")
        self.assertIsNotNone(getattr(ctx.model, "peft_config", None))
        # The reloaded model reproduces the original (base + adapter) logits.
        ctx.model.eval()
        with torch.no_grad():
            restored = ctx.model(**batch).logits.detach().clone()
            original = model(**batch).logits.detach().clone()
        self.assertTrue(torch.allclose(original, restored, atol=1e-5))

    def test_merge_path_loads_base_only_and_verifies_equivalence(self):
        """Production merge flow: base-only load, one attach, logit equivalence."""
        import torch
        from transformers import LlamaForCausalLM

        base = tiny_llama()
        base_state = state_copy(base)
        reference = apply_peft_lora(base)
        adapter_dir = self.root / "adapter"
        reference.save_pretrained(str(adapter_dir))
        inputs = torch.tensor([[1, 2, 3, 4]])
        calls = []

        def base_loader(config, **kwargs):
            calls.append(kwargs)
            fresh = tiny_llama()
            fresh.load_state_dict(base_state, strict=False)
            return fresh, StubTokenizer(), {}

        out_dir = self.root / "merged"
        result = merge.merge_adapter(
            self.config, adapter_dir, out_dir, batch={"input_ids": inputs},
            base_loader=base_loader, tolerance=1e-4)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["load_report"]["adapter_attached"], "once")
        self.assertTrue(result["verification"]["passed"],
                        result["verification"])
        self.assertTrue(result["verification"]["checked"])
        merged = LlamaForCausalLM.from_pretrained(out_dir)
        merged.eval()
        with torch.no_grad():
            merged_logits = merged(inputs).logits.detach().float()
            reference_logits = reference(inputs).logits.detach().float()
        self.assertTrue(torch.allclose(reference_logits, merged_logits, atol=1e-4))

    def test_attach_refuses_a_second_adapter_on_a_real_peft_model(self):
        model = apply_peft_lora(tiny_llama())
        directory = self.root / "adapter"
        model.save_pretrained(str(directory))
        with self.assertRaisesRegex(DataError, "already carries a PEFT adapter"):
            adapters.attach_lora_adapter(model, directory)

    def test_real_adapter_passes_artifact_verification(self):
        model = apply_peft_lora(tiny_llama())
        directory = self.root / "final" / "smoke-1000"
        directory.mkdir(parents=True)
        model.save_pretrained(str(directory))
        (directory / "tokenizer_config.json").write_text("{}")
        checkpointing.write_final_meta(
            directory, identity_sha256="a" * 64, gate="smoke-1000",
            global_step=3, supervised_tokens_seen=42, epochs_completed=1.0,
            model_id="test/TinyLlama", model_revision="b" * 40,
            adapter="minicpm5", training_method="lora",
            base_model_id="test/TinyLlama", base_model_revision="b" * 40)
        meta = checkpointing.verify_final_artifact(
            directory, identity_sha256="a" * 64, gate="smoke-1000",
            method="lora", model_id="test/TinyLlama",
            model_revision="b" * 40, adapter="minicpm5")
        self.assertEqual(meta["training_method"], "lora")
        # The same directory is refused as a full-model artifact.
        with self.assertRaises(CheckpointError):
            checkpointing.verify_final_artifact(
                directory, identity_sha256="a" * 64, gate="smoke-1000",
                method="full", model_id="test/TinyLlama",
                model_revision="b" * 40, adapter="minicpm5")


class _FlagOnlyForTraining:
    """Unsloth ``for_training`` without its checkpointing restore.

    This is exactly the state the failed resume path produced: the flags are
    set, but no ``_gradient_checkpointing_func`` is installed.
    """

    @staticmethod
    def for_training(model, use_gradient_checkpointing=True):
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = bool(use_gradient_checkpointing)
        model.train()
        return model


@support.requires_transformers
@support.requires_torch
@requires_peft
class ReloadTrainingStateTests(unittest.TestCase):
    """The reloaded-adapter training lifecycle on real PEFT/Transformers.

    ``Unsloth``'s ``for_training`` only flips ``gradient_checkpointing`` flags.
    Transformers then calls ``self._gradient_checkpointing_func`` on every
    checkpointing decoder layer, so a flag-only reload fails inside the decoder
    with a bare AttributeError. These tests reproduce that state with the real
    libraries and prove that the production preparation makes the training
    forward/backward work again.
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = support.make_config(
            training={"method": "lora", "learning_rate": 1e-4}, lora={})

    def tearDown(self):
        self.temporary.cleanup()

    def reload_adapter(self):
        base = tiny_llama()
        base_state = state_copy(base)
        reference = apply_peft_lora(base)
        adapter_dir = self.root / "adapter"
        reference.save_pretrained(str(adapter_dir))
        fresh = tiny_llama()
        fresh.load_state_dict(base_state, strict=False)
        attached = adapters.attach_lora_adapter(
            fresh, adapter_dir, is_trainable=True)
        return attached

    def batch(self):
        import torch
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }

    def test_flag_only_reload_fails_a_training_forward_in_the_decoder(self):
        model = self.reload_adapter()
        _FlagOnlyForTraining.for_training(model)
        model.train()
        with self.assertRaises(AttributeError) as caught:
            model(**self.batch())
        self.assertIn("_gradient_checkpointing_func", str(caught.exception))

    def test_prepare_restores_a_complete_state_and_the_training_step(self):
        model = self.reload_adapter()
        # Simulate the failing reload state: Unsloth for_training flips the
        # gradient-checkpointing flags but installs no checkpoint function.
        _FlagOnlyForTraining.for_training(model)
        state_before = adapters.gradient_checkpointing_state(model)
        self.assertGreater(state_before["modules_with_gradient_checkpointing"], 0)
        self.assertGreater(state_before["modules_missing_checkpoint_function"], 0)
        with self.assertRaisesRegex(DataError, "_gradient_checkpointing_func"):
            adapters.require_gradient_checkpointing_ready(
                model, context="the test training forward")

        prepared = adapters.prepare_model_for_training(
            model, self.config, loader=_FlagOnlyForTraining)
        state_after = adapters.require_gradient_checkpointing_ready(
            prepared, context="the test training forward")
        self.assertGreater(state_after["modules_with_gradient_checkpointing"], 0)
        self.assertEqual(state_after["modules_missing_checkpoint_function"], 0)

        prepared.train()
        outputs = prepared(**self.batch())
        loss = float(outputs.loss)
        self.assertTrue(math.isfinite(loss))
        outputs.loss.backward()
        adapter_grads = [
            parameter.grad for name, parameter in prepared.named_parameters()
            if "lora_" in name and parameter.grad is not None]
        self.assertTrue(adapter_grads)
        self.assertTrue(any(float(grad.abs().sum()) > 0 for grad in adapter_grads))
        # Base weights stay frozen.
        report = adapters.verify_lora_trainables(
            prepared.named_parameters(), self.config.lora)
        self.assertEqual(report["base_parameters_trainable"], 0)


@support.requires_transformers
@support.requires_torch
@unittest.skipUnless(HAS_DATASETS, "datasets is not installed")
class TrainerResumeTests(unittest.TestCase):
    """A fresh Trainer continues from a native checkpoint (public resume path)."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def build_trainer(self, output_dir, *, max_steps):
        from datasets import Dataset
        from transformers import TrainingArguments, Trainer

        model = tiny_llama()
        dataset = Dataset.from_dict({
            "input_ids": [[1, 2, 3, 4] for _ in range(8)],
            "labels": [[1, 2, 3, 4] for _ in range(8)],
            "attention_mask": [[1, 1, 1, 1] for _ in range(8)],
        })
        arguments = TrainingArguments(
            output_dir=str(output_dir), max_steps=max_steps, save_steps=2,
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            learning_rate=1e-3, lr_scheduler_type="cosine", warmup_ratio=0.03,
            logging_steps=1, report_to=[], seed=0, use_cpu=True,
            save_total_limit=1, dataloader_num_workers=0)
        return Trainer(model=model, args=arguments, train_dataset=dataset)

    def test_fresh_trainer_resumes_from_a_native_checkpoint(self):
        run_dir = self.root / "run"
        first = self.build_trainer(run_dir / "checkpoints" / "full", max_steps=2)
        first.train()
        checkpoint = Path(first.args.output_dir) / "checkpoint-2"
        self.assertTrue(checkpoint.is_dir(), "Trainer did not write a checkpoint")
        checkpointing.write_checkpoint_meta(
            checkpoint, identity_sha256="a" * 64, gate="full",
            global_step=2, supervised_tokens_seen=8, epochs_completed=0.1,
            model_id="test/TinyLlama", model_revision="b" * 40,
            adapter="minicpm5", training_method="full")
        # Our resume discovery/verification accepts the native checkpoint.
        path, meta = checkpointing.find_latest_checkpoint(
            run_dir, "full", "a" * 64, model_id="test/TinyLlama",
            model_revision="b" * 40, adapter="minicpm5", method="full")
        self.assertEqual(path, checkpoint)
        self.assertEqual(meta["global_step"], 2)

        # A fresh Trainer continues from it through the public resume path.
        second = self.build_trainer(run_dir / "checkpoints" / "full", max_steps=4)
        second.train(resume_from_checkpoint=str(checkpoint))
        self.assertGreater(second.state.global_step, 2)
        log_history = second.state.log_history
        finite = [entry["loss"] for entry in log_history
                  if isinstance(entry.get("loss"), (int, float))]
        self.assertTrue(finite)
        self.assertTrue(all(value == value for value in finite))


if __name__ == "__main__":
    unittest.main()
