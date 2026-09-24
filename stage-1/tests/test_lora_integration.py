"""Dependency-gated LoRA integration tests (torch + transformers + peft).

These exercise the real PEFT round trips that the dependency-free suite can
only fake:

- applying LoRA through PEFT and asserting the trainable-parameter contract,
- native adapter save -> fresh-base attach -> identical logits,
- merge equivalence (base+adapter vs merged logits),
- artifact verification of a real adapter directory,
- a fresh transformers Trainer resuming from a native checkpoint written by a
  previous Trainer (the production ``resume_from_checkpoint`` path, without
  manually reconstructing optimizer or scheduler objects).

Everything here is skipped where the dependencies are missing; the run report
states that the LoRA GPU/resume gates remain unverified in such environments.
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

from tests import support

from stage1 import checkpointing
from stage1.errors import CheckpointError

HAS_PEFT = importlib.util.find_spec("peft") is not None
requires_peft = unittest.skipUnless(HAS_PEFT, "peft is not installed")
HAS_DATASETS = importlib.util.find_spec("datasets") is not None


def tiny_llama():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2)
    return LlamaForCausalLM(config)


def apply_peft_lora(model, targets=("q_proj", "k_proj", "v_proj", "o_proj")):
    from peft import LoraConfig, get_peft_model

    return get_peft_model(model, LoraConfig(
        r=4, lora_alpha=4, lora_dropout=0.0, bias="none",
        target_modules=list(targets), task_type="CAUSAL_LM"))


@support.requires_transformers
@support.requires_torch
@requires_peft
class AdapterRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def stage1_lora(self):
        from stage1.config import LoraConfig
        return LoraConfig(rank=4, alpha=4, dropout=0.0, bias="none",
                          target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))

    def test_trainable_assertions_on_a_real_peft_model(self):
        from stage1 import adapters

        model = apply_peft_lora(tiny_llama())
        report = adapters.verify_lora_trainables(model.named_parameters(),
                                                 self.stage1_lora())
        self.assertEqual(report["base_parameters_trainable"], 0)
        self.assertGreater(report["trainable_parameters"], 0)
        self.assertLess(report["trainable_percentage"], 5.0)
        self.assertEqual(set(report["adapter_targets"]),
                         {"q_proj", "k_proj", "v_proj", "o_proj"})

    def test_adapter_save_attach_and_identical_logits(self):
        import torch
        from stage1 import adapters

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

    def test_merge_equivalence(self):
        import torch

        model = apply_peft_lora(tiny_llama())
        inputs = torch.tensor([[2, 3, 4]])
        model.eval()
        with torch.no_grad():
            before = model(inputs).logits.detach().float()
        merged = model.merge_and_unload()
        merged.eval()
        with torch.no_grad():
            after = merged(inputs).logits.detach().float()
        max_diff = float((before - after).abs().max())
        self.assertLessEqual(max_diff, 1e-4, f"merge diff {max_diff}")

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
