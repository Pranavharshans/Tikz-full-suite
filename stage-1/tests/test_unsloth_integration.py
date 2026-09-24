"""GPU/Unsloth regression for the reloaded-adapter training lifecycle.

The v9 preflight failed ``checkpoint.trainer_resume`` with
``AttributeError: 'LlamaDecoderLayer' object has no attribute
'_gradient_checkpointing_func'``: the saved adapter was re-attached to a fresh
base, but Unsloth's ``for_training`` only flips the ``gradient_checkpointing``
flags and nothing installed the checkpoint function.

This test runs the production loading paths against a tiny **local** Llama
checkpoint (no download): initial LoRA forward/backward -> adapter save ->
release -> base-only reload -> trainable adapter attach -> training
preparation -> forward/backward. It is skipped unless Unsloth and a CUDA
device are available, so CPU-only environments still run the dependency-free
and dependency-gated suites.
"""
import gc
import math
import tempfile
import unittest
from pathlib import Path

from tests import support

try:  # Unsloth must initialize before Transformers for its patches.
    import unsloth  # noqa: F401
    HAS_UNSLOTH = True
except Exception:  # pragma: no cover - environment dependent
    HAS_UNSLOTH = False

from stage1 import adapters  # noqa: E402

requires_unsloth_cuda = unittest.skipUnless(
    HAS_UNSLOTH and support.HAS_TORCH and support.HAS_CUDA,
    "unsloth and a CUDA device are required")


def build_tiny_model(directory, *, vocab_size=256):
    """Write a tiny bf16 Llama checkpoint plus a matching fast tokenizer."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    config = LlamaConfig(
        vocab_size=vocab_size, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512)
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(str(directory), safe_serialization=True)

    vocab = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3}
    for index in range(vocab_size - 4):
        vocab[f"tok{index}"] = index + 4
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
        bos_token="[BOS]", eos_token="[EOS]", model_max_length=256)
    tokenizer.save_pretrained(str(directory))
    return directory


def tiny_config():
    return support.make_config(
        model={"id": "test/tiny-llama", "revision": "a" * 40,
               "adapter": "minicpm5", "loader": "unsloth-language-model"},
        data={"max_seq_len": 256},
        training={"method": "lora", "learning_rate": 1e-4,
                  "per_device_train_batch_size": 1,
                  "gradient_accumulation_steps": 16},
        lora={})


@requires_unsloth_cuda
class UnslothReloadLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model_dir = self.root / "base"
        self.model_dir.mkdir()
        build_tiny_model(self.model_dir)
        self.config = tiny_config()
        self.adapter_dir = self.root / "adapter"

    def tearDown(self):
        self.temporary.cleanup()

    def make_batch(self):
        import torch
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long, device="cuda")
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids),
        }

    def test_reloaded_adapter_restores_the_full_training_lifecycle(self):
        import torch

        # 1. Initial production path: a fresh adapter through Unsloth.
        model, tokenizer, report = adapters.load_model_and_tokenizer(
            self.config, source_override=str(self.model_dir), for_training=True)
        self.assertTrue(report["adapter_applied"])
        self.assertEqual(report["for_training_patch_stage"], "after_adapter")
        state = adapters.require_gradient_checkpointing_ready(
            model, context="the GPU test forward")
        self.assertGreater(state["modules_with_gradient_checkpointing"], 0)
        model.train()
        outputs = model(**self.make_batch())
        self.assertTrue(math.isfinite(float(outputs.loss)))
        outputs.loss.backward()
        model.zero_grad(set_to_none=True)

        # 2. Save the adapter and release the model.
        model.save_pretrained(str(self.adapter_dir))
        self.assertTrue((self.adapter_dir / "adapter_config.json").is_file())
        self.assertTrue(
            (self.adapter_dir / "adapter_model.safetensors").is_file())
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # 3. Reload the pinned base only, then attach the saved adapter once,
        #    trainable, exactly like the preflight resume path.
        base, tokenizer, report = adapters.load_base_model_and_tokenizer(
            self.config, source_override=str(self.model_dir), for_training=False)
        self.assertIsNone(getattr(base, "peft_config", None))
        attached = adapters.attach_lora_adapter(
            base, self.adapter_dir, is_trainable=True)
        self.assertIsNotNone(getattr(attached, "peft_config", None))

        # 4. Restore the Unsloth training lifecycle (the fix under test).
        report = {}
        prepared = adapters.prepare_model_for_training(
            attached, self.config, report=report)
        self.assertEqual(report["training_state"]["peft_training_state_api"],
                         "FastLanguageModel.patch_peft_model")
        self.assertEqual(
            report["training_state"]["gradient_checkpointing_enable"],
            {"use_reentrant": False})
        state = adapters.require_gradient_checkpointing_ready(
            prepared, context="the GPU test forward")
        self.assertGreater(state["modules_with_gradient_checkpointing"], 0)
        self.assertEqual(state["modules_missing_checkpoint_function"], 0)

        # 5. The first training forward/backward after the reload must work.
        prepared.train()
        outputs = prepared(**self.make_batch())
        loss = float(outputs.loss)
        self.assertTrue(math.isfinite(loss))
        outputs.loss.backward()
        adapter_grads = [
            parameter.grad for name, parameter in prepared.named_parameters()
            if "lora_" in name and parameter.grad is not None]
        self.assertTrue(adapter_grads)
        self.assertTrue(any(float(grad.abs().sum()) > 0 for grad in adapter_grads))
        trainable = adapters.verify_lora_trainables(
            prepared.named_parameters(), self.config.lora)
        self.assertEqual(trainable["base_parameters_trainable"], 0)
        self.assertTrue(trainable["base_frozen"])


if __name__ == "__main__":
    unittest.main()
