"""Tests for the standalone merge flow (dependency-free structural tests plus
dependency-gated equivalence tests in test_lora_integration.py)."""
import tempfile
import unittest
from pathlib import Path

from tests import support

from stage1 import merge
from stage1.errors import DataError


class FakeMergedModel:
    def __init__(self, name="merged"):
        self.name = name
        self.saved = []

    def eval(self):
        return self

    def merge_and_unload(self):
        self.merged = True
        return self

    def save_pretrained(self, directory, **kwargs):
        self.saved.append((directory, kwargs))
        Path(directory, "config.json").write_text("{}")
        Path(directory, "model.safetensors").write_bytes(b"merged")


class FakeTokenizer:
    def __init__(self):
        self.saved = []

    def save_pretrained(self, directory):
        self.saved.append(directory)
        Path(directory, "tokenizer_config.json").write_text("{}")


class MergeFlowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.adapter = self.root / "adapter"
        self.adapter.mkdir()
        (self.adapter / "adapter_config.json").write_text("{}")
        self.config = support.make_config(training={"method": "lora"}, lora={})
        self.load_calls = []
        self.attach_calls = []

    def tearDown(self):
        self.temporary.cleanup()

    def loader(self, config, **kwargs):
        self.load_calls.append(kwargs)
        return FakeMergedModel("base"), FakeTokenizer(), {"training_method": "base"}

    def attacher(self, model, adapter_dir):
        self.attach_calls.append(str(adapter_dir))
        model.attached = True
        return model

    def test_merge_without_verification_exports_and_records_once(self):
        out = self.root / "out"
        result = merge.merge_adapter(
            self.config, self.adapter, out, batch={}, verify=False,
            base_loader=self.loader, adapter_attacher=self.attacher,
            metadata={"run_identity_sha256": "a" * 64})
        self.assertEqual(len(self.load_calls), 1)
        self.assertEqual(len(self.attach_calls), 1)
        self.assertEqual(Path(self.attach_calls[0]).resolve(),
                         self.adapter.resolve())
        self.assertEqual(result["load_report"]["adapter_attached"], "once")
        self.assertFalse(result["verification"]["checked"])
        self.assertTrue((out / "config.json").is_file())
        self.assertTrue((out / "model.safetensors").is_file())
        self.assertTrue((out / "tokenizer_config.json").is_file())
        metadata = support.load_json(out / "merge-metadata.json")
        self.assertEqual(metadata["schema_version"], "stage1-merge-v1")
        self.assertTrue(metadata["merged_from_adapter"])
        self.assertEqual(metadata["base_model_id"], self.config.model.id)
        self.assertEqual(metadata["base_model_revision"],
                         self.config.model.revision)
        self.assertEqual(metadata["run_identity_sha256"], "a" * 64)

    def test_merge_requires_a_batch_when_verifying(self):
        with self.assertRaises(Exception):
            merge.merge_adapter(
                self.config, self.adapter, self.root / "out", batch=None,
                base_loader=self.loader, adapter_attacher=self.attacher)

    def test_double_attach_is_refused(self):
        from stage1 import adapters

        class Adaptered:
            peft_config = {"default": object()}

        with self.assertRaisesRegex(DataError, "already carries a PEFT adapter"):
            adapters.attach_lora_adapter(Adaptered(), self.adapter)

    def test_prepare_base_and_adapter_uses_the_injected_loader(self):
        model, tokenizer, report = merge.prepare_base_and_adapter(
            self.config, self.adapter, base_loader=self.loader,
            adapter_attacher=self.attacher)
        self.assertTrue(model.attached)
        self.assertEqual(report["adapter_attached"], "once")


if __name__ == "__main__":
    unittest.main()
