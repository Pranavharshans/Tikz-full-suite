"""Network-marked integration tests for the real tokenizers and pinned template.

Skipped unless ``STAGE1_ALLOW_NETWORK_TESTS=1`` is set, so ordinary unit test
runs never download anything. Run this on the GPU host (or any networked
machine) before the preflight: it proves that both pinned tokenizers load, that
the verbatim template preserves literal thinking tags, and that the
assistant-only mask boundary lands exactly on the target text.

  STAGE1_ALLOW_NETWORK_TESTS=1 python3 -m unittest tests.test_network_integration -v
"""
import importlib.util
import os
import unittest
from pathlib import Path

from tests import support

from stage1 import adapters, config as config_module, formatting

STAGE1 = Path(__file__).resolve().parents[1]
ALLOW = os.environ.get("STAGE1_ALLOW_NETWORK_TESTS") == "1"
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None

INSTRUCTION = "Draw a right triangle with a labeled hypotenuse."
TIKZ = ("\\begin{tikzpicture}\\node {<think>literal</think>};"
        "\\draw (0,0) -- (2,0) -- (0,1.5) -- cycle;\\end{tikzpicture}")

CONFIGS = {
    "qwen3.5": "configs/qwen3.5-4b-full.yaml",
    "minicpm5": "configs/minicpm5-2b-full.yaml",
}


@unittest.skipUnless(ALLOW, "set STAGE1_ALLOW_NETWORK_TESTS=1 to run")
@unittest.skipUnless(HAS_TRANSFORMERS, "transformers is not installed")
@support.requires_yaml
class RealTokenizerTests(unittest.TestCase):
    def test_both_tokenizers_render_and_mask_exactly(self):
        for adapter_name, config_name in CONFIGS.items():
            with self.subTest(adapter=adapter_name):
                config = config_module.load_config(STAGE1 / config_name)
                tokenizer = adapters.load_tokenizer(config)
                template = formatting.resolve_chat_template(tokenizer, config.tokenizer)
                fingerprint = adapters.tokenizer_fingerprint(tokenizer, template, config)
                self.assertTrue(fingerprint["files"],
                                "expected local tokenizer file hashes")
                self.assertEqual(fingerprint["model_id"], config.model.id)

                example = formatting.format_example(
                    tokenizer, row_id="network-test", instruction=INSTRUCTION,
                    tikz=TIKZ, template=template,
                    kwargs=config.tokenizer.chat_template_kwargs)
                self.assertGreater(example.prompt_tokens, 0)
                self.assertGreater(example.supervised_tokens, 0)
                self.assertTrue(all(label == -100 for label in
                                    example.labels[:example.prompt_tokens]))
                self.assertNotEqual(example.labels[-1], -100)

                prompt_text, full_text = formatting.render_pair(
                    tokenizer, instruction=INSTRUCTION, tikz=TIKZ,
                    template=template,
                    kwargs=config.tokenizer.chat_template_kwargs)
                self.assertTrue(full_text.startswith(prompt_text))
                self.assertEqual(full_text[len(prompt_text):-len("<|im_end|>\n")],
                                 TIKZ)


if __name__ == "__main__":
    unittest.main()
