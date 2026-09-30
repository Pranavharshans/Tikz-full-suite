"""Gemma unified loader isolation and literal-target regression checks."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from stage1 import config as config_module
from unsloth_native import models

STAGE1 = Path(__file__).resolve().parents[1]


class GemmaNativeTests(unittest.TestCase):
    def test_unified_loader_disables_nontext_adapters(self):
        config = config_module.load_config(STAGE1 / "configs/gemma4-12b-lora.yaml")
        calls = {}
        tokenizer = types.SimpleNamespace(__len__=lambda: 42)
        model = types.SimpleNamespace(named_modules=lambda: [], named_parameters=lambda: [])

        class Loader:
            @staticmethod
            def from_pretrained(**kwargs):
                calls['load'] = kwargs
                return model, tokenizer

            @staticmethod
            def get_peft_model(model, **kwargs):
                calls['peft'] = kwargs
                return model

        with patch.dict(sys.modules, {
            'unsloth': types.SimpleNamespace(FastVisionModel=Loader),
            'torch': types.SimpleNamespace(bfloat16='bf16'),
        }), patch.object(models, '_verify_tokenizer', return_value=(None, {})), \
                patch.object(models.adapters, 'validate_target_modules'), \
                patch.object(models.adapters, 'verify_lora_trainables'):
            models.load_native_model(models.get_spec('gemma4-12b'), config, {},
                                     local_files_only=False)
        self.assertEqual(calls['load']['revision'], config.model.revision)
        self.assertFalse(calls['load']['load_in_4bit'])
        self.assertFalse(calls['peft']['finetune_vision_layers'])
        self.assertFalse(calls['peft']['finetune_audio_layers'])
        self.assertTrue(calls['peft']['finetune_language_layers'])

    def test_verbatim_template_preserves_channel_text_and_whitespace(self):
        from jinja2 import Environment
        template = Environment().from_string(
            (STAGE1 / 'templates/gemma4-verbatim.jinja').read_text())
        target = '  \\node {<|channel>thought literal<channel|>};\n'
        user = {'role': 'user', 'content': 'Draw a label.'}
        prompt = template.render(messages=[user], bos_token='<bos>',
                                 add_generation_prompt=True)
        full = template.render(messages=[user, {'role': 'assistant', 'content': target}],
                               bos_token='<bos>', add_generation_prompt=False)
        self.assertTrue(prompt.endswith('<|turn>model\n<|channel>thought\n<channel|>'))
        self.assertEqual(full, prompt + target + '<turn|>\n')


if __name__ == '__main__':
    unittest.main()
