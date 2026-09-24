"""Tests for chat-template rendering and assistant-only loss masks."""
import tempfile
import unittest
from pathlib import Path

from tests import support

from stage1 import config as config_module
from stage1 import formatting
from stage1.errors import FormattingError

INSTRUCTION = "Draw a square with a diagonal."
TIKZ = "\\begin{tikzpicture}\\draw (0,0) rectangle (1,1);\\end{tikzpicture}"


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = support.FakeTokenizer()
        self.template = formatting.resolve_chat_template(
            self.tokenizer, config_module.TokenizerConfig())

    def format(self, **overrides):
        kwargs = dict(row_id="row-1", instruction=INSTRUCTION, tikz=TIKZ,
                      template=self.template, kwargs={})
        kwargs.update(overrides)
        return formatting.format_example(self.tokenizer, **kwargs)

    def test_user_system_and_prompt_tokens_receive_minus_100(self):
        example = self.format()
        self.assertGreater(example.prompt_tokens, 0)
        self.assertTrue(all(label == -100 for label in
                            example.labels[:example.prompt_tokens]))
        self.assertEqual(example.supervised_tokens,
                         example.total_tokens - example.prompt_tokens)

    def test_assistant_tokens_are_supervised_and_match_inputs(self):
        example = self.format()
        for index in range(example.prompt_tokens, example.total_tokens):
            self.assertEqual(example.labels[index], example.input_ids[index])

    def test_last_token_is_supervised(self):
        example = self.format()
        self.assertNotEqual(example.labels[-1], -100)

    def test_no_loss_before_the_boundary_is_supervised(self):
        example = self.format()
        self.assertEqual(sum(1 for label in example.labels if label != -100),
                         example.supervised_tokens)

    def test_prompt_text_is_a_prefix_of_the_full_render(self):
        prompt_text, full_text = formatting.render_pair(
            self.tokenizer, instruction=INSTRUCTION, tikz=TIKZ,
            template=self.template, kwargs={})
        self.assertTrue(full_text.startswith(prompt_text))
        self.assertGreater(len(full_text), len(prompt_text))

    def test_template_kwargs_are_forwarded(self):
        self.format(kwargs={"enable_thinking": False})
        calls = self.tokenizer.template_calls
        self.assertTrue(all(call["kwargs"].get("enable_thinking") is False
                            for call in calls))
        self.assertTrue(all(call["chat_template"] == self.template.text
                            for call in calls))

    def test_generation_prompt_is_used_for_the_prompt_render(self):
        self.format()
        flags = [call["add_generation_prompt"] for call in self.tokenizer.template_calls]
        self.assertEqual(flags, [True, False])


class FailureTests(unittest.TestCase):
    def test_prefix_mismatch_is_refused(self):
        tokenizer = support.FakeTokenizer(prefix_breaker=True)
        template = formatting.resolve_chat_template(
            tokenizer, config_module.TokenizerConfig())
        with self.assertRaisesRegex(FormattingError, "not a prefix"):
            formatting.format_example(
                tokenizer, row_id="row-1", instruction=INSTRUCTION, tikz=TIKZ,
                template=template, kwargs={})

    def test_boundary_spanning_token_is_refused(self):
        probe = support.FakeTokenizer()
        prompt_text, _ = formatting.render_pair(
            probe, instruction=INSTRUCTION, tikz=TIKZ,
            template=formatting.resolve_chat_template(
                probe, config_module.TokenizerConfig()), kwargs={})
        boundary = len(prompt_text)
        tokenizer = support.FakeTokenizer(merge=(boundary - 1, boundary + 2))
        template = formatting.resolve_chat_template(
            tokenizer, config_module.TokenizerConfig())
        with self.assertRaisesRegex(FormattingError, "spans the prompt/assistant"):
            formatting.format_example(
                tokenizer, row_id="row-1", instruction=INSTRUCTION, tikz=TIKZ,
                template=template, kwargs={})

    def test_tokenizer_without_offsets_is_refused(self):
        tokenizer = support.FakeTokenizer(raise_on_offsets=True)
        template = formatting.resolve_chat_template(
            tokenizer, config_module.TokenizerConfig())
        with self.assertRaisesRegex(FormattingError, "return_offsets_mapping"):
            formatting.format_example(
                tokenizer, row_id="row-1", instruction=INSTRUCTION, tikz=TIKZ,
                template=template, kwargs={})

    def test_empty_instruction_and_target_are_refused(self):
        tokenizer = support.FakeTokenizer()
        template = formatting.resolve_chat_template(
            tokenizer, config_module.TokenizerConfig())
        with self.assertRaisesRegex(FormattingError, "empty instruction"):
            formatting.format_example(
                tokenizer, row_id="r", instruction="  ", tikz=TIKZ,
                template=template, kwargs={})
        with self.assertRaisesRegex(FormattingError, "empty TikZ"):
            formatting.format_example(
                tokenizer, row_id="r", instruction=INSTRUCTION, tikz=" ",
                template=template, kwargs={})

    def test_tokenizer_without_template_is_refused(self):
        class Bare:
            chat_template = None

        with self.assertRaisesRegex(FormattingError, "no chat template"):
            formatting.resolve_chat_template(Bare(), config_module.TokenizerConfig())

    def test_empty_template_file_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.jinja"
            path.write_text("   \n")
            with self.assertRaisesRegex(FormattingError, "empty"):
                formatting.resolve_chat_template(
                    support.FakeTokenizer(),
                    config_module.TokenizerConfig(chat_template_file=path))


class TemplateResolutionTests(unittest.TestCase):
    def test_tokenizer_template_is_used_by_default(self):
        tokenizer = support.FakeTokenizer()
        template = formatting.resolve_chat_template(
            tokenizer, config_module.TokenizerConfig())
        self.assertEqual(template.source, "tokenizer")
        self.assertEqual(template.text, tokenizer.chat_template)

    def test_explicit_template_file_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.jinja"
            path.write_text("{{ messages[0].content }}")
            template = formatting.resolve_chat_template(
                support.FakeTokenizer(),
                config_module.TokenizerConfig(chat_template_file=path))
            self.assertEqual(template.source, str(path))
            self.assertIn("messages", template.text)


class ExampleJsonTests(unittest.TestCase):
    def test_to_jsonable_has_no_token_lists(self):
        tokenizer = support.FakeTokenizer()
        template = formatting.resolve_chat_template(
            tokenizer, config_module.TokenizerConfig())
        example = formatting.format_example(
            tokenizer, row_id="row-1", instruction=INSTRUCTION, tikz=TIKZ,
            template=template, kwargs={})
        payload = example.to_jsonable()
        self.assertEqual(payload["row_id"], "row-1")
        self.assertNotIn("input_ids", payload)
        self.assertNotIn("labels", payload)


if __name__ == "__main__":
    unittest.main()
