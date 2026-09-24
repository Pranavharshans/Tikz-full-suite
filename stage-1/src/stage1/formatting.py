"""Render Stage 1 examples with the model's native chat template and apply
assistant-only loss masks.

Masking strategy (no heuristics):

1. Render the prompt (user turn + generation prompt) and the full conversation
   (user turn + assistant target) with the same template and options.
2. Require the prompt text to be a strict prefix of the full text.
3. Tokenize the full text once with ``return_offsets_mapping=True`` and find
   the first token that starts at or after the assistant text boundary.
4. Every token before that boundary gets label ``-100``; every token at or
   after it is supervised. A token that *spans* the boundary (starts in the
   prompt, ends in the target) is a hard error: masking it would silently drop
   target characters, and supervising it would train on prompt text. Neither
   is acceptable, so the row is reported instead of guessed at.

The tokenizer must be a fast tokenizer that reports offset mappings. Both
pinned models ship fast tokenizers.
"""
from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass

from .errors import FormattingError
from .util import sha256_text


@dataclass(frozen=True)
class ChatTemplate:
    text: str
    sha256: str
    source: str


@dataclass(frozen=True)
class FormattedExample:
    row_id: str
    input_ids: tuple
    labels: tuple
    prompt_tokens: int
    supervised_tokens: int
    total_tokens: int
    full_text_sha256: str
    template_sha256: str

    def to_jsonable(self) -> dict:
        return {
            "row_id": self.row_id,
            "prompt_tokens": self.prompt_tokens,
            "supervised_tokens": self.supervised_tokens,
            "total_tokens": self.total_tokens,
            "full_text_sha256": self.full_text_sha256,
            "template_sha256": self.template_sha256,
        }


def resolve_chat_template(tokenizer, tokenizer_config) -> ChatTemplate:
    """Use the configured template file, else the tokenizer's own template."""
    path = tokenizer_config.chat_template_file
    if path is not None:
        text = path.read_text()
        if not text.strip():
            raise FormattingError(f"Chat template file is empty: {path}")
        return ChatTemplate(text=text, sha256=sha256_text(text), source=str(path))
    text = getattr(tokenizer, "chat_template", None)
    if not text:
        raise FormattingError(
            "The tokenizer has no chat template. Pin the tokenizer revision, or "
            "set tokenizer.chat_template_file to an explicit Jinja template.")
    return ChatTemplate(text=text, sha256=sha256_text(text), source="tokenizer")


def _apply(tokenizer, messages, template: ChatTemplate, kwargs, *,
           add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            chat_template=template.text, **kwargs)
    except FormattingError:
        raise
    except Exception as exc:  # jinja errors, missing kwargs, ...
        raise FormattingError(
            f"apply_chat_template failed ({type(exc).__name__}: {exc}); "
            f"template source {template.source}, kwargs {sorted(kwargs)}") from exc


def render_pair(tokenizer, *, instruction: str, tikz: str, template: ChatTemplate,
                kwargs: dict) -> tuple:
    """Return ``(prompt_text, full_text)`` for one single-turn example."""
    prompt_messages = [{"role": "user", "content": instruction}]
    full_messages = prompt_messages + [{"role": "assistant", "content": tikz}]
    prompt_text = _apply(tokenizer, prompt_messages, template, kwargs,
                         add_generation_prompt=True)
    full_text = _apply(tokenizer, full_messages, template, kwargs,
                       add_generation_prompt=False)
    if not prompt_text:
        raise FormattingError("Chat template rendered an empty prompt")
    if not full_text.startswith(prompt_text):
        raise FormattingError(
            "The rendered prompt is not a prefix of the full conversation; "
            "the template behaves differently with an assistant turn. "
            f"prompt tail={prompt_text[-80:]!r} full at that point="
            f"{full_text[len(prompt_text):len(prompt_text) + 80]!r}")
    if len(full_text) == len(prompt_text):
        raise FormattingError("Chat template rendered an empty assistant target")
    return prompt_text, full_text


def render_prompt(tokenizer, *, instruction: str, template: ChatTemplate,
                  kwargs: dict) -> str:
    """Render only the user turn plus the generation prompt (for evaluation)."""
    messages = [{"role": "user", "content": instruction}]
    prompt_text = _apply(tokenizer, messages, template, kwargs,
                         add_generation_prompt=True)
    if not prompt_text:
        raise FormattingError("Chat template rendered an empty prompt")
    return prompt_text


def _tokenize_with_offsets(tokenizer, text: str):
    try:
        encoded = tokenizer(text, add_special_tokens=False,
                            return_offsets_mapping=True)
    except TypeError as exc:
        raise FormattingError(
            "The tokenizer does not support return_offsets_mapping; Stage 1 "
            "requires a fast tokenizer so the assistant boundary is exact") from exc
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
        offsets = offsets[0]
    if not input_ids:
        raise FormattingError("Tokenizer produced no tokens for the rendered text")
    if any(offset is None for offset in offsets):
        raise FormattingError(
            "Tokenizer returned a null offset mapping; cannot locate the "
            "assistant boundary exactly")
    return list(input_ids), [tuple(offset) for offset in offsets]


def format_example(tokenizer, *, row_id: str, instruction: str, tikz: str,
                   template: ChatTemplate, kwargs: dict | None = None) -> FormattedExample:
    """Tokenize one example and mask everything before the assistant target."""
    if not instruction or not instruction.strip():
        raise FormattingError(f"{row_id}: empty instruction")
    if not tikz or not tikz.strip():
        raise FormattingError(f"{row_id}: empty TikZ target")
    kwargs = dict(kwargs or {})
    prompt_text, full_text = render_pair(
        tokenizer, instruction=instruction, tikz=tikz, template=template, kwargs=kwargs)
    input_ids, offsets = _tokenize_with_offsets(tokenizer, full_text)

    target_start = len(prompt_text)
    starts = [offset[0] for offset in offsets]
    split_index = bisect.bisect_left(starts, target_start)

    if split_index == 0:
        raise FormattingError(
            f"{row_id}: no prompt tokens precede the assistant target; refusing "
            "to train on an unmasked example")
    if split_index >= len(input_ids):
        raise FormattingError(f"{row_id}: assistant target tokenized to nothing")
    spanning = offsets[split_index - 1][1] > target_start
    if spanning:
        raise FormattingError(
            f"{row_id}: a token spans the prompt/assistant boundary "
            f"(offset {offsets[split_index - 1]!r} crosses {target_start}). "
            "Masking it would drop target characters and supervising it would "
            "train on prompt text; inspect this row before proceeding.")
    if offsets[split_index][0] != target_start:
        raise FormattingError(
            f"{row_id}: first assistant token starts at {offsets[split_index][0]} "
            f"but the target starts at {target_start}; offset mapping is not "
            "contiguous with the rendered text")
    if offsets[-1][1] != len(full_text):
        raise FormattingError(
            f"{row_id}: last token ends at {offsets[-1][1]} but the rendered text "
            f"is {len(full_text)} characters; refusing to guess the target extent")

    labels = [-100] * split_index + input_ids[split_index:]
    if all(label == -100 for label in labels):
        raise FormattingError(f"{row_id}: loss mask covers every token")
    return FormattedExample(
        row_id=row_id,
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        prompt_tokens=split_index,
        supervised_tokens=len(input_ids) - split_index,
        total_tokens=len(input_ids),
        full_text_sha256=hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
        template_sha256=template.sha256,
    )
