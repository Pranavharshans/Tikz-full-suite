"""Batched generation for base-model and checkpoint evaluation.

The prompt-construction and completion-decoding helpers are pure and unit
tested; ``generate_records`` is the GPU path exercised by the preflight and the
evaluation gates.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .errors import DataError
from .formatting import ChatTemplate, render_prompt
from .util import sha256_text


@dataclass(frozen=True)
class GenerationRecord:
    row_id: str
    source_row_index: int
    instruction_sha256: str
    raw_output: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    batch_size: int
    finish_reason: str      # stop | length | error
    error: str | None

    def to_jsonable(self) -> dict:
        return {
            "row_id": self.row_id,
            "source_row_index": self.source_row_index,
            "instruction_sha256": self.instruction_sha256,
            "raw_output": self.raw_output,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_s": round(self.latency_s, 4),
            "batch_size": self.batch_size,
            "finish_reason": self.finish_reason,
            "error": self.error,
        }


def build_generation_prompt(tokenizer, instruction: str, template: ChatTemplate,
                            kwargs: dict) -> str:
    """Exactly the prompt the model was trained to continue."""
    return render_prompt(tokenizer, instruction=instruction, template=template,
                         kwargs=kwargs)


def decode_completion(tokenizer, generated_ids, prompt_length: int) -> tuple:
    """Return ``(text, finish_reason, output_tokens)`` for one sequence."""
    completion = list(generated_ids)[prompt_length:]
    eos_ids = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        eos_ids.add(eos)
    elif isinstance(eos, (list, tuple)):
        eos_ids.update(int(item) for item in eos)
    finish_reason = "stop" if any(token in eos_ids for token in completion) else "length"
    text = tokenizer.decode(completion, skip_special_tokens=True)
    return text, finish_reason, len(completion)


def generate_records(model, tokenizer, rows, *, template: ChatTemplate,
                     kwargs: dict, evaluation, max_examples=None,
                     progress=None) -> list:
    """Generate one completion per row; greedy by default."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataError("torch is required for generation") from exc

    rows = list(rows)
    if max_examples is not None:
        rows = rows[:max_examples]
    if not rows:
        raise DataError("No rows to evaluate")

    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    records = []
    batch_size = max(1, evaluation.batch_size)
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                prompts = [build_generation_prompt(
                    tokenizer, row["instruction"], template, kwargs) for row in batch]
                encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                                    add_special_tokens=False)
                device = next(model.parameters()).device
                encoded = {key: value.to(device) for key, value in encoded.items()}
                input_tokens = int(encoded["input_ids"].shape[1])
                begin = time.monotonic()
                generated = model.generate(
                    **encoded,
                    max_new_tokens=evaluation.max_new_tokens,
                    do_sample=evaluation.do_sample,
                    temperature=evaluation.temperature if evaluation.do_sample else None,
                    top_p=evaluation.top_p if evaluation.do_sample else None,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id)
                elapsed = time.monotonic() - begin
                per_row_latency = elapsed / len(batch)
                for index, row in enumerate(batch):
                    text, finish_reason, output_tokens = decode_completion(
                        tokenizer, generated[index], input_tokens)
                    records.append(GenerationRecord(
                        row_id=row["row_id"],
                        source_row_index=row["source_row_index"],
                        instruction_sha256=sha256_text(row["instruction"]),
                        raw_output=text,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        latency_s=per_row_latency,
                        batch_size=len(batch),
                        finish_reason=finish_reason,
                        error=None))
                if progress:
                    progress(f"generated {len(records)}/{len(rows)}")
    except RuntimeError as exc:
        raise DataError(f"Generation failed: {type(exc).__name__}: {exc}") from exc
    finally:
        tokenizer.padding_side = previous_padding_side
    return records
