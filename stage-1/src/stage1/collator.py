"""Batching, padding and loss-mask validation.

The pure planning layer works on plain integer sequences so it is fully
testable without torch. ``make_torch_collator`` is the thin tensor conversion
used by the trainer; it imports nothing at module scope and never mutates
training state.

Stage 1 does not support sequence packing: resetting ``position_ids`` under a
normal all-ones attention mask does not stop attention across examples, and a
correct implementation would need a backend-specific varlen path that has not
been validated. Batches are right-padded; padding gets ``-100`` labels and
attention ``0``.
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import BatchError


@dataclass(frozen=True)
class BatchPlan:
    input_ids: tuple
    labels: tuple
    attention_mask: tuple
    position_ids: tuple
    padded_tokens: int
    supervised_tokens: int

    @property
    def rows(self) -> int:
        return len(self.input_ids)


def count_supervised(labels) -> int:
    """Number of tokens that carry loss (label != -100)."""
    return sum(1 for label in labels if label != -100)


def _ids_and_labels(example):
    if isinstance(example, dict):
        return list(example["input_ids"]), list(example["labels"])
    return list(example.input_ids), list(example.labels)


def pad_batch(examples, *, pad_token_id: int,
              pad_to_multiple_of: int | None = None) -> BatchPlan:
    """Right-pad a batch of already-formatted examples."""
    if not examples:
        raise BatchError("Cannot build an empty batch")
    sequences = []
    for example in examples:
        input_ids, labels = _ids_and_labels(example)
        if len(input_ids) != len(labels):
            raise BatchError("input_ids and labels length mismatch")
        sequences.append((input_ids, labels))
    width = max(len(ids) for ids, _ in sequences)
    if pad_to_multiple_of:
        remainder = width % pad_to_multiple_of
        if remainder:
            width += pad_to_multiple_of - remainder

    input_rows, label_rows, masks, positions = [], [], [], []
    supervised = 0
    for input_ids, labels in sequences:
        padding = width - len(input_ids)
        input_rows.append(tuple(input_ids + [pad_token_id] * padding))
        label_rows.append(tuple(labels + [-100] * padding))
        masks.append(tuple([1] * len(input_ids) + [0] * padding))
        positions.append(tuple(range(len(input_ids))) + tuple([0] * padding))
        supervised += count_supervised(labels)
    return BatchPlan(
        input_ids=tuple(input_rows), labels=tuple(label_rows),
        attention_mask=tuple(masks), position_ids=tuple(positions),
        padded_tokens=width * len(sequences), supervised_tokens=supervised)


def build_batch(examples, *, pad_token_id: int,
                pad_to_multiple_of: int | None = None) -> BatchPlan:
    return pad_batch(examples, pad_token_id=pad_token_id,
                     pad_to_multiple_of=pad_to_multiple_of)


def validate_batch_plan(plan: BatchPlan, *, pad_token_id: int) -> list:
    """Return a list of violations; an empty list means the batch is sound.

    Per row: padding is ``pad_token_id`` with label ``-100``, attention ``0``
    and position ``0``; real tokens carry either ``-100`` or their own token as
    label; the masked region is a prefix of the real region (no ``-100`` after
    a supervised token); and every row has at least one supervised token.
    """
    problems = []
    for row in range(plan.rows):
        input_ids = plan.input_ids[row]
        labels = plan.labels[row]
        mask = plan.attention_mask[row]
        positions = plan.position_ids[row]
        if not (len(input_ids) == len(labels) == len(mask) == len(positions)):
            problems.append(f"row {row}: sequence fields have different lengths")
            continue
        if any(value not in (0, 1) for value in mask):
            problems.append(f"row {row}: attention mask must be 0/1")
        seen_supervised = False
        for index in range(len(input_ids)):
            padded = mask[index] == 0
            if padded:
                if input_ids[index] != pad_token_id:
                    problems.append(f"row {row}[{index}]: padding token mismatch")
                if labels[index] != -100:
                    problems.append(f"row {row}[{index}]: padding label is not -100")
                if positions[index] != 0:
                    problems.append(f"row {row}[{index}]: padding position is not 0")
                continue
            if labels[index] == -100:
                if seen_supervised:
                    problems.append(
                        f"row {row}[{index}]: -100 label after a supervised token")
                    break
            elif labels[index] != input_ids[index]:
                problems.append(
                    f"row {row}[{index}]: label is neither -100 nor the input token")
            else:
                seen_supervised = True
        if not seen_supervised:
            problems.append(f"row {row}: no supervised token")
    return problems


def assert_valid_batch(plan: BatchPlan, *, pad_token_id: int) -> None:
    problems = validate_batch_plan(plan, pad_token_id=pad_token_id)
    if problems:
        detail = "\n".join(f"  - {line}" for line in problems[:10])
        raise BatchError(f"Batch plan violates the loss-mask contract:\n{detail}")


def make_torch_collator(torch, *, pad_token_id: int, validate: bool,
                        pad_to_multiple_of: int | None = None):
    """Return a collator callable producing torch tensors for the Trainer.

    The collator is pure: it does not touch counters or metrics. Supervised
    token accounting happens in the trainer where evaluation and prefetched
    batches cannot contaminate it.
    """

    def collate(features):
        plan = build_batch(features, pad_token_id=pad_token_id,
                           pad_to_multiple_of=pad_to_multiple_of)
        if validate:
            assert_valid_batch(plan, pad_token_id=pad_token_id)
        return {
            "input_ids": torch.tensor(plan.input_ids, dtype=torch.long),
            "labels": torch.tensor(plan.labels, dtype=torch.long),
            "attention_mask": torch.tensor(plan.attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(plan.position_ids, dtype=torch.long),
        }

    return collate
