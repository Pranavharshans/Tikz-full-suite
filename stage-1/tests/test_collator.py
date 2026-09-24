"""Tests for padding, batch validation and loss-mask contracts.

Stage 1 does not support sequence packing; these tests cover the padded path
that the trainer actually uses.
"""
import unittest

from tests import support

from stage1 import collator
from stage1.errors import BatchError


def example(length, prompt=2, token_base=100):
    input_ids = tuple(token_base + index for index in range(length))
    labels = tuple([-100] * prompt + list(input_ids[prompt:]))
    return {"input_ids": input_ids, "labels": labels}


class PadBatchTests(unittest.TestCase):
    def test_padding_masks_and_positions(self):
        plan = collator.pad_batch([example(4), example(3)], pad_token_id=0)
        self.assertEqual(plan.rows, 2)
        self.assertEqual(plan.input_ids[0], (100, 101, 102, 103))
        self.assertEqual(plan.input_ids[1], (100, 101, 102, 0))
        self.assertEqual(plan.labels[1], (-100, -100, 102, -100))
        self.assertEqual(plan.attention_mask[1], (1, 1, 1, 0))
        self.assertEqual(plan.position_ids[1], (0, 1, 2, 0))
        self.assertEqual(plan.supervised_tokens, 2 + 1)
        self.assertEqual(plan.padded_tokens, 8)

    def test_pad_to_multiple_of(self):
        plan = collator.pad_batch([example(3)], pad_token_id=0,
                                  pad_to_multiple_of=4)
        self.assertEqual(len(plan.input_ids[0]), 4)

    def test_batch_plan_has_no_packing_fields(self):
        plan = collator.pad_batch([example(3)], pad_token_id=0)
        self.assertFalse(hasattr(plan, "segments"))

    def test_empty_batch_is_refused(self):
        with self.assertRaises(BatchError):
            collator.pad_batch([], pad_token_id=0)

    def test_length_mismatch_is_refused(self):
        with self.assertRaises(BatchError):
            collator.pad_batch([{"input_ids": [1, 2], "labels": [-100]}],
                               pad_token_id=0)


class ValidateBatchTests(unittest.TestCase):
    def test_valid_padded_batch_passes(self):
        plan = collator.pad_batch([example(5), example(3)], pad_token_id=0)
        self.assertEqual(collator.validate_batch_plan(plan, pad_token_id=0), [])

    def test_padding_label_violation_is_reported(self):
        plan = collator.pad_batch([example(3), example(2)], pad_token_id=0)
        broken = list(plan.labels[1])
        broken[2] = 42
        plan = collator.BatchPlan(
            plan.input_ids, (plan.labels[0], tuple(broken)), plan.attention_mask,
            plan.position_ids, plan.padded_tokens, plan.supervised_tokens)
        problems = collator.validate_batch_plan(plan, pad_token_id=0)
        self.assertTrue(any("padding label is not -100" in item for item in problems))

    def test_label_must_be_minus_100_or_input(self):
        plan = collator.pad_batch([example(4)], pad_token_id=0)
        broken = list(plan.labels[0])
        broken[2] = 999
        plan = collator.BatchPlan(
            plan.input_ids, (tuple(broken),), plan.attention_mask,
            plan.position_ids, plan.padded_tokens, plan.supervised_tokens)
        problems = collator.validate_batch_plan(plan, pad_token_id=0)
        self.assertTrue(any("neither -100 nor the input token" in item
                            for item in problems))

    def test_supervised_token_after_mask_is_reported(self):
        plan = collator.pad_batch([example(4, prompt=1)], pad_token_id=0)
        broken = (-100, 101, -100, 103)
        plan = collator.BatchPlan(
            plan.input_ids, (broken,), plan.attention_mask, plan.position_ids,
            plan.padded_tokens, plan.supervised_tokens)
        problems = collator.validate_batch_plan(plan, pad_token_id=0)
        self.assertTrue(any("after a supervised token" in item for item in problems))

    def test_position_ids_must_be_zero_in_padding(self):
        plan = collator.pad_batch([example(3), example(2)], pad_token_id=0)
        positions = (list(plan.position_ids[0]), [0, 1, 5])
        plan = collator.BatchPlan(
            plan.input_ids, plan.labels, plan.attention_mask,
            (tuple(positions[0]), tuple(positions[1])), plan.padded_tokens,
            plan.supervised_tokens)
        problems = collator.validate_batch_plan(plan, pad_token_id=0)
        self.assertTrue(any("padding position is not 0" in item for item in problems))

    def test_no_supervised_token_is_reported(self):
        plan = collator.pad_batch([example(3, prompt=3)], pad_token_id=0)
        problems = collator.validate_batch_plan(plan, pad_token_id=0)
        self.assertTrue(any("no supervised token" in item for item in problems))

    def test_assert_valid_batch_raises_with_detail(self):
        plan = collator.pad_batch([example(3)], pad_token_id=0)
        plan = collator.BatchPlan(
            plan.input_ids, ((-100, -100, -100),), plan.attention_mask,
            plan.position_ids, plan.padded_tokens, 0)
        with self.assertRaisesRegex(BatchError, "no supervised token"):
            collator.assert_valid_batch(plan, pad_token_id=0)


class BuildBatchTests(unittest.TestCase):
    def test_build_batch_is_padding_only(self):
        plan = collator.build_batch([example(3), example(5)], pad_token_id=0)
        self.assertEqual(plan.rows, 2)
        self.assertEqual(len(plan.input_ids[0]), 5)
        self.assertEqual(plan.supervised_tokens, 1 + 3)

    def test_build_batch_rejects_unknown_packing_argument(self):
        with self.assertRaises(TypeError):
            collator.build_batch([example(3)], pad_token_id=0, packing=True)


class CountingTests(unittest.TestCase):
    def test_count_supervised(self):
        self.assertEqual(collator.count_supervised([-100, 1, 2, -100]), 2)
        self.assertEqual(collator.count_supervised([-100]), 0)

    def test_collator_has_no_counter_hook(self):
        import inspect
        signature = inspect.signature(collator.make_torch_collator)
        self.assertNotIn("counter", signature.parameters)
        self.assertNotIn("packing", signature.parameters)


@support.requires_torch
class TorchCollatorTests(unittest.TestCase):
    def test_collator_produces_long_tensors(self):
        import torch
        collate = collator.make_torch_collator(
            torch, pad_token_id=0, validate=True)
        batch = collate([example(4), example(3)])
        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["labels"].dtype, torch.long)
        self.assertEqual(batch["attention_mask"].shape, (2, 4))
        self.assertEqual(batch["position_ids"].shape, (2, 4))

    def test_collator_validates_batches(self):
        import torch
        collate = collator.make_torch_collator(torch, pad_token_id=0, validate=True)
        with self.assertRaises(BatchError):
            collate([{"input_ids": [1, 2], "labels": [-100, -100]}])


if __name__ == "__main__":
    unittest.main()
