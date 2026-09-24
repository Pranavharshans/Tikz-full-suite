"""GPU-marked integration tests.

These tests run only on a machine with CUDA and the target GPU; ordinary CPU
test runs skip them. They are the local half of the RTX PRO 6000 preflight
evidence and never load models or start training.
"""
import unittest
from pathlib import Path

from tests import support

from stage1 import preflight


def context(**overrides):
    config = support.make_config(**overrides)
    return preflight.PreflightContext(
        config=config, run_dir=Path("/tmp/stage1-gpu-test"),
        identity={"data": {"data_identity_sha256": "0" * 64}},
        prepared={"data_identity_sha256": "0" * 64, "manifest": {
            "data_identity": {"dataset_logical_sha256": "1" * 64}}})


@support.requires_cuda
class A40ProfileTests(unittest.TestCase):
    """The A40 LoRA hardware profile, verified on a real A40."""

    def test_a40_profile_passes_on_an_a40(self):
        import torch
        name = torch.cuda.get_device_properties(0).name
        if "A40" not in name:
            self.skipTest(f"device {name!r} is not an A40")
        config = support.make_config(hardware={"profile": "a40-lora"})
        ctx = preflight.PreflightContext(
            config=config, run_dir=Path("/tmp/stage1-a40-test"),
            identity={"data": {"data_identity_sha256": "0" * 64}},
            prepared={"data_identity_sha256": "0" * 64, "manifest": {
                "data_identity": {"dataset_logical_sha256": "1" * 64}}})
        outcome = preflight.check_gpu_identity(ctx)
        self.assertEqual(outcome["status"], "pass", outcome["detail"])
        self.assertGreaterEqual(outcome["data"]["vram_gib"], 44.0)


@support.requires_cuda
class GpuIdentityTests(unittest.TestCase):
    def setUp(self):
        import torch
        name = torch.cuda.get_device_properties(0).name
        if "RTX PRO 6000" not in name:
            self.skipTest(f"device {name!r} is not an RTX PRO 6000")

    def test_gpu_identity_check_passes(self):
        outcome = preflight.check_gpu_identity(context())
        self.assertEqual(outcome["status"], "pass", outcome["detail"])
        self.assertGreaterEqual(outcome["data"]["vram_gib"], 88)

    def test_bf16_is_available(self):
        self.assertEqual(preflight.check_bf16(context())["status"], "pass")

    def test_torch_cuda_kernel_smoke(self):
        self.assertEqual(preflight.check_torch_cuda(context())["status"], "pass")

    def test_fused_adamw_constructs(self):
        self.assertEqual(preflight.check_fused_optimizer(context())["status"], "pass")


if __name__ == "__main__":
    unittest.main()
