"""Import safety: the Stage 1 package must stay importable without ML libraries.

This mirrors the cleaning pipeline's convention: configuration validation,
dataset preparation, TeX tooling and CLI help must work on a login node or a
laptop, so importing any ``stage1`` module must not pull in torch,
transformers, unsloth, datasets, pyarrow, Pillow or numpy.
"""
import subprocess
import sys
import unittest

STAGE1 = __import__("pathlib").Path(__file__).resolve().parents[1]

FORBIDDEN = ("torch", "transformers", "unsloth", "datasets", "pyarrow", "PIL",
             "numpy", "trl", "accelerate", "safetensors", "bitsandbytes")

PROBE = """
import sys
sys.path.insert(0, {src!r})
import stage1.adapters
import stage1.checkpointing
import stage1.cli
import stage1.collator
import stage1.compile_tikz
import stage1.config
import stage1.data
import stage1.errors
import stage1.evaluate
import stage1.formatting
import stage1.generate
import stage1.identity
import stage1.preflight
import stage1.report
import stage1.schema
import stage1.train
import stage1.util
loaded = sorted(name for name in {forbidden!r} if name in sys.modules)
print(",".join(loaded))
"""


class ImportSafetyTests(unittest.TestCase):
    def test_package_imports_without_ml_libraries(self):
        completed = subprocess.run(
            [sys.executable, "-c", PROBE.format(
                src=str(STAGE1 / "src"), forbidden=FORBIDDEN)],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        loaded = [name for name in completed.stdout.strip().split(",") if name]
        self.assertEqual(loaded, [],
                         f"importing stage1 eagerly imported {loaded}")

    def test_scripts_support_help_without_ml_libraries(self):
        scripts = STAGE1 / "scripts"
        for name in ("prepare_dataset.py", "preflight.py", "train.py",
                     "evaluate.py", "compare_models.py", "make_slurm_script.py"):
            completed = subprocess.run(
                [sys.executable, str(scripts / name), "--help"],
                capture_output=True, text=True, timeout=120,
                env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"})
            self.assertEqual(completed.returncode, 0,
                             f"{name}: {completed.stderr}")


if __name__ == "__main__":
    unittest.main()
