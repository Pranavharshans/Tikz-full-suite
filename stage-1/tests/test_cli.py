"""CLI contract tests: help for every entry point, Slurm generation, comparison."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests import support

STAGE1 = Path(__file__).resolve().parents[1]
SCRIPTS = STAGE1 / "scripts"
ENTRY_POINTS = (
    "prepare_dataset.py", "preflight.py", "train.py", "evaluate.py",
    "compare_models.py", "merge_adapter.py", "make_slurm_script.py",
)


def run_script(name, *args, env=None):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        capture_output=True, text=True, cwd=str(STAGE1), env=env, timeout=120)


class HelpTests(unittest.TestCase):
    def test_every_entry_point_supports_help(self):
        for name in ENTRY_POINTS:
            completed = run_script(name, "--help")
            self.assertEqual(completed.returncode, 0,
                             f"{name} --help failed: {completed.stderr}")
            self.assertIn("usage", completed.stdout.lower(), name)


class TrainArgumentTests(unittest.TestCase):
    def test_unknown_gate_is_rejected(self):
        completed = run_script("train.py", "--gate", "tiny",
                               "--run-dir", "/tmp/x")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("invalid choice", completed.stderr)

    def test_help_explains_gate_separation(self):
        completed = run_script("train.py", "--help")
        self.assertIn("separate commands", " ".join(completed.stdout.split()))


class SlurmScriptTests(unittest.TestCase):
    @support.requires_yaml
    def test_generated_script_never_submits_and_is_valid_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed = run_script(
                "make_slurm_script.py",
                "--config", str(STAGE1 / "configs" / "minicpm5-2b-full.yaml"),
                "--export", str(root / "export"),
                "--prepared", str(root / "prepared"),
                "--run-dir", str(root / "run"),
                "--gate", "smoke-1000",
                "--python", sys.executable,
                "--wall-time", "6:00:00")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            script = completed.stdout
            self.assertIn("#SBATCH --gres=gpu:rtxpro6k:1", script)
            self.assertIn("--gate smoke-1000", script)
            self.assertIn("--time=6:00:00", script)
            # The only mentions of sbatch must be in comments or echo text.
            for line in script.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith("echo"):
                    continue
                self.assertNotIn("sbatch", stripped,
                                 f"non-comment line invokes sbatch: {line!r}")
            if support.HAS_BASH:
                path = root / "script.sbatch"
                path.write_text(script)
                check = subprocess.run(["bash", "-n", str(path)],
                                       capture_output=True, text=True)
                self.assertEqual(check.returncode, 0, check.stderr)

    @support.requires_yaml
    def test_no_preflight_first_comments_out_the_preflight_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed = run_script(
                "make_slurm_script.py",
                "--config", str(STAGE1 / "configs" / "minicpm5-2b-full.yaml"),
                "--export", str(root / "export"),
                "--prepared", str(root / "prepared"),
                "--run-dir", str(root / "run"),
                "--gate", "full",
                "--python", sys.executable,
                "--no-preflight-first")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            for line in completed.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                self.assertNotIn("preflight.py", stripped)

    @support.requires_yaml
    def test_relative_paths_are_rejected(self):
        completed = run_script(
            "make_slurm_script.py",
            "--config", str(STAGE1 / "configs" / "minicpm5-2b-full.yaml"),
            "--export", "relative/export",
            "--prepared", "/tmp/prepared",
            "--run-dir", "/tmp/run",
            "--gate", "full",
            "--python", sys.executable)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("absolute", completed.stderr)

    @support.requires_yaml
    def test_local_files_only_sets_offline_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed = run_script(
                "make_slurm_script.py",
                "--config", str(STAGE1 / "configs" / "minicpm5-2b-full.yaml"),
                "--export", str(root / "export"),
                "--prepared", str(root / "prepared"),
                "--run-dir", str(root / "run"),
                "--gate", "full",
                "--python", sys.executable,
                "--local-files-only")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("HF_HUB_OFFLINE=1", completed.stdout)
            self.assertIn("--local-files-only", completed.stdout)


class CompareModelsTests(unittest.TestCase):
    def _write_metrics(self, path, kind, data_identity="1" * 64, split="test",
                       training_method=None, evaluation_set="2" * 64,
                       max_seq_len=4096):
        template = {
            "schema_version": "stage1-eval-v1",
            "artifact": {"kind": kind},
            "training_method": training_method or ("base" if kind == "base" else "full"),
            "data_identity_sha256": data_identity,
            "max_seq_len": max_seq_len,
            "evaluation_set_sha256": evaluation_set,
            "split": split,
            "generation": {"examples": 10, "completed": 4, "truncated": 6,
                           "errors": 0, "empty": 1},
            "compilation": {"attempted": 10, "success": 2,
                            "success_rate": 0.2, "categories": {}},
            "latency": {"mean": 2.0, "tokens_per_second": 100.0},
            "duplicates": {"duplicate_rows": 5},
            "memorization": {"exact_matches": 0, "near_matches": 1},
        }
        Path(path).write_text(json.dumps(template))

    def test_comparison_cli_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.json"
            trained = root / "trained.json"
            self._write_metrics(base, "base")
            self._write_metrics(trained, "final")
            out = root / "comparison"
            completed = run_script(
                "compare_models.py",
                "--metrics", str(base), "--metrics", str(trained),
                "--labels", "base", "trained", "--out", str(out))
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((out / "comparison.json").is_file())
            self.assertTrue((out / "comparison.md").is_file())
            markdown = (out / "comparison.md").read_text()
            self.assertIn("base", markdown)
            self.assertIn("trained", markdown)
            self.assertIn("READY", markdown)

    def test_incomplete_comparison_exits_one_unless_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trained = root / "trained.json"
            self._write_metrics(trained, "final")
            out = root / "comparison"
            completed = run_script(
                "compare_models.py", "--metrics", str(trained),
                "--labels", "trained", "--out", str(out))
            self.assertEqual(completed.returncode, 1)
            self.assertIn("NOT READY", completed.stderr + completed.stdout)
            allowed = run_script(
                "compare_models.py", "--metrics", str(trained),
                "--labels", "trained", "--out", str(out),
                "--allow-incomplete")
            self.assertEqual(allowed.returncode, 0, allowed.stderr)

    def test_mixed_method_comparison_is_refused_unless_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = root / "full.json"
            lora = root / "lora.json"
            self._write_metrics(full, "final", training_method="full")
            self._write_metrics(lora, "final", training_method="lora")
            out = root / "comparison"
            completed = run_script(
                "compare_models.py", "--metrics", str(full), "--metrics", str(lora),
                "--labels", "full", "lora", "--out", str(out))
            self.assertEqual(completed.returncode, 1)
            self.assertIn("mixes training methods", completed.stderr)
            allowed = run_script(
                "compare_models.py", "--metrics", str(full), "--metrics", str(lora),
                "--labels", "full", "lora", "--out", str(out),
                "--allow-non-comparable",
                "--allow-incomplete")
            self.assertEqual(allowed.returncode, 0, allowed.stderr)
            markdown = (out / "comparison.md").read_text()
            self.assertIn("NOT COMPARABLE", markdown)

    def test_label_count_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "m.json"
            metrics.write_text(json.dumps({"schema_version": "stage1-eval-v1"}))
            completed = run_script(
                "compare_models.py", "--metrics", str(metrics),
                "--labels", "a", "b", "--out", str(root / "out"))
            self.assertEqual(completed.returncode, 2)
            self.assertIn("must match", completed.stderr)


class MergeCommandTests(unittest.TestCase):
    def test_training_never_merges_adapters(self):
        source = (STAGE1 / "src" / "stage1" / "train.py").read_text()
        self.assertNotIn("merge_and_unload", source)
        self.assertNotIn("merge_adapter", source)

    def test_merge_cli_help(self):
        completed = run_script("merge_adapter.py", "--help")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("never automatic", " ".join(completed.stdout.split()))

    @support.requires_yaml
    def test_merge_refuses_a_full_mode_config(self):
        completed = run_script(
            "merge_adapter.py",
            "--config", str(STAGE1 / "configs" / "qwen3.5-4b-full.yaml"),
            "--run-dir", "/tmp/run", "--adapter", "/tmp/adapter",
            "--out", "/tmp/out")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("requires training.method: lora", completed.stderr)


class ExitCodeTests(unittest.TestCase):
    """The documented exit-code contract, without subprocesses."""

    def raise_from(self, exception):
        def main(argv):
            raise exception
        return main

    def test_exit_codes(self):
        from stage1 import cli
        from stage1.errors import (ConfigError, DataError, GateError,
                                   GateFailed, IdentityMismatch)
        cases = [
            (GateFailed("criteria"), 1),
            (GateError("prerequisites"), 3),
            (ConfigError("bad config"), 2),
            (DataError("bad data"), 2),
            (IdentityMismatch(["seed differs"]), 2),
        ]
        for exception, expected in cases:
            with self.subTest(exception=type(exception).__name__):
                self.assertEqual(
                    cli.run(self.raise_from(exception)), expected)

    def test_success_is_zero(self):
        from stage1 import cli
        self.assertEqual(cli.run(lambda argv: 0), 0)


if __name__ == "__main__":
    unittest.main()
