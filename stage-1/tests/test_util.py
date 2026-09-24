"""Tests for stage1.util (hashing, atomic writes, locks, wall time)."""
import tempfile
import unittest
from pathlib import Path

from stage1 import util
from stage1.errors import ConfigError


class DigestTests(unittest.TestCase):
    def test_canonical_digest_matches_cleaning_pipeline(self):
        import importlib.util
        cleaning = Path(__file__).resolve().parents[2] / "cleaning"
        spec = importlib.util.spec_from_file_location(
            "cleaning_benchmark_for_digest", cleaning / "benchmark.py")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"cannot import cleaning/benchmark.py: {exc}")
        samples = [
            {"b": 1, "a": [1, 2, {"z": "ü"}]},
            ["x", None, True, 1.5],
            {"nested": {"unicode": "TikZ \\draw (0,0) -- (1,1);"}},
            [],
        ]
        for sample in samples:
            self.assertEqual(util.canonical_digest(sample), module.digest(sample))

    def test_canonical_digest_is_key_order_independent(self):
        self.assertEqual(util.canonical_digest({"a": 1, "b": 2}),
                         util.canonical_digest({"b": 2, "a": 1}))

    def test_canonical_digest_detects_changes(self):
        self.assertNotEqual(util.canonical_digest({"a": 1}),
                            util.canonical_digest({"a": 2}))


class AtomicWriteTests(unittest.TestCase):
    def test_write_json_atomic_round_trip_and_no_temp_left(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "artifact.json"
            util.write_json_atomic(path, {"b": 2, "a": 1})
            self.assertEqual(util.read_json(path), {"a": 1, "b": 2})
            leftovers = [item.name for item in path.parent.iterdir()
                         if item.name.startswith(".")]
            self.assertEqual(leftovers, [])

    def test_write_text_atomic_replaces_existing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file.txt"
            util.write_text_atomic(path, "one")
            util.write_text_atomic(path, "two")
            self.assertEqual(path.read_text(), "two")


class LockFileTests(unittest.TestCase):
    def test_parse_lock_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env.lock"
            path.write_text("# comment\ntorch==2.12.1\npyyaml==6.0.3\n\n")
            self.assertEqual(util.parse_lock_file(path),
                             {"torch": "2.12.1", "pyyaml": "6.0.3"})

    def test_parse_lock_file_rejects_unpinned_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env.lock"
            path.write_text("torch>=2.0\n")
            with self.assertRaisesRegex(ConfigError, "name==version"):
                util.parse_lock_file(path)

    def test_parse_lock_file_rejects_interpreter_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env.lock"
            path.write_text("python==3.12\n")
            with self.assertRaisesRegex(ConfigError, "python-version"):
                util.parse_lock_file(path)

    def test_parse_lock_file_rejects_invalid_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env.lock"
            path.write_text("not a package==1.0\n")
            with self.assertRaisesRegex(ConfigError, "distribution name"):
                util.parse_lock_file(path)

    def test_parse_lock_file_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env.lock"
            path.write_text("torch==2.0\ntorch==2.1\n")
            with self.assertRaisesRegex(ConfigError, "duplicate"):
                util.parse_lock_file(path)

    def test_verify_lock_versions_reports_differences(self):
        pins = {"torch": "2.12.1", "unknown-pkg": "1.0"}
        installed = {"torch": "2.12.1"}
        differences = util.verify_lock_versions(pins, installed)
        self.assertEqual(len(differences), 1)
        self.assertIn("unknown-pkg", differences[0])

    def test_verify_lock_versions_detects_mismatch(self):
        differences = util.verify_lock_versions(
            {"torch": "9.9.9"}, {"torch": "2.12.1"})
        self.assertEqual(differences, ["torch: pinned 9.9.9, installed 2.12.1"])

    def test_shipped_locks_are_installable_requirements(self):
        locks = Path(__file__).resolve().parents[1] / "locks"
        for name in ("qwen3.5-4b.lock", "minicpm5-2b.lock"):
            path = locks / name
            lines = [line.strip() for line in path.read_text().splitlines()
                     if line.strip() and not line.strip().startswith("#")]
            self.assertTrue(lines, name)
            for line in lines:
                self.assertRegex(line, r"^[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9][A-Za-z0-9._+!-]*$",
                                 f"{name}: {line!r} is not an installable pin")
                self.assertFalse(line.lower().startswith("python=="),
                                 f"{name} still pins the interpreter")
            pins = util.parse_lock_file(path)
            self.assertIn("torch", pins)
            self.assertIn("transformers", pins)
            expected_transformers = {
                "qwen3.5-4b.lock": "5.5.0",
                "minicpm5-2b.lock": "4.57.3",
            }
            self.assertEqual(pins["transformers"], expected_transformers[name])
            self.assertNotIn("python", pins)
            self.assertNotIn("flash-attn", pins)

    def test_every_lock_pin_is_verifiable(self):
        locks = Path(__file__).resolve().parents[1] / "locks"
        for name in ("qwen3.5-4b.lock", "minicpm5-2b.lock"):
            pins = util.parse_lock_file(locks / name)
            for distribution in pins:
                self.assertIsNotNone(
                    util.distribution_import_name(distribution),
                    f"{name}: {distribution} has no import-name mapping, so the "
                    "preflight could not verify it")

    def test_verify_lock_versions_accepts_underscore_and_dash_names(self):
        differences = util.verify_lock_versions(
            {"unsloth_zoo": "1.0"}, {"unsloth_zoo": "1.0"})
        self.assertEqual(differences, [])

    def test_repository_python_pin_exists_and_matches_configs(self):
        stage1 = Path(__file__).resolve().parents[1]
        pin = (stage1 / ".python-version").read_text().strip()
        self.assertRegex(pin, r"^3\.\d+$")
        self.assertEqual(util.python_version_pin(), pin)


class WallTimeTests(unittest.TestCase):
    def test_valid_formats(self):
        self.assertEqual(util.parse_wall_time("90"), 90)
        self.assertEqual(util.parse_wall_time("5:00"), 5)
        self.assertEqual(util.parse_wall_time("00:30:00"), 30)
        self.assertEqual(util.parse_wall_time("12:00:00"), 720)
        self.assertEqual(util.parse_wall_time("1-00:00:00"), 1440)
        self.assertEqual(util.parse_wall_time("0:00:30"), 1)

    def test_invalid_formats(self):
        for value in ("", "abc", "12:99:00", "0", "1-25:00:00", "1:2:3:4"):
            with self.assertRaises(ConfigError):
                util.parse_wall_time(value)


class PathAndIdentityTests(unittest.TestCase):
    def test_require_absolute_rejects_relative(self):
        with self.assertRaisesRegex(ConfigError, "absolute"):
            util.require_absolute("relative/path", "--export")

    def test_source_tree_sha256_is_deterministic_and_sensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.py").write_text("x = 1\n")
            (root / "sub").mkdir()
            (root / "sub" / "b.py").write_text("y = 2\n")
            first = util.source_tree_sha256(root)
            self.assertEqual(first, util.source_tree_sha256(root))
            (root / "sub" / "b.py").write_text("y = 3\n")
            self.assertNotEqual(first, util.source_tree_sha256(root))

    def test_dependency_versions_reports_python(self):
        versions = util.dependency_versions()
        self.assertIn("python", versions)
        self.assertRegex(versions["python"], r"^\d+\.\d+\.\d+$")
        self.assertIn("torch", versions)

    def test_pinned_revision_and_sha256_helpers(self):
        self.assertTrue(util.is_pinned_revision("a" * 40))
        self.assertFalse(util.is_pinned_revision("main"))
        self.assertTrue(util.is_sha256("b" * 64))
        self.assertFalse(util.is_sha256("b" * 63))

    def test_human_bytes(self):
        self.assertEqual(util.human_bytes(512), "512.0 B")
        self.assertEqual(util.human_bytes(1024 ** 3), "1.0 GiB")


if __name__ == "__main__":
    unittest.main()
