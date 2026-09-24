"""Cross-checks against the cleaning pipeline's exact checksum contract.

These tests import ``cleaning/build_dataset.py`` and
``cleaning/benchmark.py`` read-only. If the import fails (for example a
missing optional dependency at module import time), they skip.
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from tests import support

from stage1 import data as data_module
from stage1 import util

CLEANING = Path(__file__).resolve().parents[2] / "cleaning"


def load_cleaning_module(name, filename):
    import sys
    spec = importlib.util.spec_from_file_location(name, CLEANING / filename)
    module = importlib.util.module_from_spec(spec)
    # Python's dataclasses resolve the defining module through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


try:
    BUILD = load_cleaning_module("cleaning_build_dataset_crosscheck",
                                 "build_dataset.py")
    BENCH = load_cleaning_module("cleaning_benchmark_crosscheck", "benchmark.py")
    HAS_CLEANING = True
except Exception as exc:  # pragma: no cover - environment dependent
    HAS_CLEANING = False
    CLEANING_IMPORT_ERROR = str(exc)


@unittest.skipUnless(HAS_CLEANING, "cleaning modules are not importable")
class ChecksumContractTests(unittest.TestCase):
    def sample_rows(self):
        return [
            {"id": "a" * 32, "source_row_index": 0, "image_sha256": "b" * 64,
             "tikz_sha256": "c" * 64, "instruction": "Draw a circle.",
             "prompt_version": "caption-v1"},
            {"id": "d" * 32, "source_row_index": 1, "image_sha256": "e" * 64,
             "tikz_sha256": "f" * 64,
             "instruction": "Draw a labeled diagram with ünïcode symbols: $\\alpha$.",
             "prompt_version": "caption-v2"},
        ]

    def test_logical_row_encoding_is_identical(self):
        for row in self.sample_rows():
            self.assertEqual(data_module.logical_row_json(row), BUILD.logical_row(row))

    def test_shard_logical_checksum_is_identical(self):
        rows = self.sample_rows()
        self.assertEqual(data_module.shard_logical_checksum(rows),
                         BUILD.logical_checksum(rows))

    def test_dataset_logical_digest_is_identical(self):
        shard_hashes = [data_module.shard_logical_checksum(self.sample_rows()),
                        "0" * 64]
        self.assertEqual(util.canonical_digest(shard_hashes),
                         BUILD.canonical_digest(shard_hashes))

    def test_canonical_digest_is_identical(self):
        for value in ({"a": 1, "b": [1, 2]}, ["x", None, True], {"ü": "TikZ"}):
            self.assertEqual(util.canonical_digest(value), BENCH.digest(value))


@support.requires_pyarrow
@unittest.skipUnless(HAS_CLEANING, "cleaning modules are not importable")
class FixtureContractTests(unittest.TestCase):
    def test_export_checksummed_by_cleaning_functions_verifies(self):
        """A fixture whose checksums come from cleaning's own code must verify."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = support.make_export(root / "export", shard_size=3)
            import pyarrow.parquet as parquet
            entries = []
            for record in export["shards"]:
                path = export["root"] / "shards" / record["name"]
                rows = parquet.read_table(path).to_pylist()
                entries.append({
                    "name": record["name"],
                    "rows": len(rows),
                    "logical_sha256": BUILD.logical_checksum(rows),
                    "file_sha256": util.sha256_file(path),
                })
            dataset_logical = BUILD.canonical_digest(
                [entry["logical_sha256"] for entry in entries])
            checksums = json.loads(
                (export["root"] / "checksums.json").read_text())
            checksums["shards"] = entries
            checksums["dataset_logical_sha256"] = dataset_logical
            (export["root"] / "checksums.json").write_text(json.dumps(checksums))
            meta = json.loads((export["root"] / "export.meta.json").read_text())
            meta["dataset_logical_sha256"] = dataset_logical
            (export["root"] / "export.meta.json").write_text(json.dumps(meta))
            provenance = json.loads(
                (export["root"] / "run-metadata.json").read_text())
            provenance["export"] = meta
            (export["root"] / "run-metadata.json").write_text(json.dumps(provenance))
            info = data_module.verify_export(export["root"])
            self.assertEqual(info.dataset_logical_sha256, dataset_logical)
            self.assertEqual(info.rows, 6)


if __name__ == "__main__":
    unittest.main()
