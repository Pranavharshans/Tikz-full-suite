"""Tests for export verification, duplicate-safe splitting and preparation."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests import support

from stage1 import data as data_module
from stage1 import schema
from stage1.errors import DataError


@support.requires_pyarrow
class VerifyExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.export = support.make_export(self.root / "export", shard_size=2)

    def tearDown(self):
        self.temporary.cleanup()

    def rewrite_shard(self, shard_name, mutate, *, fix_file_hash=True):
        """Rewrite a shard; by default keep its recorded file hash consistent so
        the content/logical checks are the ones under test."""
        import pyarrow as pa
        import pyarrow.parquet as parquet
        from stage1.util import sha256_file
        path = self.export["root"] / "shards" / shard_name
        table = parquet.read_table(path)
        rows = table.to_pylist()
        mutate(rows)
        parquet.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        if fix_file_hash:
            checksums_path = self.export["root"] / "checksums.json"
            checksums = json.loads(checksums_path.read_text())
            for entry in checksums["shards"]:
                if entry["name"] == shard_name:
                    entry["file_sha256"] = sha256_file(path)
            checksums_path.write_text(json.dumps(checksums))

    def test_verified_export_reports_expected_facts(self):
        info = data_module.verify_export(self.export["root"])
        self.assertEqual(info.rows, 6)
        self.assertEqual(len(info.shards), 3)
        self.assertEqual(info.dataset_logical_sha256,
                         self.export["dataset_logical_sha256"])
        self.assertEqual(info.source_dataset, "nllg/DaTikZ-V4")
        self.assertEqual(info.prompt_version, "caption-v1")
        self.assertEqual(info.complete_rows, 6)
        self.assertEqual(info.rejected_rows, 0)

    def test_quick_mode_skips_file_and_content_hashes(self):
        info = data_module.verify_export(self.export["root"], quick=True)
        self.assertEqual(info.rows, 6)

    def test_tampered_file_hash_is_detected(self):
        path = self.export["root"] / "shards" / "shard-00000.parquet"
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 0xFF
        path.write_bytes(bytes(payload))
        with self.assertRaisesRegex(DataError, "file hash mismatch"):
            data_module.verify_export(self.export["root"])

    def test_tampered_content_is_detected_by_logical_checksum(self):
        self.rewrite_shard("shard-00000.parquet",
                           lambda rows: rows[0].update(instruction="changed"))
        with self.assertRaisesRegex(DataError, "logical checksum mismatch"):
            data_module.verify_export(self.export["root"])

    def test_tikz_content_hash_drift_is_detected(self):
        self.rewrite_shard("shard-00000.parquet",
                           lambda rows: rows[0].update(tikz_code="\\draw changed;"))
        with self.assertRaisesRegex(DataError, "TikZ content"):
            data_module.verify_export(self.export["root"])

    def test_image_content_hash_drift_is_detected(self):
        self.rewrite_shard("shard-00000.parquet",
                           lambda rows: rows[0].update(png_image=b"\x89PNG\r\n\x1a\nx"))
        with self.assertRaisesRegex(DataError, "image content"):
            data_module.verify_export(self.export["root"])

    def test_missing_metadata_is_refused(self):
        (self.export["root"] / "export.meta.json").unlink()
        with self.assertRaisesRegex(DataError, "export.meta.json is missing"):
            data_module.verify_export(self.export["root"])

    def test_extra_shard_file_is_refused(self):
        shutil.copy(self.export["root"] / "shards" / "shard-00000.parquet",
                    self.export["root"] / "shards" / "shard-99999.parquet")
        with self.assertRaisesRegex(DataError, "do not match"):
            data_module.verify_export(self.export["root"])

    def test_incomplete_export_is_refused(self):
        meta_path = self.export["root"] / "export.meta.json"
        meta = json.loads(meta_path.read_text())
        meta["complete_rows"] = meta["rows"] - 1
        meta_path.write_text(json.dumps(meta))
        provenance_path = self.export["root"] / "run-metadata.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["export"] = meta
        provenance_path.write_text(json.dumps(provenance))
        with self.assertRaisesRegex(DataError, "incomplete"):
            data_module.verify_export(self.export["root"])

    def test_ordering_violation_is_detected(self):
        def swap(rows):
            rows[0]["source_row_index"], rows[1]["source_row_index"] = (
                rows[1]["source_row_index"], rows[0]["source_row_index"])
        self.rewrite_shard("shard-00000.parquet", swap)
        with self.assertRaises(DataError):
            data_module.verify_export(self.export["root"])

    def test_rejected_source_index_gap_is_allowed(self):
        rows = support.default_rows()
        for index, row in enumerate(rows):
            row["source_row_index"] = index if index < 3 else index + 1
        export = support.make_export(self.root / "gap-export", rows=rows,
                                     rejected_rows=1)
        info = data_module.verify_export(export["root"])
        self.assertEqual(info.rows, len(rows))
        self.assertEqual(info.rejected_rows, 1)

    def test_duplicate_row_ids_are_detected(self):
        import pyarrow as pa
        import pyarrow.parquet as parquet
        first = parquet.read_table(self.export["root"] / "shards" / "shard-00000.parquet")
        rows = first.to_pylist()
        rows[1]["id"] = rows[0]["id"]
        parquet.write_table(pa.Table.from_pylist(rows, schema=first.schema),
                            self.export["root"] / "shards" / "shard-00000.parquet")
        with self.assertRaises(DataError):
            data_module.verify_export(self.export["root"])

    def test_rejected_count_mismatch_is_detected(self):
        meta_path = self.export["root"] / "export.meta.json"
        meta = json.loads(meta_path.read_text())
        meta["rejected_rows"] = 3
        meta_path.write_text(json.dumps(meta))
        provenance_path = self.export["root"] / "run-metadata.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["export"] = meta
        provenance_path.write_text(json.dumps(provenance))
        with self.assertRaisesRegex(DataError, "rejected"):
            data_module.verify_export(self.export["root"])

    def test_rejected_file_hash_mismatch_is_detected(self):
        rejected_path = self.export["root"] / "rejected.parquet"
        payload = bytearray(rejected_path.read_bytes())
        payload[len(payload) // 2] ^= 0xFF  # keep the parquet footer readable
        rejected_path.write_bytes(bytes(payload))
        with self.assertRaisesRegex(DataError, "rejected.parquet file hash"):
            data_module.verify_export(self.export["root"])

    def test_unreadable_rejected_file_is_a_handled_error(self):
        rejected_path = self.export["root"] / "rejected.parquet"
        rejected_path.write_bytes(b"not a parquet file")
        checksums_path = self.export["root"] / "checksums.json"
        checksums = json.loads(checksums_path.read_text())
        checksums["rejected"]["file_sha256"] = support.sha256_bytes(
            rejected_path.read_bytes())
        checksums_path.write_text(json.dumps(checksums))
        with self.assertRaisesRegex(DataError, "unreadable"):
            data_module.verify_export(self.export["root"])

    def test_provenance_disagreement_is_detected(self):
        provenance_path = self.export["root"] / "run-metadata.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["export"]["run_id"] = "different-run"
        provenance_path.write_text(json.dumps(provenance))
        with self.assertRaisesRegex(DataError, "run_id"):
            data_module.verify_export(self.export["root"])

    def test_manifest_provenance_disagreement_is_detected(self):
        provenance_path = self.export["root"] / "run-metadata.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["manifest"]["manifest_sha256"] = "0" * 64
        provenance_path.write_text(json.dumps(provenance))
        with self.assertRaisesRegex(DataError, "manifest metadata"):
            data_module.verify_export(self.export["root"])

    def test_iter_rows_enforces_order_and_projections(self):
        info = data_module.verify_export(self.export["root"], quick=True)
        logical = list(data_module.iter_rows(info, columns="logical"))
        self.assertEqual([row["source_row_index"] for row in logical], list(range(6)))
        self.assertNotIn("tikz_code", logical[0])
        text = list(data_module.iter_rows(info, columns="text"))
        self.assertIn("tikz_code", text[0])
        self.assertNotIn("png_image", text[0])
        full = list(data_module.iter_rows(info, columns="full", include_image=True))
        self.assertIn("png_image", full[0])
        with self.assertRaisesRegex(DataError, "Unknown column projection"):
            list(data_module.iter_rows(info, columns="mystery"))


class GroupingTests(unittest.TestCase):
    def rows(self, *entries):
        result = []
        for index, (tikz, image) in enumerate(entries):
            result.append({"id": f"row-{index}", "tikz_sha256": tikz,
                           "image_sha256": image, "source_row_index": index})
        return result

    def test_rows_sharing_tikz_or_image_are_grouped(self):
        rows = self.rows(("t1", "i1"), ("t1", "i2"), ("t2", "i2"), ("t3", "i3"))
        groups, count = data_module.duplicate_groups(rows)
        self.assertEqual(count, 2)
        self.assertEqual(groups["row-0"], groups["row-1"])
        self.assertEqual(groups["row-1"], groups["row-2"])
        self.assertNotEqual(groups["row-0"], groups["row-3"])

    def test_group_ids_are_deterministic_and_order_independent(self):
        rows = self.rows(("t1", "i1"), ("t1", "i2"))
        first, _ = data_module.duplicate_groups(rows)
        second, _ = data_module.duplicate_groups(list(reversed(rows)))
        self.assertEqual(first, second)

    def test_missing_hash_is_refused(self):
        with self.assertRaises(DataError):
            data_module.duplicate_groups([{"id": "x", "image_sha256": "i"}])


class SplitTests(unittest.TestCase):
    def make_groups(self, count=20):
        group_by_row = {}
        for index in range(count):
            group_by_row[f"row-{index}"] = f"group-{index:03d}"
        return group_by_row

    def test_assignment_is_deterministic(self):
        groups = self.make_groups()
        first = data_module.assign_splits(
            groups, row_count=20, seed=5, validation_fraction=0.25,
            test_fraction=0.25)
        second = data_module.assign_splits(
            groups, row_count=20, seed=5, validation_fraction=0.25,
            test_fraction=0.25)
        self.assertEqual(first, second)

    def test_seed_changes_assignment(self):
        groups = self.make_groups(50)
        first = data_module.assign_splits(
            groups, row_count=50, seed=1, validation_fraction=0.2,
            test_fraction=0.2)
        second = data_module.assign_splits(
            groups, row_count=50, seed=2, validation_fraction=0.2,
            test_fraction=0.2)
        self.assertNotEqual(first, second)

    def test_groups_never_cross_splits(self):
        group_by_row = {}
        for index in range(30):
            group = f"group-{index // 3:03d}"
            group_by_row[f"row-{index}"] = group
        assignment = data_module.assign_splits(
            group_by_row, row_count=30, seed=3, validation_fraction=0.2,
            test_fraction=0.2)
        by_group = {}
        for row_id, split in assignment.items():
            by_group.setdefault(group_by_row[row_id], set()).add(split)
        for splits in by_group.values():
            self.assertEqual(len(splits), 1)

    def test_counts_cover_every_row(self):
        groups = self.make_groups(40)
        assignment = data_module.assign_splits(
            groups, row_count=40, seed=0, validation_fraction=0.1,
            test_fraction=0.1)
        self.assertEqual(len(assignment), 40)
        counts = {split: list(assignment.values()).count(split)
                  for split in ("train", "validation", "test")}
        self.assertGreater(counts["validation"], 0)
        self.assertGreater(counts["test"], 0)
        self.assertGreater(counts["train"], 0)

    def test_fraction_rounding_to_zero_is_refused(self):
        groups = self.make_groups(5)
        with self.assertRaisesRegex(DataError, "rounds to 0"):
            data_module.assign_splits(groups, row_count=5, seed=0,
                                      validation_fraction=0.01,
                                      test_fraction=0.0)

    def test_empty_train_split_is_refused(self):
        groups = self.make_groups(2)
        with self.assertRaisesRegex(DataError, "empty train split"):
            data_module.assign_splits(groups, row_count=2, seed=0,
                                      validation_fraction=1.0 - 1e-9,
                                      test_fraction=0.0)


@support.requires_pyarrow
class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, **kwargs):
        return support.prepared_export_and_dir(self.root, **kwargs)

    def test_prepare_writes_self_verifying_artifacts(self):
        export, prepared, report = self.prepare()
        manifest = data_module.load_split_manifest(prepared)
        self.assertEqual(manifest["data_identity"]["sha256"],
                         report["data_identity"]["sha256"])
        self.assertEqual(manifest["counts"]["train"] + manifest["counts"]["validation"]
                         + manifest["counts"]["test"], 6)
        self.assertEqual(len(manifest["rows"]), 6)
        ids = {row["id"] for row in manifest["rows"]}
        self.assertEqual(ids, {row["id"] for row in export["records"]})
        self.assertTrue((prepared / "dataset-report.json").is_file())
        self.assertTrue((prepared / "models" / "minicpm5" / "token-report.json").is_file())
        self.assertTrue((prepared / "models" / "minicpm5" / "model.json").is_file())

    def test_prepare_is_deterministic_across_runs(self):
        export, prepared_one, report_one = self.prepare()
        second_dir = self.root / "prepared-two"
        config = support.make_config(data={"splits": {
            "validation_fraction": 0.2, "test_fraction": 0.2}})
        report_two = data_module.prepare_dataset(
            [config], export_dir=export["root"], prepared_dir=second_dir,
            tokenizer_factory=support.fake_tokenizer_factory())
        self.assertEqual(report_one["data_identity"]["sha256"],
                         report_two["data_identity"]["sha256"])
        self.assertEqual(report_one["data_identity"]["split_manifest_sha256"],
                         report_two["data_identity"]["split_manifest_sha256"])

    def test_split_manifest_hash_is_time_independent(self):
        """Regression: created_at is provenance and must not affect the hash."""
        from unittest import mock
        export, prepared_one, report_one = self.prepare()
        second_dir = self.root / "prepared-time"
        config = support.make_config(data={"splits": {
            "validation_fraction": 0.2, "test_fraction": 0.2}})
        with mock.patch.object(data_module, "utc_now_iso",
                               return_value="2001-01-01T00:00:00Z"):
            report_two = data_module.prepare_dataset(
                [config], export_dir=export["root"], prepared_dir=second_dir,
                tokenizer_factory=support.fake_tokenizer_factory())
        self.assertEqual(report_one["data_identity"]["split_manifest_sha256"],
                         report_two["data_identity"]["split_manifest_sha256"])
        self.assertEqual(report_one["data_identity"]["sha256"],
                         report_two["data_identity"]["sha256"])
        manifest = data_module.load_split_manifest(prepared_one)
        self.assertIn("created_at", manifest)
        self.assertIn("created_at",
                      json.loads((prepared_one / "split-manifest.json").read_text()))

    def test_duplicate_groups_never_cross_splits_end_to_end(self):
        rows = support.default_rows(8)
        rows[1]["tikz_code"] = rows[0]["tikz_code"]  # identical content
        for seed in range(8):
            with tempfile.TemporaryDirectory() as directory:
                export = support.make_export(Path(directory) / "export", rows=rows)
                config = support.make_config(seed=seed, data={"splits": {
                    "validation_fraction": 0.35, "test_fraction": 0.35}})
                prepared_dir = Path(directory) / "prepared"
                data_module.prepare_dataset(
                    [config], export_dir=export["root"], prepared_dir=prepared_dir,
                    tokenizer_factory=support.fake_tokenizer_factory())
                manifest = data_module.load_split_manifest(prepared_dir)
                by_id = {row["id"]: row for row in manifest["rows"]}
                first = export["records"][0]["id"]
                second = export["records"][1]["id"]
                self.assertEqual(by_id[first]["split"], by_id[second]["split"])
                self.assertEqual(by_id[first]["group_id"], by_id[second]["group_id"])

    def test_overlength_rows_are_quarantined_with_reason(self):
        export, prepared, report = self.prepare(
            rows=support.default_rows(6, pad_tikz=40), max_seq_len=128,
            max_quarantined_fraction=1.0,
            tokenizer_factory=support.fake_tokenizer_factory())
        quarantine_path = prepared / "models" / "minicpm5" / "quarantine.jsonl"
        self.assertTrue(quarantine_path.is_file())
        entries = [json.loads(line) for line in quarantine_path.read_text().splitlines()]
        self.assertEqual(len(entries), 6)
        for entry in entries:
            self.assertEqual(entry["reason"], "over_length")
            self.assertGreater(entry["total_tokens"], 128)
            self.assertGreater(entry["supervised_tokens"], 0)
        token_report = json.loads(
            (prepared / "models" / "minicpm5" / "token-report.json").read_text())
        self.assertEqual(token_report["quarantine"]["over_length"], 6)
        self.assertTrue(token_report["quarantine"]["sha256"])
        quarantine_sha = support.sha256_bytes(quarantine_path.read_bytes())
        self.assertEqual(token_report["quarantine"]["sha256"], quarantine_sha)
        report = json.loads((prepared / "dataset-report.json").read_text())
        self.assertEqual(report["models"]["minicpm5"]["quarantine_sha256"],
                         quarantine_sha)
        for split in ("train", "validation", "test"):
            counts = report["models"]["minicpm5"]["by_split"][split]
            self.assertEqual(counts["eligible"], 0)
            self.assertEqual(counts["quarantined"],
                             token_report["splits"][split]["over_max_seq_len"])

    def test_quarantine_fraction_above_limit_is_refused(self):
        with self.assertRaisesRegex(DataError, "quarantined"):
            self.prepare(rows=support.default_rows(6, pad_tikz=40),
                         max_seq_len=128, max_quarantined_fraction=0.0)

    def test_missing_model_artifact_is_refused(self):
        export, prepared, report = self.prepare()
        (prepared / "models" / "minicpm5" / "token-report.json").unlink()
        with self.assertRaises(DataError):
            data_module.verify_prepared(prepared, adapter_slug="minicpm5")

    def test_tampered_split_manifest_is_refused(self):
        export, prepared, report = self.prepare()
        manifest_path = prepared / "split-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["counts"]["train"] += 1
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(DataError, "hash mismatch"):
            data_module.load_split_manifest(prepared)

    def test_expected_identity_mismatch_is_refused(self):
        export, prepared, report = self.prepare()
        with self.assertRaisesRegex(DataError, "identity mismatch"):
            data_module.verify_prepared(prepared, expected_data_identity_sha256="0" * 64)

    def test_artifacts_match_schemas(self):
        export, prepared, report = self.prepare()
        schema.validate_artifact(
            json.loads((prepared / "split-manifest.json").read_text()),
            schema.load_schema("split-manifest.schema.json"), "split manifest")
        schema.validate_artifact(
            json.loads((prepared / "dataset-report.json").read_text()),
            schema.load_schema("dataset-report.schema.json"), "dataset report")


@support.requires_pyarrow
class EligibilityTests(unittest.TestCase):
    """Quarantined rows must be excluded everywhere, and the artifact verified."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.export, self.prepared, self.report,
         self.eligibility) = support.mixed_quarantine_export_and_dir(self.root)
        self.manifest = data_module.load_split_manifest(self.prepared)

    def tearDown(self):
        self.temporary.cleanup()

    def test_eligibility_counts_and_ids(self):
        self.assertEqual(self.eligibility["counts"]["total"], 8)
        self.assertEqual(self.eligibility["counts"]["quarantined"], 3)
        self.assertEqual(self.eligibility["counts"]["eligible"], 5)
        for split, counts in self.eligibility["by_split"].items():
            self.assertEqual(counts["eligible"] + counts["quarantined"],
                             self.manifest["counts"][split])
        quarantined = self.eligibility["quarantined"]
        self.assertEqual(len(quarantined), 3)
        expected = {self.export["records"][index]["id"] for index in (0, 1, 2)}
        self.assertEqual(set(quarantined), expected)

    def test_select_split_ids_excludes_quarantined(self):
        quarantined = self.eligibility["quarantined"]
        for split in ("train", "validation", "test"):
            selected = data_module.select_split_ids(
                self.manifest, split, exclude=quarantined)
            self.assertFalse(set(selected) & set(quarantined), split)

    def reanchor(self, *, token_report=True, fingerprint=False,
                 quarantine_sha256=None):
        """Rewrite dataset-report.json hashes after a deliberate tamper.

        The report is the anchor for the model/token/quarantine artifacts; a
        test that tampers one artifact can re-anchor the report to exercise the
        *content* checks instead of the hash check.
        """
        report_path = self.prepared / "dataset-report.json"
        report = json.loads(report_path.read_text())
        entry = report["models"]["minicpm5"]
        if token_report:
            entry["token_report_sha256"] = support.sha256_bytes(
                (self.prepared / "models" / "minicpm5" / "token-report.json").read_bytes())
        if fingerprint:
            entry["model_fingerprint_sha256"] = support.sha256_bytes(
                (self.prepared / "models" / "minicpm5" / "model.json").read_bytes())
        if quarantine_sha256 is not None:
            entry["quarantine_sha256"] = quarantine_sha256
        report_path.write_text(json.dumps(report))

    def write_token_report(self, report):
        report_path = self.prepared / "models" / "minicpm5" / "token-report.json"
        report_path.write_text(json.dumps(report))

    def read_token_report(self):
        report_path = self.prepared / "models" / "minicpm5" / "token-report.json"
        return json.loads(report_path.read_text())

    def test_tampered_quarantine_is_refused(self):
        path = (self.prepared / "models" / "minicpm5" / "quarantine.jsonl")
        path.write_text(path.read_text() + json.dumps({"id": "row-9"}) + "\n")
        with self.assertRaisesRegex(DataError, "does not match the token report"):
            data_module.load_eligibility(self.prepared, "minicpm5")

    def test_quarantine_count_mismatch_is_refused(self):
        path = (self.prepared / "models" / "minicpm5" / "quarantine.jsonl")
        lines = path.read_text().splitlines()
        text = "\n".join(lines[:-1]) + "\n"
        path.write_text(text)
        digest = support.sha256_bytes(text.encode())
        report = self.read_token_report()
        report["quarantine"]["sha256"] = digest
        self.write_token_report(report)
        self.reanchor(quarantine_sha256=digest)
        with self.assertRaisesRegex(DataError, "entries but the token report"):
            data_module.load_eligibility(self.prepared, "minicpm5")

    def test_quarantine_id_not_in_manifest_is_refused(self):
        path = (self.prepared / "models" / "minicpm5" / "quarantine.jsonl")
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        entries[0]["id"] = "not-in-manifest"
        text = "".join(json.dumps(item, sort_keys=True) + "\n" for item in entries)
        path.write_text(text)
        digest = support.sha256_bytes(text.encode())
        report = self.read_token_report()
        report["quarantine"]["sha256"] = digest
        self.write_token_report(report)
        self.reanchor(quarantine_sha256=digest)
        with self.assertRaisesRegex(DataError, "not in the split manifest"):
            data_module.load_eligibility(self.prepared, "minicpm5")

    def test_quarantine_split_disagreement_is_refused(self):
        path = (self.prepared / "models" / "minicpm5" / "quarantine.jsonl")
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        entries[0]["split"] = "validation" if entries[0]["split"] != "validation" else "train"
        text = "".join(json.dumps(item, sort_keys=True) + "\n" for item in entries)
        path.write_text(text)
        digest = support.sha256_bytes(text.encode())
        report = self.read_token_report()
        report["quarantine"]["sha256"] = digest
        self.write_token_report(report)
        self.reanchor(quarantine_sha256=digest)
        with self.assertRaisesRegex(DataError, "split manifest says"):
            data_module.load_eligibility(self.prepared, "minicpm5")

    def test_token_report_disagreement_is_refused(self):
        report = self.read_token_report()
        report["splits"]["train"]["examples"] += 1
        self.write_token_report(report)
        self.reanchor(token_report=True)
        with self.assertRaisesRegex(DataError, "eligible rows"):
            data_module.load_eligibility(self.prepared, "minicpm5")

    def test_token_report_hash_is_anchored_to_dataset_report(self):
        report = self.read_token_report()
        report["quarantine"]["total"] = 0
        self.write_token_report(report)
        with self.assertRaisesRegex(DataError, "does not match dataset-report.json"):
            data_module.load_eligibility(self.prepared, "minicpm5")

    def test_model_fingerprint_hash_is_anchored_to_dataset_report(self):
        fingerprint_path = self.prepared / "models" / "minicpm5" / "model.json"
        fingerprint = json.loads(fingerprint_path.read_text())
        fingerprint["vocab_size"] = 999
        fingerprint_path.write_text(json.dumps(fingerprint))
        with self.assertRaisesRegex(DataError, "Model fingerprint .* does not match"):
            data_module.load_eligibility(self.prepared, "minicpm5")

    def test_replaced_quarantine_with_consistent_token_report_is_still_refused(self):
        # Replace the quarantine artifact AND the token report consistently
        # (hash, total and per-split counts), but not the dataset report: the
        # dataset-report anchor must still refuse it.
        path = (self.prepared / "models" / "minicpm5" / "quarantine.jsonl")
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        entries = entries[:-1]
        text = "".join(json.dumps(item, sort_keys=True) + "\n" for item in entries)
        path.write_text(text)
        digest = support.sha256_bytes(text.encode())
        report = self.read_token_report()
        report["quarantine"]["sha256"] = digest
        report["quarantine"]["total"] = len(entries)
        for split in ("train", "validation", "test"):
            report["splits"][split]["over_max_seq_len"] = sum(
                1 for entry in entries if entry["split"] == split)
        self.write_token_report(report)
        self.reanchor(token_report=True)  # token report hash matches again
        with self.assertRaisesRegex(DataError, "dataset-report.json records"):
            data_module.load_eligibility(self.prepared, "minicpm5")

    def test_verify_prepared_anchors_fingerprint_and_returns_report_hash(self):
        prepared = data_module.verify_prepared(self.prepared,
                                               adapter_slug="minicpm5")
        expected = support.sha256_bytes(
            (self.prepared / "dataset-report.json").read_bytes())
        self.assertEqual(prepared["report_sha256"], expected)
        fingerprint_path = self.prepared / "models" / "minicpm5" / "model.json"
        fingerprint = json.loads(fingerprint_path.read_text())
        fingerprint["pad_token_id"] = 12345
        fingerprint_path.write_text(json.dumps(fingerprint))
        with self.assertRaisesRegex(DataError, "model fingerprint .* does not match"):
            data_module.verify_prepared(self.prepared, adapter_slug="minicpm5")

    def test_wrong_adapter_report_is_refused(self):
        with self.assertRaises(DataError):
            data_module.load_eligibility(self.prepared, "qwen3.5")

    def test_prepared_without_quarantine_has_clean_eligibility(self):
        with tempfile.TemporaryDirectory() as directory:
            export, prepared, report = support.prepared_export_and_dir(directory)
            eligibility = data_module.load_eligibility(prepared, "minicpm5")
            self.assertEqual(eligibility["counts"]["quarantined"], 0)
            self.assertEqual(eligibility["counts"]["eligible"],
                             eligibility["counts"]["total"])
            self.assertFalse(
                (prepared / "models" / "minicpm5" / "quarantine.jsonl").exists())


@support.requires_pyarrow
class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.export, self.prepared, self.report = support.prepared_export_and_dir(
            self.root, rows=support.default_rows(30))
        self.manifest = data_module.load_split_manifest(self.prepared)
        self.info = data_module.verify_export(self.export["root"], quick=True)

    def tearDown(self):
        self.temporary.cleanup()

    def test_select_split_ids_is_deterministic_and_sampled(self):
        split = "train" if self.manifest["counts"]["train"] > 5 else "validation"
        first = data_module.select_split_ids(self.manifest, split, limit=5, seed=1)
        second = data_module.select_split_ids(self.manifest, split, limit=5, seed=1)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        all_ids = data_module.select_split_ids(self.manifest, split)
        self.assertNotEqual(set(first), set(all_ids[:5]))

    def test_select_without_limit_returns_all(self):
        ids = data_module.select_split_ids(self.manifest, "train")
        self.assertEqual(len(ids), self.manifest["counts"]["train"])

    def test_load_rows_by_ids_returns_source_order(self):
        ids = data_module.select_split_ids(self.manifest, "train")
        rows = data_module.load_rows_by_ids(self.info, list(reversed(ids)))
        self.assertEqual([row["id"] for row in rows], ids)

    def test_load_rows_by_ids_missing_id_is_refused(self):
        with self.assertRaisesRegex(DataError, "missing"):
            data_module.load_rows_by_ids(self.info, ["does-not-exist"])


class AnchoringTests(unittest.TestCase):
    """Dependency-free tests for the dataset-report artifact anchoring chain."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.prepared = support.synthetic_prepared_dir(
            self.temporary.name, quarantined=("row-1",))
        self.slug = "minicpm5"

    def tearDown(self):
        self.temporary.cleanup()

    def model_dir(self):
        return self.prepared / "models" / self.slug

    def token_report_path(self):
        return self.model_dir() / "token-report.json"

    def read_token_report(self):
        return json.loads(self.token_report_path().read_text())

    def write_token_report(self, report):
        self.token_report_path().write_text(json.dumps(report))

    def read_dataset_report(self):
        return json.loads((self.prepared / "dataset-report.json").read_text())

    def write_dataset_report(self, report):
        (self.prepared / "dataset-report.json").write_text(json.dumps(report))

    def reanchor(self, *, token_report=True, fingerprint=False,
                 quarantine_sha256=None):
        report = self.read_dataset_report()
        entry = report["models"][self.slug]
        if token_report:
            entry["token_report_sha256"] = support.sha256_bytes(
                self.token_report_path().read_bytes())
        if fingerprint:
            entry["model_fingerprint_sha256"] = support.sha256_bytes(
                (self.model_dir() / "model.json").read_bytes())
        if quarantine_sha256 is not None:
            entry["quarantine_sha256"] = quarantine_sha256
        self.write_dataset_report(report)

    def test_valid_chain_loads(self):
        eligibility = data_module.load_eligibility(self.prepared, self.slug)
        self.assertEqual(eligibility["counts"]["quarantined"], 1)
        self.assertEqual(eligibility["counts"]["eligible"], 5)
        self.assertIn("row-1", eligibility["quarantined"])

    def test_token_report_hash_is_anchored(self):
        report = self.read_token_report()
        report["quarantine"]["total"] = 0
        self.write_token_report(report)
        with self.assertRaisesRegex(DataError, "does not match dataset-report.json"):
            data_module.load_eligibility(self.prepared, self.slug)

    def test_model_fingerprint_hash_is_anchored(self):
        fingerprint_path = self.model_dir() / "model.json"
        fingerprint = json.loads(fingerprint_path.read_text())
        fingerprint["vocab_size"] = 999
        fingerprint_path.write_text(json.dumps(fingerprint))
        with self.assertRaisesRegex(DataError, "Model fingerprint .* does not match"):
            data_module.load_eligibility(self.prepared, self.slug)

    def test_quarantine_hash_is_anchored_to_the_token_report(self):
        (self.model_dir() / "quarantine.jsonl").write_text(
            json.dumps({"id": "row-9", "reason": "over_length",
                        "split": "train"}) + "\n")
        with self.assertRaisesRegex(DataError, "does not match the token report"):
            data_module.load_eligibility(self.prepared, self.slug)

    def test_consistent_quarantine_and_token_report_replacement_is_refused(self):
        # Replace the artifact and the token report consistently, but leave the
        # dataset report untouched: the report anchor must still refuse it.
        entries = [json.loads(line) for line in
                   (self.model_dir() / "quarantine.jsonl").read_text().splitlines()]
        text = ""
        (self.model_dir() / "quarantine.jsonl").write_text(text)
        digest = support.sha256_bytes(text.encode())
        report = self.read_token_report()
        report["quarantine"] = {"total": 0, "over_length": 0, "sha256": digest}
        for split in ("train", "validation", "test"):
            report["splits"][split]["over_max_seq_len"] = 0
        self.write_token_report(report)
        self.reanchor(token_report=True)  # token report hash matches again
        with self.assertRaisesRegex(DataError, "dataset-report.json records"):
            data_module.load_eligibility(self.prepared, self.slug)
        self.assertEqual(len(entries), 1)

    def test_verify_prepared_returns_and_checks_the_report_hash(self):
        prepared = data_module.verify_prepared(self.prepared,
                                               adapter_slug=self.slug)
        expected = support.sha256_bytes(
            (self.prepared / "dataset-report.json").read_bytes())
        self.assertEqual(prepared["report_sha256"], expected)
        fingerprint_path = self.model_dir() / "model.json"
        fingerprint = json.loads(fingerprint_path.read_text())
        fingerprint["pad_token_id"] = 12345
        fingerprint_path.write_text(json.dumps(fingerprint))
        with self.assertRaisesRegex(DataError, "model fingerprint .* does not match"):
            data_module.verify_prepared(self.prepared, adapter_slug=self.slug)

    def test_split_manifest_verification_accepts_the_synthetic_layout(self):
        manifest = data_module.load_split_manifest(self.prepared)
        self.assertEqual(manifest["counts"]["train"], 2)
        self.assertEqual(manifest["data_identity"]["sha256"],
                         data_module.load_split_manifest(self.prepared)
                         ["data_identity"]["sha256"])


if __name__ == "__main__":
    unittest.main()
