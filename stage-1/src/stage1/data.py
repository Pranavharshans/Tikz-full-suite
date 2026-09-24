"""Deterministic dataset preparation from the audited cleaning export.

Contract (see ``cleaning/PRODUCTION.md`` section 9 and
``cleaning/build_dataset.py:export_dataset``):

- Input is ``WORK/export`` written by the cleaning pipeline: atomic Parquet
  shards ordered by ``source_row_index`` plus ``export.meta.json``,
  ``checksums.json``, ``run-metadata.json``, ``stats.json`` and companion
  Parquet files.
- ``dataset_logical_sha256`` is re-derived from the shards with the cleaning
  pipeline's exact algorithm (``logical_row_json`` below), never trusted from
  the metadata alone.
- Preparation never writes inside the export directory. Output is a separate
  ``prepared/`` directory that is safe to rebuild: every file is written
  atomically and the split manifest is self-verifying.

Splitting:

- Rows that share a ``tikz_sha256`` **or** an ``image_sha256`` form one
  duplicate group (union-find over both hashes), and a group is never split
  across train/validation/test.
- Groups are ordered deterministically, shuffled with the configured seed, and
  assigned whole groups until each split reaches its row target.
- Stable row ids from the cleaning export are preserved everywhere.

Quarantine:

- Tokenization is model-specific, so quarantine is model-specific too: a row
  whose rendered prompt+assistant exceeds ``max_seq_len`` for that tokenizer is
  written to ``models/<adapter>/quarantine.jsonl`` with an explicit reason and
  its exact token counts. Nothing is ever truncated silently.
- The token report records the quarantine artifact's SHA256, and every consumer
  loads it through :func:`load_eligibility`, which validates the hash, the
  entry count, the split relationship and the per-split counts. Quarantined
  rows are excluded from training, validation, evaluation, memorization
  sampling and gate row selection.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .errors import DataError
from .util import (canonical_digest, sha256_bytes, sha256_file, sha256_text,
                   utc_now_iso, write_json_atomic, write_text_atomic)

PREPARED_SCHEMA_VERSION = "stage1-prepared-v1"
SPLIT_MANIFEST_NAME = "split-manifest.json"
DATASET_REPORT_NAME = "dataset-report.json"
MODELS_DIRNAME = "models"
MODEL_FINGERPRINT_NAME = "model.json"
TOKEN_REPORT_NAME = "token-report.json"
TOKEN_REPORT_MARKDOWN = "token-report.md"
QUARANTINE_NAME = "quarantine.jsonl"

EXPORT_SCHEMA_VERSION = "export-v1"

LOGICAL_COLUMNS = ("id", "source_row_index", "instruction", "prompt_version",
                   "image_sha256", "tikz_sha256")
TEXT_COLUMNS = LOGICAL_COLUMNS + ("tikz_code",)
FULL_COLUMNS = TEXT_COLUMNS + ("file_id", "source_dataset", "source_revision",
                               "caption_model", "caption_model_revision",
                               "png_image")

SPLIT_NAMES = ("train", "validation", "test")


# ---------------------------------------------------------------------------
# Cleaning-pipeline checksum contract (must stay byte-for-byte compatible)
# ---------------------------------------------------------------------------


def logical_row_json(row: dict) -> str:
    """Exact copy of ``cleaning/build_dataset.py:logical_row``."""
    return json.dumps(dict(id=row["id"], source_row_index=row["source_row_index"],
                           image_sha256=row["image_sha256"], tikz_sha256=row["tikz_sha256"],
                           instruction=row["instruction"], prompt_version=row["prompt_version"]),
                      sort_keys=True, ensure_ascii=False)


def shard_logical_checksum(rows) -> str:
    """Exact copy of the cleaning pipeline's per-shard logical checksum."""
    return sha256_text("".join(logical_row_json(row) + "\n" for row in rows))


# ---------------------------------------------------------------------------
# Export verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportInfo:
    root: Path
    meta: dict
    checksums: dict
    provenance: dict
    shards: tuple
    rows: int
    complete_rows: int
    rejected_rows: int
    source_dataset: str
    source_revision: str
    caption_model: str
    caption_model_revision: str
    prompt_version: str
    dataset_logical_sha256: str
    checksums_sha256: str

    def to_jsonable(self) -> dict:
        return {
            "root": str(self.root),
            "dataset_logical_sha256": self.dataset_logical_sha256,
            "checksums_sha256": self.checksums_sha256,
            "rows": self.rows,
            "complete_rows": self.complete_rows,
            "rejected_rows": self.rejected_rows,
            "shards": len(self.shards),
            "source_dataset": self.source_dataset,
            "source_revision": self.source_revision,
            "caption_model": self.caption_model,
            "caption_model_revision": self.caption_model_revision,
            "prompt_version": self.prompt_version,
            "run_id": self.meta.get("run_id"),
        }


def _require_pyarrow():
    try:
        import pyarrow.parquet  # noqa: F401
        return pyarrow.parquet
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataError(
            "pyarrow is required to read the cleaning export; install it with "
            "'pip install pyarrow'") from exc


def _load_json(path: Path, what: str) -> dict:
    if not path.is_file():
        raise DataError(f"{what} is missing: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise DataError(f"{what} is not valid JSON: {path}: {exc}") from exc


def _shard_rows(path: Path, columns) -> list:
    parquet = _require_pyarrow()
    try:
        table = parquet.read_table(path, columns=list(columns))
    except Exception as exc:
        raise DataError(f"Failed to read shard {path}: {exc}") from exc
    return table.to_pylist()


def verify_export(export_dir, *, require_complete: bool = True,
                  quick: bool = False) -> ExportInfo:
    """Verify export metadata, checksums and (unless quick) row content."""
    root = Path(export_dir).resolve()
    if not root.is_dir():
        raise DataError(f"Export directory does not exist: {root}")
    meta = _load_json(root / "export.meta.json", "export.meta.json")
    checksums = _load_json(root / "checksums.json", "checksums.json")
    provenance = _load_json(root / "run-metadata.json", "run-metadata.json")

    if meta.get("schema_version") != EXPORT_SCHEMA_VERSION:
        raise DataError(
            f"Unsupported export schema {meta.get('schema_version')!r}; expected "
            f"{EXPORT_SCHEMA_VERSION!r}")
    if not isinstance(meta.get("rows"), int) or not isinstance(meta.get("shards"), int):
        raise DataError("export.meta.json is missing integer rows/shards fields")

    run_record = provenance.get("run") or {}
    identity = run_record.get("identity") or {}
    try:
        source = identity["dataset"]
        model = identity["model"]
        prompt = identity["prompt"]
        source_dataset = source["dataset_id"]
        source_revision = source["revision"]
        caption_model = model["model_id"]
        caption_model_revision = model["revision"]
        prompt_version = prompt["version"]
    except KeyError as exc:
        raise DataError(
            f"run-metadata.json is missing provenance field {exc}; refusing to "
            "prepare from an incomplete export") from exc

    embedded = provenance.get("export") or {}
    for key in ("run_id", "identity_sha256", "manifest_sha256", "rows",
                "rejected_rows", "shards", "dataset_logical_sha256",
                "complete_rows", "schema_version"):
        if embedded.get(key) != meta.get(key):
            raise DataError(
                f"export.meta.json disagrees with run-metadata.json on {key}: "
                f"{meta.get(key)!r} vs {embedded.get(key)!r}")
    # The cleaning run record identifies the generation run, but the manifest
    # checksum belongs to the separately embedded manifest metadata.  The
    # production exporter intentionally does not copy manifest_sha256 into
    # ``run`` (see cleaning.build_dataset.build_run_record).
    for key in ("run_id", "identity_sha256"):
        if run_record.get(key) != meta.get(key):
            raise DataError(
                f"export.meta.json disagrees with the run record on {key}: "
                f"{meta.get(key)!r} vs {run_record.get(key)!r}")
    manifest_record = provenance.get("manifest") or {}
    if manifest_record.get("manifest_sha256") != meta.get("manifest_sha256"):
        raise DataError(
            "export.meta.json disagrees with the manifest metadata on "
            f"manifest_sha256: {meta.get('manifest_sha256')!r} vs "
            f"{manifest_record.get('manifest_sha256')!r}")

    shard_records = checksums.get("shards")
    if not isinstance(shard_records, list) or not shard_records:
        raise DataError("checksums.json has no shard list")
    shard_root = root / "shards"
    on_disk = sorted(path.name for path in shard_root.glob("shard-*.parquet"))
    listed = sorted(record.get("name") for record in shard_records)
    if on_disk != listed:
        raise DataError(
            "Shard files on disk do not match checksums.json:\n"
            f"  on disk: {on_disk[:5]}{'...' if len(on_disk) > 5 else ''}\n"
            f"  listed:  {listed[:5]}{'...' if len(listed) > 5 else ''}")

    logical_hashes = []
    total_rows = 0
    previous_source_index = None
    row_ordinal = 0
    seen_ids = set()
    for record in shard_records:
        name = record.get("name")
        path = shard_root / name
        if not isinstance(record.get("rows"), int):
            raise DataError(f"checksums.json entry for {name} has no integer row count")
        if not quick:
            actual_file = sha256_file(path)
            if actual_file != record.get("file_sha256"):
                raise DataError(
                    f"Shard {name} file hash mismatch: expected "
                    f"{record.get('file_sha256')}, computed {actual_file}")
        columns = LOGICAL_COLUMNS if quick else FULL_COLUMNS
        rows = _shard_rows(path, columns)
        if len(rows) != record["rows"]:
            raise DataError(
                f"Shard {name} row count mismatch: metadata says {record['rows']}, "
                f"file has {len(rows)}")
        digest_parts = []
        for row in rows:
            for key in LOGICAL_COLUMNS:
                if row.get(key) is None:
                    raise DataError(f"Shard {name}: row is missing {key!r}")
            digest_parts.append(logical_row_json(row) + "\n")
            if not quick:
                _verify_row_content(row, name, row_ordinal)
                if row["id"] in seen_ids:
                    raise DataError(f"Shard {name}: duplicate row id {row['id']}")
                seen_ids.add(row["id"])
                source_index = row["source_row_index"]
                if not isinstance(source_index, int) or source_index < 0:
                    raise DataError(
                        f"Shard {name}: row {row['id']} has invalid "
                        f"source_row_index {source_index!r}")
                if (previous_source_index is not None
                        and source_index <= previous_source_index):
                    raise DataError(
                        f"Shard {name}: source_row_index {source_index} follows "
                        f"{previous_source_index}; completed rows are not in "
                        "strict source order")
                previous_source_index = source_index
                row_ordinal += 1
                for key, expected in (("source_dataset", source_dataset),
                                      ("source_revision", source_revision),
                                      ("caption_model", caption_model),
                                      ("caption_model_revision", caption_model_revision),
                                      ("prompt_version", prompt_version)):
                    if row.get(key) != expected:
                        raise DataError(
                            f"Shard {name}: row {row['id']} has {key}="
                            f"{row.get(key)!r}, expected {expected!r}")
        if rows:
            for key, actual in (("first_index", rows[0]["source_row_index"]),
                                ("last_index", rows[-1]["source_row_index"])):
                if key in record and record.get(key) != actual:
                    raise DataError(
                        f"Shard {name} {key} mismatch: metadata says "
                        f"{record.get(key)}, file has {actual}")
        shard_digest = sha256_text("".join(digest_parts))
        if shard_digest != record.get("logical_sha256"):
            raise DataError(
                f"Shard {name} logical checksum mismatch: expected "
                f"{record.get('logical_sha256')}, computed {shard_digest}")
        logical_hashes.append(shard_digest)
        total_rows += len(rows)

    dataset_logical = canonical_digest(logical_hashes)
    if dataset_logical != checksums.get("dataset_logical_sha256"):
        raise DataError(
            "dataset_logical_sha256 mismatch: checksums.json says "
            f"{checksums.get('dataset_logical_sha256')}, recomputed {dataset_logical}")
    if dataset_logical != meta.get("dataset_logical_sha256"):
        raise DataError(
            "dataset_logical_sha256 mismatch between checksums.json and export.meta.json")
    if total_rows != meta["rows"]:
        raise DataError(
            f"Export row count mismatch: meta says {meta['rows']}, shards contain "
            f"{total_rows}")
    if require_complete and meta.get("complete_rows") != meta["rows"]:
        raise DataError(
            f"Export is incomplete: complete_rows={meta.get('complete_rows')} but "
            f"rows={meta['rows']}. Finish or repair the cleaning export first.")

    rejected_rows = meta.get("rejected_rows")
    rejected_path = root / "rejected.parquet"
    if rejected_path.is_file():
        recorded = (checksums.get("rejected") or {}).get("file_sha256")
        if not quick and recorded:
            actual_hash = sha256_file(rejected_path)
            if actual_hash != recorded:
                raise DataError(
                    "rejected.parquet file hash mismatch: checksums.json says "
                    f"{recorded}, computed {actual_hash}")
        parquet = _require_pyarrow()
        try:
            actual_rejected = parquet.read_metadata(rejected_path).num_rows
        except Exception as exc:
            raise DataError(
                f"rejected.parquet is unreadable: {type(exc).__name__}: {exc}") from exc
        if actual_rejected != rejected_rows:
            raise DataError(
                f"rejected.parquet has {actual_rejected} rows but export.meta.json "
                f"says {rejected_rows}")
    elif rejected_rows:
        raise DataError(
            f"export.meta.json reports {rejected_rows} rejected rows but "
            "rejected.parquet is missing")

    return ExportInfo(
        root=root, meta=meta, checksums=checksums, provenance=provenance,
        shards=tuple(shard_records), rows=total_rows,
        complete_rows=meta.get("complete_rows", total_rows),
        rejected_rows=rejected_rows or 0, source_dataset=source_dataset,
        source_revision=source_revision, caption_model=caption_model,
        caption_model_revision=caption_model_revision,
        prompt_version=prompt_version, dataset_logical_sha256=dataset_logical,
        checksums_sha256=sha256_file(root / "checksums.json"))


def _verify_row_content(row: dict, shard_name: str, expected_index: int) -> None:
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise DataError(f"Shard {shard_name}: row {expected_index} has no id")
    if not isinstance(row.get("tikz_code"), str) or not row["tikz_code"].strip():
        raise DataError(f"Shard {shard_name}: row {row['id']} has empty TikZ")
    if not isinstance(row.get("instruction"), str) or not row["instruction"].strip():
        raise DataError(f"Shard {shard_name}: row {row['id']} has empty instruction")
    tikz = row["tikz_code"].encode("utf-8")
    if sha256_bytes(tikz) != row["tikz_sha256"]:
        raise DataError(
            f"Shard {shard_name}: row {row['id']} TikZ content does not match "
            "tikz_sha256")
    image = row.get("png_image")
    if not isinstance(image, (bytes, bytearray)) or not image:
        raise DataError(f"Shard {shard_name}: row {row['id']} has no image bytes")
    if sha256_bytes(bytes(image)) != row["image_sha256"]:
        raise DataError(
            f"Shard {shard_name}: row {row['id']} image content does not match "
            "image_sha256")


def iter_rows(export_info: ExportInfo, *, columns: str = "text",
              include_image: bool = False):
    """Stream verified rows in global order.

    ``columns`` selects a projection: ``"logical"`` (hashes and metadata),
    ``"text"`` (plus ``tikz_code``) or ``"full"`` (plus provenance fields and,
    when ``include_image``, the PNG bytes). Ordering and id uniqueness are
    always enforced; per-row content hashing is only done by
    :func:`verify_export`.
    """
    if columns == "logical":
        projection = LOGICAL_COLUMNS
    elif columns == "text":
        projection = TEXT_COLUMNS
    elif columns == "full":
        projection = FULL_COLUMNS
    else:
        raise DataError(f"Unknown column projection {columns!r}")
    parquet = _require_pyarrow()
    expected_index = 0
    seen = set()
    for record in export_info.shards:
        path = export_info.root / "shards" / record["name"]
        handle = parquet.ParquetFile(path)
        for batch in handle.iter_batches(batch_size=64, columns=list(projection)):
            for row in batch.to_pylist():
                if row.get("id") in seen:
                    raise DataError(f"Duplicate row id while streaming: {row.get('id')}")
                seen.add(row.get("id"))
                if row.get("source_row_index") != expected_index:
                    raise DataError(
                        f"Row {row.get('id')} has source_row_index "
                        f"{row.get('source_row_index')}, expected {expected_index}")
                expected_index += 1
                if not include_image:
                    row.pop("png_image", None)
                yield row


# ---------------------------------------------------------------------------
# Duplicate grouping and deterministic splitting
# ---------------------------------------------------------------------------


def duplicate_groups(rows) -> tuple:
    """Union-find over rows sharing ``tikz_sha256`` or ``image_sha256``.

    Returns ``(group_by_row, group_count)`` where ``group_by_row`` maps row id
    to a deterministic group id derived from the sorted member ids.
    """
    parent = {}

    def find(item):
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return root

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    first_by_hash = {}
    for row in rows:
        row_id = row["id"]
        parent.setdefault(row_id, row_id)
        for key in ("tikz_sha256", "image_sha256"):
            value = row.get(key)
            if not value:
                raise DataError(f"Row {row_id} has no {key}")
            other = first_by_hash.setdefault((key, value), row_id)
            if other != row_id:
                union(row_id, other)

    members = {}
    for row_id in parent:
        members.setdefault(find(row_id), []).append(row_id)
    group_by_row = {}
    for root, member_ids in members.items():
        member_ids.sort()
        group_id = sha256_text("stage1-group-v1\x1f" + "\x1f".join(member_ids))[:32]
        for row_id in member_ids:
            group_by_row[row_id] = group_id
    return group_by_row, len(members)


def assign_splits(group_by_row: dict, *, row_count: int, seed: int,
                  validation_fraction: float, test_fraction: float) -> dict:
    """Assign whole duplicate groups to splits, deterministically."""
    group_members = {}
    for row_id, group_id in group_by_row.items():
        group_members.setdefault(group_id, []).append(row_id)
    group_ids = sorted(group_members)
    random.Random(seed).shuffle(group_ids)

    target_validation = round(validation_fraction * row_count)
    target_test = round(test_fraction * row_count)
    if validation_fraction > 0 and target_validation == 0:
        raise DataError(
            f"validation_fraction={validation_fraction} rounds to 0 rows for "
            f"{row_count} rows; increase the fraction or the dataset size")
    if test_fraction > 0 and target_test == 0:
        raise DataError(
            f"test_fraction={test_fraction} rounds to 0 rows for {row_count} rows; "
            "increase the fraction or the dataset size")

    assignment = {}
    index = 0
    for split, target in (("validation", target_validation), ("test", target_test)):
        assigned = 0
        while index < len(group_ids) and assigned < target:
            group_id = group_ids[index]
            index += 1
            for row_id in group_members[group_id]:
                assignment[row_id] = split
            assigned += len(group_members[group_id])
    for group_id in group_ids[index:]:
        for row_id in group_members[group_id]:
            assignment[row_id] = "train"

    counts = {split: 0 for split in SPLIT_NAMES}
    for split in assignment.values():
        counts[split] += 1
    if counts["train"] == 0:
        raise DataError(
            "The split assignment produced an empty train split; reduce the "
            "validation/test fractions")
    if len(assignment) != row_count:
        raise DataError(
            f"Split assignment covered {len(assignment)} of {row_count} rows")
    for group_id, member_ids in group_members.items():
        splits = {assignment[row_id] for row_id in member_ids}
        if len(splits) != 1:
            raise DataError(
                f"Duplicate group {group_id} crosses splits {sorted(splits)}; "
                "this is a defect in split assignment")
    return assignment


# ---------------------------------------------------------------------------
# Length statistics
# ---------------------------------------------------------------------------


def percentile(values, fraction: float):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


@dataclass
class LengthAccumulator:
    values: list = field(default_factory=list)
    total: int = 0

    def add(self, value: int) -> None:
        self.values.append(int(value))
        self.total += int(value)

    def summary(self) -> dict:
        if not self.values:
            return {"count": 0, "total": 0, "mean": None, "min": None,
                    "p50": None, "p90": None, "p95": None, "p99": None,
                    "max": None}
        return {
            "count": len(self.values),
            "total": self.total,
            "mean": round(self.total / len(self.values), 3),
            "min": min(self.values),
            "p50": percentile(self.values, 0.50),
            "p90": percentile(self.values, 0.90),
            "p95": percentile(self.values, 0.95),
            "p99": percentile(self.values, 0.99),
            "max": max(self.values),
        }


# ---------------------------------------------------------------------------
# Prepared-artifact IO
# ---------------------------------------------------------------------------


def split_manifest_path(prepared_dir) -> Path:
    return Path(prepared_dir) / SPLIT_MANIFEST_NAME


def dataset_report_path(prepared_dir) -> Path:
    return Path(prepared_dir) / DATASET_REPORT_NAME


def model_dir(prepared_dir, slug: str) -> Path:
    return Path(prepared_dir) / MODELS_DIRNAME / slug


def load_split_manifest(prepared_dir) -> dict:
    path = split_manifest_path(prepared_dir)
    data = _load_json(path, "split manifest")
    if data.get("schema_version") != PREPARED_SCHEMA_VERSION:
        raise DataError(
            f"Unsupported split manifest schema {data.get('schema_version')!r}")
    payload = {key: value for key, value in data.items()
               if key not in ("split_manifest_sha256", "data_identity",
                              "created_at")}
    recomputed = canonical_digest(payload)
    if recomputed != data.get("split_manifest_sha256"):
        raise DataError(
            "Split manifest hash mismatch: file says "
            f"{data.get('split_manifest_sha256')}, recomputed {recomputed}. "
            "The manifest was edited or corrupted.")
    identity = data.get("data_identity") or {}
    identity_payload = {key: value for key, value in identity.items() if key != "sha256"}
    recomputed_identity = canonical_digest(identity_payload)
    if recomputed_identity != identity.get("sha256"):
        raise DataError("Split manifest data identity hash mismatch")
    return data


def verify_prepared(prepared_dir, *, expected_data_identity_sha256=None,
                    adapter_slug=None) -> dict:
    """Verify the prepared directory; optionally pin identity and load a model."""
    prepared_dir = Path(prepared_dir).resolve()
    manifest = load_split_manifest(prepared_dir)
    identity_sha = manifest["data_identity"]["sha256"]
    if expected_data_identity_sha256 and identity_sha != expected_data_identity_sha256:
        raise DataError(
            "Prepared data identity mismatch: requested "
            f"{expected_data_identity_sha256}, found {identity_sha}")
    report = _load_json(dataset_report_path(prepared_dir), "dataset report")
    if report.get("status") != "pass":
        raise DataError(
            f"Dataset report status is {report.get('status')!r}; re-run "
            "scripts/prepare_dataset.py before training")
    report_identity = (report.get("data_identity") or {}).get("sha256")
    if report_identity != identity_sha:
        raise DataError(
            "dataset-report.json does not match the split manifest identity; "
            "re-run scripts/prepare_dataset.py")
    result = {"manifest": manifest, "report": report,
              "data_identity_sha256": identity_sha,
              "report_sha256": sha256_file(dataset_report_path(prepared_dir))}
    if adapter_slug:
        directory = model_dir(prepared_dir, adapter_slug)
        fingerprint_path = directory / MODEL_FINGERPRINT_NAME
        token_report_path = directory / TOKEN_REPORT_NAME
        fingerprint = _load_json(fingerprint_path,
                                 f"model fingerprint for {adapter_slug}")
        token_report = _load_json(token_report_path,
                                  f"token report for {adapter_slug}")
        recorded = (report.get("models") or {}).get(adapter_slug)
        if recorded is None:
            raise DataError(
                f"dataset-report.json has no entry for adapter {adapter_slug!r}; "
                "re-run scripts/prepare_dataset.py")
        for label, path, expected in (
                ("model fingerprint", fingerprint_path,
                 recorded.get("model_fingerprint_sha256")),
                ("token report", token_report_path,
                 recorded.get("token_report_sha256"))):
            actual = sha256_file(path)
            if actual != expected:
                raise DataError(
                    f"{label} for {adapter_slug} does not match "
                    f"dataset-report.json: report says {expected}, file hashes "
                    f"to {actual}. The prepared artifacts were edited or "
                    "replaced; re-run preparation.")
        result["model"] = fingerprint
        result["token_report"] = token_report
    return result


def rows_by_split(manifest: dict) -> dict:
    splits = {name: [] for name in SPLIT_NAMES}
    for row in manifest["rows"]:
        split = row.get("split")
        if split not in splits:
            raise DataError(f"Split manifest has unknown split {split!r}")
        splits[split].append(row)
    return splits


def select_split_ids(manifest: dict, split: str, *, limit=None,
                     seed: int = 0, exclude=()) -> list:
    """Deterministically select row ids from a split.

    ``limit=None`` returns every id in source order. A limited selection is a
    seeded random sample (not a prefix) so that adjacent duplicate content in
    the frozen source slice cannot make a small gate trivially easy; the
    selection is sorted back into source order for streaming. ``exclude``
    removes model-specific ineligible ids (quarantined rows) before sampling.
    """
    excluded = set(exclude)
    ids = [row["id"] for row in manifest["rows"]
           if row["split"] == split and row["id"] not in excluded]
    if limit is None or limit >= len(ids):
        return ids
    if limit < 1:
        raise DataError("limit must be positive")
    chosen = random.Random(seed).sample(ids, limit)
    return sorted(chosen)


def load_rows_by_ids(export_info: ExportInfo, ids, *,
                     include_image: bool = False) -> list:
    """Stream the export once and return the requested rows in source order."""
    wanted = set(ids)
    if not wanted:
        return []
    rows = []
    for row in iter_rows(export_info, columns="text",
                         include_image=include_image):
        if row["id"] in wanted:
            rows.append(row)
            if len(rows) == len(wanted):
                break
    missing = wanted - {row["id"] for row in rows}
    if missing:
        raise DataError(
            f"{len(missing)} requested row id(s) are missing from the export; "
            f"first missing: {sorted(missing)[0]}")
    return rows


def load_eligibility(prepared_dir, adapter_slug: str) -> dict:
    """Load and validate the model-specific quarantine artifact.

    Anchoring chain: ``dataset-report.json`` records the hashes of the model
    fingerprint, the token report and the quarantine artifact; the token report
    records the quarantine hash and per-split counts. Every link is verified
    here, so replacing an artifact (even together with the token report) is
    refused unless the dataset report is replaced too - and the report itself
    is anchored to the run identity.

    Returns the quarantined id set plus per-split eligible/quarantined counts.
    Every inconsistency is a hard error: a hash mismatch against the report or
    the token report, an entry missing from the split manifest, a split
    disagreement, or per-split counts that disagree with the token report.
    """
    prepared_dir = Path(prepared_dir).resolve()
    directory = model_dir(prepared_dir, adapter_slug)
    dataset_report = _load_json(dataset_report_path(prepared_dir), "dataset report")
    recorded = (dataset_report.get("models") or {}).get(adapter_slug)
    if recorded is None:
        raise DataError(
            f"dataset-report.json has no entry for adapter {adapter_slug!r}; "
            "re-run scripts/prepare_dataset.py")
    token_report_path = directory / TOKEN_REPORT_NAME
    token_report = _load_json(token_report_path, f"token report for {adapter_slug}")
    token_report_sha = sha256_file(token_report_path)
    if token_report_sha != recorded.get("token_report_sha256"):
        raise DataError(
            f"Token report for {adapter_slug} does not match "
            f"dataset-report.json: report says "
            f"{recorded.get('token_report_sha256')}, file hashes to "
            f"{token_report_sha}")
    fingerprint_path = directory / MODEL_FINGERPRINT_NAME
    fingerprint_sha = sha256_file(fingerprint_path)
    if fingerprint_sha != recorded.get("model_fingerprint_sha256"):
        raise DataError(
            f"Model fingerprint for {adapter_slug} does not match "
            f"dataset-report.json: report says "
            f"{recorded.get('model_fingerprint_sha256')}, file hashes to "
            f"{fingerprint_sha}")
    if token_report.get("adapter") != adapter_slug:
        raise DataError(
            f"Token report in {directory} is for adapter "
            f"{token_report.get('adapter')!r}, not {adapter_slug!r}")
    quarantine_info = token_report.get("quarantine") or {}
    expected_total = quarantine_info.get("total")
    expected_sha = quarantine_info.get("sha256")
    if not isinstance(expected_total, int) or not expected_sha:
        raise DataError(
            f"Token report for {adapter_slug} lacks quarantine counts/hash; "
            "re-run scripts/prepare_dataset.py")
    if expected_sha != recorded.get("quarantine_sha256"):
        raise DataError(
            f"Token report for {adapter_slug} records quarantine hash "
            f"{expected_sha} but dataset-report.json records "
            f"{recorded.get('quarantine_sha256')}; re-run preparation")

    quarantine_path = directory / QUARANTINE_NAME
    text = quarantine_path.read_text() if quarantine_path.is_file() else ""
    actual_sha = sha256_text(text)
    if actual_sha != expected_sha:
        raise DataError(
            f"Quarantine artifact for {adapter_slug} does not match the token "
            f"report: token report says {expected_sha}, file hashes to "
            f"{actual_sha}. The artifact was edited or replaced; re-run "
            "preparation.")
    entries = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise DataError(
                f"{quarantine_path}:{number}: invalid JSON: {exc}") from exc
    if len(entries) != expected_total:
        raise DataError(
            f"Quarantine artifact for {adapter_slug} has {len(entries)} "
            f"entries but the token report says {expected_total}")

    manifest = load_split_manifest(prepared_dir)
    split_by_id = {row["id"]: row["split"] for row in manifest["rows"]}
    quarantined = set()
    quarantined_by_split = {split: 0 for split in SPLIT_NAMES}
    for entry in entries:
        row_id, reason, split = entry.get("id"), entry.get("reason"), entry.get("split")
        if not row_id or not reason:
            raise DataError(
                f"Quarantine entry in {quarantine_path} is missing id/reason: {entry!r}")
        if row_id in quarantined:
            raise DataError(f"Quarantine artifact lists {row_id} more than once")
        if row_id not in split_by_id:
            raise DataError(
                f"Quarantined id {row_id} is not in the split manifest; the "
                "quarantine artifact belongs to another prepared dataset")
        if split_by_id[row_id] != split:
            raise DataError(
                f"Quarantined id {row_id} is recorded under split {split!r} but "
                f"the split manifest says {split_by_id[row_id]!r}")
        quarantined.add(row_id)
        quarantined_by_split[split] += 1

    by_split = {}
    for split in SPLIT_NAMES:
        report_split = (token_report.get("splits") or {}).get(split) or {}
        report_quarantined = report_split.get("over_max_seq_len")
        report_eligible = report_split.get("examples")
        if report_quarantined is not None and report_quarantined != quarantined_by_split[split]:
            raise DataError(
                f"Token report for {adapter_slug} says {report_quarantined} "
                f"quarantined rows in {split!r}, the quarantine artifact has "
                f"{quarantined_by_split[split]}")
        eligible = sum(1 for row in manifest["rows"]
                       if row["split"] == split and row["id"] not in quarantined)
        if report_eligible is not None and report_eligible != eligible:
            raise DataError(
                f"Token report for {adapter_slug} says {report_eligible} eligible "
                f"rows in {split!r}, the split manifest minus quarantine has "
                f"{eligible}")
        by_split[split] = {"eligible": eligible,
                           "quarantined": quarantined_by_split[split]}
    return {
        "adapter": adapter_slug,
        "quarantine_sha256": expected_sha,
        "quarantined": frozenset(quarantined),
        "counts": {
            "total": len(manifest["rows"]),
            "eligible": len(manifest["rows"]) - len(quarantined),
            "quarantined": len(quarantined),
        },
        "by_split": by_split,
        "max_seq_len": token_report.get("max_seq_len"),
    }


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def prepare_dataset(configs, *, export_dir, prepared_dir, local_files_only=False,
                    cache_dir=None, progress=None, tokenizer_factory=None) -> dict:
    """Build splits and per-model tokenization reports from a verified export.

    ``tokenizer_factory`` is an injection seam for tests; production callers
    leave it None so the pinned tokenizer loader is used.
    """
    from . import adapters, formatting

    configs = list(configs)
    if not configs:
        raise DataError("prepare_dataset requires at least one model config")
    slugs = [config.model.adapter for config in configs]
    if len(set(slugs)) != len(slugs):
        raise DataError(f"Duplicate adapter in prepare configs: {slugs}")
    base = configs[0]
    for config in configs[1:]:
        if (config.seed != base.seed
                or config.data.splits != base.data.splits
                or config.data.require_complete_export != base.data.require_complete_export):
            raise DataError(
                "All model configs in one prepare invocation must share seed, "
                "splits and require_complete_export")

    def say(message):
        if progress:
            progress(message)

    prepared_dir = Path(prepared_dir).resolve()
    say(f"verifying export at {export_dir}")
    export_info = verify_export(
        export_dir, require_complete=base.data.require_complete_export, quick=False)

    say("building the content index")
    index = []
    for row in iter_rows(export_info, columns="text"):
        index.append({
            "id": row["id"],
            "source_row_index": row["source_row_index"],
            "tikz_sha256": row["tikz_sha256"],
            "image_sha256": row["image_sha256"],
            "instruction_chars": len(row["instruction"]),
            "tikz_chars": len(row["tikz_code"]),
        })

    say("grouping duplicate TikZ/image content")
    group_by_row, group_count = duplicate_groups(index)
    assignment = assign_splits(
        group_by_row, row_count=len(index), seed=base.seed,
        validation_fraction=base.data.splits.validation_fraction,
        test_fraction=base.data.splits.test_fraction)
    counts = {split: 0 for split in SPLIT_NAMES}
    for split in assignment.values():
        counts[split] += 1
    say(f"splits: {counts} across {group_count} duplicate groups")

    rows = [{
        "id": row["id"],
        "source_row_index": row["source_row_index"],
        "group_id": group_by_row[row["id"]],
        "split": assignment[row["id"]],
    } for row in index]
    rows.sort(key=lambda item: item["source_row_index"])

    manifest_payload = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "tool_version": base.tool_version,
        "seed": base.seed,
        "splits": {
            "validation_fraction": base.data.splits.validation_fraction,
            "test_fraction": base.data.splits.test_fraction,
        },
        "counts": counts,
        "group_count": group_count,
        "rows": rows,
    }
    # The hash covers only the deterministic payload: created_at is
    # provenance and must not change the split-manifest hash across runs.
    split_manifest_sha256 = canonical_digest(manifest_payload)
    data_identity_payload = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "dataset_logical_sha256": export_info.dataset_logical_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "seed": base.seed,
        "splits": manifest_payload["splits"],
        "row_count": len(rows),
        "complete_rows": export_info.complete_rows,
        "source_dataset": export_info.source_dataset,
        "source_revision": export_info.source_revision,
    }
    data_identity = dict(data_identity_payload)
    data_identity["sha256"] = canonical_digest(data_identity_payload)
    manifest = dict(manifest_payload)
    manifest["created_at"] = utc_now_iso()
    manifest["split_manifest_sha256"] = split_manifest_sha256
    manifest["data_identity"] = data_identity
    prepared_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(split_manifest_path(prepared_dir), manifest)

    row_index = {row["id"]: row for row in rows}
    model_summaries = {}
    load_tokenizer = tokenizer_factory or adapters.load_tokenizer
    for config in configs:
        slug = config.model.adapter
        say(f"tokenizing with the {slug} tokenizer")
        tokenizer = load_tokenizer(
            config, local_files_only=local_files_only, cache_dir=cache_dir)
        template = formatting.resolve_chat_template(tokenizer, config.tokenizer)
        fingerprint = adapters.tokenizer_fingerprint(tokenizer, template, config)
        report, quarantine = _tokenize_rows(
            config, tokenizer, template, export_info, row_index, say)
        directory = model_dir(prepared_dir, slug)
        directory.mkdir(parents=True, exist_ok=True)
        quarantine_text = "".join(
            json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n"
            for item in quarantine)
        report["quarantine"]["sha256"] = sha256_text(quarantine_text)
        write_json_atomic(directory / MODEL_FINGERPRINT_NAME, fingerprint)
        if quarantine_text:
            write_text_atomic(directory / QUARANTINE_NAME, quarantine_text)
        elif (directory / QUARANTINE_NAME).exists():
            (directory / QUARANTINE_NAME).unlink()
        write_json_atomic(directory / TOKEN_REPORT_NAME, report)
        write_text_atomic(directory / TOKEN_REPORT_MARKDOWN,
                          _render_token_report(config, report))
        quarantined = len(quarantine)
        fraction = quarantined / len(rows) if rows else 0.0
        if fraction > config.data.max_quarantined_fraction:
            reasons = {}
            for item in quarantine:
                reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
            raise DataError(
                f"{quarantined} of {len(rows)} rows ({fraction:.4%}) were "
                f"quarantined for {slug}, above max_quarantined_fraction="
                f"{config.data.max_quarantined_fraction:.4%}. Reasons: {reasons}. "
                "Raise data.max_seq_len or investigate the rows before "
                "proceeding; nothing was truncated.")
        model_summaries[slug] = {
            "token_report_sha256": sha256_file(directory / TOKEN_REPORT_NAME),
            "model_fingerprint_sha256": sha256_file(directory / MODEL_FINGERPRINT_NAME),
            "quarantined": quarantined,
            "quarantine_fraction": round(fraction, 6),
            "quarantine_sha256": report["quarantine"]["sha256"],
            "max_seq_len": config.data.max_seq_len,
            "by_split": {
                split: {
                    "eligible": report["splits"][split]["examples"],
                    "quarantined": report["splits"][split]["over_max_seq_len"],
                } for split in SPLIT_NAMES
            },
        }
        say(f"{slug}: {quarantined} quarantined, report at {directory}")

    report = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "status": "pass",
        "created_at": utc_now_iso(),
        "tool_version": base.tool_version,
        "export": export_info.to_jsonable(),
        "data_identity": data_identity,
        "models": model_summaries,
    }
    write_json_atomic(dataset_report_path(prepared_dir), report)
    return report


def _tokenize_rows(config, tokenizer, template, export_info, row_index, say):
    from .formatting import format_example

    max_seq_len = config.data.max_seq_len
    totals = {split: LengthAccumulator() for split in SPLIT_NAMES}
    prompts = {split: LengthAccumulator() for split in SPLIT_NAMES}
    supervised = {split: LengthAccumulator() for split in SPLIT_NAMES}
    over_limit = {split: 0 for split in SPLIT_NAMES}
    quarantine = []
    processed = 0
    for row in iter_rows(export_info, columns="text"):
        entry = row_index[row["id"]]
        split = entry["split"]
        processed += 1
        if processed % 20000 == 0:
            say(f"  {processed} rows tokenized")
        example = format_example(
            tokenizer, row_id=row["id"], instruction=row["instruction"],
            tikz=row["tikz_code"], template=template,
            kwargs=config.tokenizer.chat_template_kwargs)
        if example.total_tokens > max_seq_len:
            over_limit[split] += 1
            quarantine.append({
                "id": row["id"],
                "source_row_index": row["source_row_index"],
                "split": split,
                "reason": "over_length",
                "max_seq_len": max_seq_len,
                "total_tokens": example.total_tokens,
                "prompt_tokens": example.prompt_tokens,
                "supervised_tokens": example.supervised_tokens,
            })
            continue
        totals[split].add(example.total_tokens)
        prompts[split].add(example.prompt_tokens)
        supervised[split].add(example.supervised_tokens)

    report = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "model_id": config.model.id,
        "model_revision": config.model.revision,
        "adapter": config.model.adapter,
        "max_seq_len": max_seq_len,
        "chat_template_sha256": template.sha256,
        "splits": {},
        "quarantine": {
            "total": len(quarantine),
            "over_length": sum(over_limit.values()),
        },
    }
    for split in SPLIT_NAMES:
        report["splits"][split] = {
            "examples": totals[split].summary()["count"],
            "total_tokens": totals[split].summary(),
            "prompt_tokens": prompts[split].summary(),
            "supervised_tokens": supervised[split].summary(),
            "supervised_tokens_sum": supervised[split].total,
            "over_max_seq_len": over_limit[split],
        }
    return report, quarantine


def _render_token_report(config, report: dict) -> str:
    lines = [
        f"# Token-length report: {config.model.id}",
        "",
        f"- adapter: `{config.model.adapter}`",
        f"- revision: `{config.model.revision}`",
        f"- max_seq_len: {report['max_seq_len']}",
        f"- chat template SHA256: `{report['chat_template_sha256'][:16]}...`",
        f"- quarantined rows: {report['quarantine']['total']} "
        f"(over length: {report['quarantine']['over_length']})",
        "",
        "| split | examples | supervised tokens | total p50 | total p90 | "
        "total p95 | total p99 | total max | over limit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLIT_NAMES:
        entry = report["splits"][split]
        totals = entry["total_tokens"]
        lines.append(
            f"| {split} | {entry['examples']} | {entry['supervised_tokens_sum']} | "
            f"{totals['p50']} | {totals['p90']} | {totals['p95']} | {totals['p99']} | "
            f"{totals['max']} | {entry['over_max_seq_len']} |")
    lines.append("")
    lines.append("Overlength rows are quarantined, never truncated. Supervised "
                 "tokens are the assistant/TikZ tokens that carry loss.")
    lines.append("")
    return "\n".join(lines)
