#!/usr/bin/env python3
"""Production dataset builder for the TikZ cleaning stage.

Builds the text-to-TikZ instruction dataset from a frozen slice of
``nllg/DaTikZ-V4`` captioned by ``nvidia/Qwen3.8-27B-NVFP4``.

Stages, each explicit and resumable::

    plan           print the resolved configuration and provisional run identity
    prepare        freeze exactly the first 100,000 source rows into a manifest
    run            identity-safe two-replica inference writing to a state ledger
    status         human-readable progress report
    validate       re-validate every accepted instruction
    audit          verify structural invariants (nonzero exit on violation)
    export         deterministic atomic Parquet shards for complete rows
    checkpoint     copy the ledger to a timestamped backup

``benchmark.py`` is preserved untouched; this module reuses only its audited
helpers.  See ``cleaning/PRODUCTION.md`` for the production runbook.
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import benchmark as bench  # noqa: E402  (audited helpers, imported from the same directory)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOOL_VERSION = "build-dataset-v1"
IDENTITY_VERSION = "run-id-v1"
MANIFEST_SCHEMA_VERSION = "manifest-v1"
LEDGER_SCHEMA_VERSION = "ledger-v1"
DATASET_SCHEMA_VERSION = "dataset-v1"
ROW_ID_VERSION = "rid-v1"
VALIDATION_POLICY_VERSION = "validation-v1"

DATASET_ID = "nllg/DaTikZ-V4"
DATASET_SPLIT = "train"
ROW_START = 0
ROW_LIMIT = 100_000

MODEL_ID = "nvidia/Qwen3.8-27B-NVFP4"

DEFAULT_PROMPT_VERSION = "caption-v1"

STATE_PENDING = "pending"
STATE_RUNNING = "running"
STATE_RETRYABLE = "retryable"
STATE_COMPLETE = "complete"
STATE_REJECTED = "rejected"
STATES = (STATE_PENDING, STATE_RUNNING, STATE_RETRYABLE, STATE_COMPLETE, STATE_REJECTED)

REVISION_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")

PROMPTS_DIR = HERE / "prompts"

# Measured on two RTX PRO 6000 Blackwell 96GB GPUs, 2026-09-23 (cleaning/results).
MEASURED_SAMPLES_PER_HOUR = 14_756.96


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """Raised when a configuration is invalid or inconsistent."""


class IdentityMismatch(RuntimeError):
    """Raised when a work directory belongs to a different run identity."""

    def __init__(self, differences):
        self.differences = list(differences)
        detail = "\n".join(f"  - {line}" for line in self.differences)
        super().__init__(
            "Work directory run identity does not match the requested configuration.\n"
            f"{detail}\nUse a fresh --work directory for a new run, or restore the "
            "matching configuration.")


class PromptError(RuntimeError):
    """Raised when a prompt version is missing or its text does not match the registry."""


# ---------------------------------------------------------------------------
# Hashing and atomic file helpers
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value) -> str:
    """Stable SHA256 over a JSON-compatible value (sorted keys, no whitespace games)."""
    return bench.digest(value)


def is_pinned_revision(value: str) -> bool:
    return bool(REVISION_RE.fullmatch(value or ""))


def is_sha256(value: str) -> bool:
    return bool(SHA256_RE.fullmatch(value or ""))


def detect_git_commit() -> str:
    """Best-effort runner commit, recorded for provenance only (never hashed)."""
    try:
        result = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10)
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


# ---------------------------------------------------------------------------
# Prompt registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prompt:
    version: str
    sha256: str
    text: str


def normalize_prompt(raw: str) -> str:
    """Collapse all whitespace so the hash is insensitive to line wrapping."""
    return " ".join(raw.split())


def load_prompt(version: str = DEFAULT_PROMPT_VERSION, prompts_dir=None) -> Prompt:
    directory = Path(prompts_dir) if prompts_dir else PROMPTS_DIR
    registry_path = directory / "registry.json"
    if not registry_path.is_file():
        raise PromptError(f"Prompt registry not found: {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if version not in registry:
        raise PromptError(f"Unknown prompt version {version!r}; registered: {sorted(registry)}")
    entry = registry[version]
    path = directory / entry["file"]
    if not path.is_file():
        raise PromptError(f"Prompt file missing for {version!r}: {path}")
    text = normalize_prompt(path.read_text(encoding="utf-8"))
    if not text:
        raise PromptError(f"Prompt {version!r} is empty after normalization")
    digest = sha256_text(text)
    if digest != entry["sha256"]:
        raise PromptError(
            f"Prompt {version!r} text hashes to {digest} but the registry pins "
            f"{entry['sha256']}. Do not edit a registered prompt; create a new version.")
    return Prompt(version=version, sha256=digest, text=text)


# ---------------------------------------------------------------------------
# Configuration and run identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSource:
    dataset_id: str = DATASET_ID
    revision: str = "UNRESOLVED"
    split: str = DATASET_SPLIT
    row_start: int = ROW_START
    row_limit: int = ROW_LIMIT


@dataclass(frozen=True)
class ModelSource:
    model_id: str = MODEL_ID
    revision: str = "UNRESOLVED"
    processor_revision: str = "UNRESOLVED"
    path: str = ""


@dataclass(frozen=True)
class InferenceConfig:
    """Validated production baseline.  Fixed fields intentionally have no CLI switch."""

    engine: str = "vllm-offline"
    tensor_parallel: int = 1
    replicas: int = 2
    replicas_per_gpu: int = 1
    aggregate_concurrency: int = 64
    mtp: int = 1
    enable_thinking: bool = False
    greedy: bool = True
    batch_token_budget: int = 16_384
    context: int = 32_768
    max_output_tokens: int = 256
    truncation_retry_tokens: int = 384
    prefix_caching: bool = False
    custom_all_reduce: bool = False
    nccl_p2p: str = "disabled"
    gpu_memory_utilization: float = 0.90

    def per_replica_concurrency(self, index: int) -> int:
        return self.aggregate_concurrency // self.replicas + (
            1 if index < self.aggregate_concurrency % self.replicas else 0)

    def validate(self) -> None:
        if self.engine != "vllm-offline":
            raise ConfigError(f"Production engine is vllm-offline, got {self.engine!r}")
        if (self.tensor_parallel, self.replicas) != (1, 2):
            raise ConfigError(
                "Production baseline is TP1 with exactly two replicas, one per GPU "
                f"(got tp={self.tensor_parallel}, replicas={self.replicas})")
        if self.replicas_per_gpu != 1:
            raise ConfigError("Colocated replicas are the rejected topology; replicas_per_gpu must be 1")
        if self.enable_thinking:
            raise ConfigError("Thinking must be explicitly disabled for production captioning")
        if not self.greedy:
            raise ConfigError("Production decoding is greedy (temperature 0)")
        if self.prefix_caching:
            raise ConfigError("Prefix caching stays disabled in the validated baseline")
        if self.custom_all_reduce:
            raise ConfigError("vLLM custom all-reduce stays disabled in the validated baseline")
        if self.nccl_p2p not in ("disabled", "auto"):
            raise ConfigError(f"Unknown nccl_p2p mode {self.nccl_p2p!r}")
        if self.mtp not in (0, 1, 2, 3):
            raise ConfigError(f"MTP must be 0-3, got {self.mtp}")
        if self.aggregate_concurrency < self.replicas:
            raise ConfigError("Aggregate concurrency must cover every replica")
        if self.batch_token_budget < 1:
            raise ConfigError("batch_token_budget must be positive")
        if self.context < 4096:
            raise ConfigError("Context must be at least 4096 tokens")
        if self.max_output_tokens < 1:
            raise ConfigError("max_output_tokens must be positive")
        if self.truncation_retry_tokens <= self.max_output_tokens:
            raise ConfigError("truncation_retry_tokens must exceed max_output_tokens")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ConfigError("gpu_memory_utilization must be between 0 and 1")


@dataclass(frozen=True)
class ValidationPolicy:
    version: str = VALIDATION_POLICY_VERSION
    min_chars: int = 20
    max_chars: int = 2000
    max_words: int = 120

    def validate(self) -> None:
        if self.min_chars < 1 or self.max_chars < self.min_chars or self.max_words < 1:
            raise ConfigError("Invalid instruction length policy")


@dataclass(frozen=True)
class RunConfig:
    """Everything scientifically relevant about one production run.

    ``git_commit`` and ``model_path`` are recorded for provenance but excluded
    from the identity hash: code changes are captured by content hashes, and the
    local snapshot path does not change the generated data.
    """

    dataset: DatasetSource
    model: ModelSource
    prompt: Prompt
    inference: InferenceConfig
    validation: ValidationPolicy
    container_sha256: str = "UNRESOLVED"
    runner_sha256: str = ""
    helper_sha256: str = ""
    schema_version: str = DATASET_SCHEMA_VERSION
    git_commit: str = ""

    def identity(self) -> dict:
        return {
            "identity_version": IDENTITY_VERSION,
            "dataset": {
                "dataset_id": self.dataset.dataset_id,
                "revision": self.dataset.revision,
                "split": self.dataset.split,
                "row_start": self.dataset.row_start,
                "row_limit": self.dataset.row_limit,
            },
            "model": {
                "model_id": self.model.model_id,
                "revision": self.model.revision,
                "processor_revision": self.model.processor_revision,
            },
            "prompt": {"version": self.prompt.version, "sha256": self.prompt.sha256},
            "runner": {"code_sha256": self.runner_sha256, "helpers_sha256": self.helper_sha256},
            "container": {"vllm_sif_sha256": self.container_sha256},
            "inference": dataclasses.asdict(self.inference),
            "validation": dataclasses.asdict(self.validation),
            "output": {"schema_version": self.schema_version},
        }

    def identity_sha256(self) -> str:
        return canonical_digest(self.identity())

    def run_id(self) -> str:
        return self.identity_sha256()[:16]

    def record(self) -> dict:
        return {
            "identity": self.identity(),
            "identity_sha256": self.identity_sha256(),
            "run_id": self.run_id(),
            "git_commit": self.git_commit,
            "model_path": self.model.path,
            "tool_version": TOOL_VERSION,
            "row_id_version": ROW_ID_VERSION,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        }

    def require_pinned(self) -> None:
        if not is_pinned_revision(self.dataset.revision):
            raise ConfigError(
                f"Dataset revision must be a pinned 40-char commit sha, got {self.dataset.revision!r}. "
                "Run prepare to pin the revision (or pass --dataset-revision).")
        if not is_pinned_revision(self.model.revision):
            raise ConfigError(
                f"Model revision must be a pinned 40-char commit sha, got {self.model.revision!r}. "
                "Run prepare to pin the revision (or pass --model-revision).")
        if not is_sha256(self.container_sha256):
            raise ConfigError(
                f"Container hash must be a SHA256 of the vLLM SIF, got {self.container_sha256!r}. "
                "Pass --vllm-sif to run.")
        if not is_sha256(self.runner_sha256) or not is_sha256(self.helper_sha256):
            raise ConfigError("Runner code hashes are missing")

    def validate(self, require_pinned: bool = False) -> None:
        if self.dataset.dataset_id != DATASET_ID:
            raise ConfigError(f"Production dataset is {DATASET_ID}, got {self.dataset.dataset_id!r}")
        if self.dataset.split != DATASET_SPLIT:
            raise ConfigError(f"Production split is {DATASET_SPLIT!r}, got {self.dataset.split!r}")
        if (self.dataset.row_start, self.dataset.row_limit) != (ROW_START, ROW_LIMIT):
            raise ConfigError(
                f"Production freezes source rows {ROW_START}-{ROW_START + ROW_LIMIT - 1} exactly")
        self.inference.validate()
        self.validation.validate()
        if not self.prompt.text or self.prompt.sha256 != sha256_text(self.prompt.text):
            raise ConfigError("Prompt text does not match its recorded hash")
        if self.schema_version != DATASET_SCHEMA_VERSION:
            raise ConfigError(f"Unknown output schema {self.schema_version!r}")
        if require_pinned:
            self.require_pinned()


def identity_diff(left, right, path: str = "") -> list[str]:
    """Human-readable differences between two identity payloads."""
    differences: list[str] = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                differences.append(f"{path}{key}: missing from stored identity")
            elif key not in right:
                differences.append(f"{path}{key}: missing from requested identity")
            else:
                differences.extend(identity_diff(left[key], right[key], f"{path}{key}."))
    elif left != right:
        differences.append(f"{path.rstrip('.')}: stored {left!r} != requested {right!r}")
    return differences


def run_record_path(work) -> Path:
    return Path(work) / "run.json"


def build_run_record(config: RunConfig, created_at=None) -> dict:
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created_at)),
        "tool_version": TOOL_VERSION,
        "identity": config.identity(),
        "identity_sha256": config.identity_sha256(),
        "run_id": config.run_id(),
        "provenance": {
            "git_commit": config.git_commit,
            "model_path": config.model.path,
            "runner_sha256": config.runner_sha256,
            "helper_sha256": config.helper_sha256,
        },
    }


def load_run_record(work) -> dict | None:
    path = run_record_path(work)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def verify_run_record(work, config: RunConfig) -> dict:
    """Return the stored run record or raise IdentityMismatch on any drift."""
    stored = load_run_record(work)
    if stored is None:
        raise ConfigError(f"No run record at {run_record_path(work)}; run prepare first")
    differences = identity_diff(stored.get("identity", {}), config.identity())
    if differences:
        raise IdentityMismatch(differences)
    return stored


def write_run_record(work, config: RunConfig) -> dict:
    record = build_run_record(config)
    bench.dump(run_record_path(work), record)
    return record


def config_from_args(args, *, manifest_meta=None, prompt: Prompt | None = None,
                     container_sha256: str = "UNRESOLVED", require_pinned: bool = False) -> RunConfig:
    meta = manifest_meta or {}
    prompt = prompt or load_prompt(args.prompt_version)
    if meta:
        if args.dataset_revision and args.dataset_revision != meta.get("dataset_revision"):
            raise ConfigError(
                f"--dataset-revision {args.dataset_revision} does not match the frozen manifest "
                f"{meta.get('dataset_revision')}")
        if args.model_revision and args.model_revision != meta.get("model_revision"):
            raise ConfigError(
                f"--model-revision {args.model_revision} does not match the frozen manifest "
                f"{meta.get('model_revision')}")
        if meta.get("dataset_id") and args.dataset != meta["dataset_id"]:
            raise ConfigError(f"--dataset {args.dataset} does not match the frozen manifest")
        if meta.get("model_id") and args.model != meta["model_id"]:
            raise ConfigError(f"--model {args.model} does not match the frozen manifest")
    dataset_revision = meta.get("dataset_revision") or args.dataset_revision or "UNRESOLVED"
    model_revision = meta.get("model_revision") or args.model_revision or "UNRESOLVED"
    config = RunConfig(
        dataset=DatasetSource(dataset_id=args.dataset, revision=dataset_revision),
        model=ModelSource(model_id=args.model, revision=model_revision,
                          processor_revision=model_revision, path=meta.get("model_path", "")),
        prompt=prompt,
        inference=InferenceConfig(
            aggregate_concurrency=args.concurrency, mtp=args.mtp,
            batch_token_budget=args.batch_token_budget, context=args.context,
            max_output_tokens=args.max_output_tokens,
            truncation_retry_tokens=args.truncation_retry_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization, nccl_p2p=args.nccl_p2p),
        validation=ValidationPolicy(min_chars=args.min_instruction_chars,
                                    max_chars=args.max_instruction_chars,
                                    max_words=args.max_instruction_words),
        container_sha256=container_sha256,
        runner_sha256=sha256_file(Path(__file__).resolve()),
        helper_sha256=sha256_file(HERE / "benchmark.py"),
        git_commit=detect_git_commit(),
    )
    config.validate(require_pinned=require_pinned)
    return config


# ---------------------------------------------------------------------------
# Manifest freeze
# ---------------------------------------------------------------------------

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ManifestError(RuntimeError):
    """Raised when a frozen manifest is missing, partial, or inconsistent."""


@dataclass(frozen=True)
class FreezeLimits:
    max_tikz_chars: int = 200_000
    max_image_bytes: int = 20_000_000
    estimate_rows: int = 200
    safety_factor: float = 1.5
    reserve_bytes: int = 5 * 1024**3


def manifest_path(work) -> Path:
    return Path(work) / "manifest.jsonl"


def manifest_meta_path(work) -> Path:
    return Path(work) / "manifest.meta.json"


def images_dir(work) -> Path:
    return Path(work) / "images"


def stable_row_id(dataset_id: str, revision: str, split: str, source_index: int,
                  tikz_sha256: str, image_sha256: str) -> str:
    """Deterministic identity from immutable source information only."""
    payload = "\x1f".join((ROW_ID_VERSION, dataset_id, revision, split,
                           str(source_index), tikz_sha256, image_sha256))
    return sha256_text(payload)[:32]


def extract_image_bytes(value) -> bytes | None:
    """Accept the shapes the Hub can hand back with decode=False."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        try:
            return base64.b64decode(value, validate=True)
        except Exception:
            path = Path(value)
            return path.read_bytes() if path.is_file() else None
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)
        if isinstance(raw, str):
            try:
                return base64.b64decode(raw, validate=True)
            except Exception:
                return None
        path = value.get("path")
        if path and Path(path).is_file():
            return Path(path).read_bytes()
    return None


def pillow_available() -> bool:
    try:
        import PIL.Image  # noqa: F401
        return True
    except Exception:
        return False


def inspect_image(raw: bytes | None, limits: FreezeLimits) -> str | None:
    """Return a rejection reason for unusable image bytes, else None."""
    if not raw:
        return "missing_image"
    if len(raw) > limits.max_image_bytes:
        return "image_too_large"
    if not raw.startswith(PNG_SIGNATURE):
        return "invalid_image"
    if pillow_available():
        try:
            from PIL import Image
            with io.BytesIO(raw) as buffer:
                with Image.open(buffer) as image:
                    image.verify()
            with io.BytesIO(raw) as buffer:
                with Image.open(buffer) as image:
                    if image.size[0] < 1 or image.size[1] < 1:
                        return "invalid_image"
        except Exception:
            return "invalid_image"
    return None


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def freeze_row(index: int, row: dict, *, dataset_id: str, revision: str, split: str,
               limits: FreezeLimits, stage_images: Path | None) -> dict:
    """Build one manifest entry.  Invalid rows are recorded, never replaced."""
    tikz = row.get("tikz_code")
    tikz = tikz if isinstance(tikz, str) else ""
    raw = extract_image_bytes(row.get("png_image"))
    reason = None
    if not tikz.strip():
        reason = "empty_tikz"
    elif len(tikz) > limits.max_tikz_chars:
        reason = "tikz_too_long"
    else:
        reason = inspect_image(raw, limits)
    tikz_sha = sha256_text(tikz)
    image_sha = sha256_bytes(raw or b"")
    entry = {
        "row_id": stable_row_id(dataset_id, revision, split, index, tikz_sha, image_sha),
        "source_row_index": index,
        "file_id": _json_safe(row.get("file_id")),
        "source": _json_safe(row.get("source")),
        "dataset_id": dataset_id,
        "dataset_revision": revision,
        "split": split,
        "tikz_code": tikz,
        "tikz_sha256": tikz_sha,
        "image": None,
        "image_sha256": image_sha,
        "status": "valid" if reason is None else "rejected",
        "rejection_reason": reason,
    }
    if reason is None:
        entry["image"] = f"images/{index:06d}.png"
        if stage_images is not None:
            (Path(stage_images) / f"{index:06d}.png").write_bytes(raw)
    return entry


def estimate_storage(rows, limits: FreezeLimits, row_limit: int = ROW_LIMIT) -> dict:
    image_sizes, tikz_sizes = [], []
    for row in rows:
        image_sizes.append(len(extract_image_bytes(row.get("png_image")) or b""))
        tikz = row.get("tikz_code")
        tikz_sizes.append(len(tikz) if isinstance(tikz, str) else 0)
    count = max(1, len(image_sizes))
    average_image = sum(image_sizes) / count
    average_tikz = sum(tikz_sizes) / count
    projected = int(row_limit * (average_image + average_tikz + 256))
    return {
        "sample_rows": len(image_sizes),
        "average_image_bytes": round(average_image, 1),
        "average_tikz_bytes": round(average_tikz, 1),
        "projected_bytes": projected,
    }


def check_disk_space(work, estimate: dict, limits: FreezeLimits,
                     row_limit: int = ROW_LIMIT) -> int:
    required = int(estimate["projected_bytes"] * limits.safety_factor) + limits.reserve_bytes
    free = shutil.disk_usage(str(work)).free
    if free < required:
        raise SystemExit(
            f"Insufficient free space under {work}: need about {human_bytes(required)} "
            f"(projected {human_bytes(estimate['projected_bytes'])} x safety {limits.safety_factor} "
            f"+ {human_bytes(limits.reserve_bytes)} reserve), have {human_bytes(free)}. "
            "Free space or choose a different --work directory before freezing.")
    return required


def human_bytes(value) -> str:
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def fsync_dir(path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def clean_incomplete_freeze(work) -> list[str]:
    """Remove artifacts from an interrupted freeze.  Never touches a committed manifest."""
    removed = []
    for path in (manifest_path(work), images_dir(work)):
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(str(path))
        elif path.exists():
            path.unlink()
            removed.append(str(path))
    for staging in Path(work).glob(".staging-*"):
        shutil.rmtree(staging, ignore_errors=True)
        removed.append(str(staging))
    return removed


def freeze_source(rows, *, work, dataset_id: str, revision: str, split: str,
                  limits: FreezeLimits, meta_extra: dict | None = None,
                  row_limit: int = ROW_LIMIT, write_images: bool = True,
                  progress_every: int = 0) -> dict:
    """Freeze exactly ``row_limit`` streamed rows and commit the manifest atomically.

    A partial freeze leaves no ``manifest.jsonl``/``manifest.meta.json`` at the
    top level; ``manifest.meta.json`` is written last and is the only marker
    that makes a manifest authoritative.
    """
    work = Path(work)
    staging = work / f".staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    stage_images = staging / "images"
    stage_images.mkdir(parents=True)
    manifest_tmp = staging / "manifest.jsonl"
    counts = {"valid": 0, "rejected": 0}
    rejections: dict[str, int] = {}
    written = 0
    try:
        with manifest_tmp.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(islice(rows, row_limit)):
                entry = freeze_row(index, row, dataset_id=dataset_id, revision=revision,
                                   split=split, limits=limits,
                                   stage_images=stage_images if write_images else None)
                counts[entry["status"]] += 1
                if entry["rejection_reason"]:
                    rejections[entry["rejection_reason"]] = rejections.get(
                        entry["rejection_reason"], 0) + 1
                handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
                if progress_every and written % progress_every == 0:
                    print(f"  frozen {written}/{row_limit} rows", flush=True)
            handle.flush()
            os.fsync(handle.fileno())
        if written != row_limit:
            raise ManifestError(
                f"Dataset stream ended after {written} rows; production requires exactly "
                f"{row_limit} source rows. Check the pinned revision and split.")
        manifest_sha = sha256_file(manifest_tmp)
        manifest_bytes = manifest_tmp.stat().st_size
        final_images = images_dir(work)
        if final_images.exists():
            shutil.rmtree(final_images)
        os.replace(stage_images, final_images)
        os.replace(manifest_tmp, manifest_path(work))
        fsync_dir(work)
        meta = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tool_version": TOOL_VERSION,
            "dataset_id": dataset_id,
            "dataset_revision": revision,
            "split": split,
            "row_start": 0,
            "row_limit": row_limit,
            "rows_frozen": written,
            "valid_rows": counts["valid"],
            "rejected_rows": counts["rejected"],
            "rejection_counts": rejections,
            "row_id_version": ROW_ID_VERSION,
            "image_validation": "pillow" if pillow_available() else "signature",
            "manifest_sha256": manifest_sha,
            "manifest_bytes": manifest_bytes,
            "freeze_limits": dataclasses.asdict(limits),
        }
        meta.update(meta_extra or {})
        bench.dump(manifest_meta_path(work), meta)
        fsync_dir(work)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return meta


def verify_manifest(work, *, quick: bool = True) -> dict:
    """Verify the frozen manifest and return its metadata; raises ManifestError."""
    work = Path(work)
    meta_file = manifest_meta_path(work)
    if not meta_file.is_file():
        raise ManifestError(
            f"No frozen manifest at {meta_file}. A manifest is authoritative only after "
            "prepare writes manifest.meta.json.")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if meta.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(f"Manifest schema {meta.get('schema_version')!r} is not supported")
    path = manifest_path(work)
    if not path.is_file():
        raise ManifestError(f"manifest.jsonl missing under {work}")
    digest = hashlib.sha256()
    count = 0
    seen_ids = set()
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if not line.strip():
                continue
            entry = json.loads(line)
            index = entry.get("source_row_index")
            if index != count:
                raise ManifestError(
                    f"Manifest row {count} has source_row_index {index}; indices must be contiguous from 0")
            if entry["row_id"] in seen_ids:
                raise ManifestError(f"Duplicate stable id {entry['row_id']} at source row {index}")
            seen_ids.add(entry["row_id"])
            expected = stable_row_id(entry["dataset_id"], entry["dataset_revision"],
                                     entry["split"], index, entry["tikz_sha256"],
                                     entry["image_sha256"])
            if expected != entry["row_id"]:
                raise ManifestError(f"Stable id mismatch at source row {index}")
            if sha256_text(entry["tikz_code"]) != entry["tikz_sha256"]:
                raise ManifestError(f"TikZ checksum mismatch at source row {index}")
            if entry["status"] == "valid":
                if not entry.get("image"):
                    raise ManifestError(f"Valid row {index} has no image path")
                image_path = work / entry["image"]
                if not image_path.is_file():
                    raise ManifestError(f"Image missing for source row {index}: {image_path}")
                if not quick and sha256_file(image_path) != entry["image_sha256"]:
                    raise ManifestError(f"Image checksum mismatch at source row {index}")
            elif entry["status"] == "rejected":
                if not entry.get("rejection_reason"):
                    raise ManifestError(f"Rejected row {index} has no rejection reason")
            else:
                raise ManifestError(f"Unknown row status {entry['status']!r} at source row {index}")
            count += 1
    if count != meta.get("rows_frozen"):
        raise ManifestError(f"Manifest holds {count} rows but meta records {meta.get('rows_frozen')}")
    if digest.hexdigest() != meta.get("manifest_sha256"):
        raise ManifestError("Manifest content hash does not match manifest.meta.json")
    return meta


def iter_manifest(work):
    with manifest_path(work).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def manifest_index(work) -> dict:
    """Compact per-row identity index used for inference and audit checks."""
    index = {}
    for entry in iter_manifest(work):
        index[entry["row_id"]] = {
            "source_row_index": entry["source_row_index"],
            "tikz_sha256": entry["tikz_sha256"],
            "image_sha256": entry["image_sha256"],
            "image": entry["image"],
            "status": entry["status"],
            "tikz_chars": len(entry["tikz_code"]),
        }
    return index


def check_manifest_agreement(args, meta: dict) -> None:
    if args.dataset != meta.get("dataset_id"):
        raise ConfigError(f"--dataset {args.dataset} does not match frozen manifest "
                          f"{meta.get('dataset_id')}")
    if args.dataset_revision and args.dataset_revision != meta.get("dataset_revision"):
        raise ConfigError(f"--dataset-revision {args.dataset_revision} does not match frozen "
                          f"manifest {meta.get('dataset_revision')}")
    if args.model != meta.get("model_id"):
        raise ConfigError(f"--model {args.model} does not match frozen manifest {meta.get('model_id')}")
    if args.model_revision and args.model_revision != meta.get("model_revision"):
        raise ConfigError(f"--model-revision {args.model_revision} does not match frozen "
                          f"manifest {meta.get('model_revision')}")


class PrepareDeps:
    """Injectable seams so prepare runs without network access in tests."""

    def resolve_dataset_revision(self, args) -> str:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(args.dataset, revision=args.dataset_revision or "main").sha

    def resolve_model_revision(self, args) -> str:
        from huggingface_hub import HfApi
        return HfApi().model_info(args.model, revision=args.model_revision or "main").sha

    def download_model(self, args, revision: str) -> str:
        from huggingface_hub import snapshot_download
        cache = Path(args.model_cache_dir).resolve() if args.model_cache_dir \
            else Path(args.work).resolve() / "hf"
        return str(Path(snapshot_download(args.model, revision=revision,
                                          cache_dir=str(cache))).resolve())

    def row_source(self, args, revision: str):
        from datasets import Image, load_dataset
        dataset = load_dataset(args.dataset, split=DATASET_SPLIT, revision=revision,
                               streaming=True).cast_column("png_image", Image(decode=False))
        return iter(dataset)


def cmd_prepare(args, deps=None) -> int:
    require_slurm(args)
    deps = deps or PrepareDeps()
    work = Path(args.work).resolve()
    row_limit = getattr(args, "row_limit", ROW_LIMIT)
    if row_limit != ROW_LIMIT and not args.allow_non_slurm:
        raise ConfigError(
            "--row-limit is a synthetic-fixture control; production freezes exactly "
            f"{ROW_LIMIT} rows")
    limits = FreezeLimits(max_tikz_chars=args.max_tikz_chars,
                          max_image_bytes=args.max_image_bytes,
                          estimate_rows=args.estimate_rows)
    if manifest_meta_path(work).is_file():
        meta = verify_manifest(work, quick=True)
        check_manifest_agreement(args, meta)
        model_path = meta.get("model_path", "")
        if args.download_model and not model_path:
            model_path = deps.download_model(args, meta["model_revision"])
            meta["model_path"] = model_path
            bench.dump(manifest_meta_path(work), meta)
        print(f"Reusing frozen manifest: {manifest_path(work)}")
        print(f"  rows={meta['rows_frozen']} valid={meta['valid_rows']} "
              f"rejected={meta['rejected_rows']} dataset_revision={meta['dataset_revision']}")
        print(f"  model_revision={meta['model_revision']}")
        if model_path:
            print(f"  model_path={model_path}")
        if meta.get("rejection_counts"):
            print(f"  rejection_counts={meta['rejection_counts']}")
        return 0
    work.mkdir(parents=True, exist_ok=True)
    removed = clean_incomplete_freeze(work)
    if removed:
        print(f"Removed incomplete freeze artifacts: {', '.join(removed)}")
    dataset_revision = deps.resolve_dataset_revision(args)
    model_revision = deps.resolve_model_revision(args)
    print(f"Pinned dataset revision: {dataset_revision}")
    print(f"Pinned model revision:   {model_revision}")
    sample = list(islice(deps.row_source(args, dataset_revision), limits.estimate_rows))
    if len(sample) < min(limits.estimate_rows, row_limit):
        raise SystemExit(
            f"Dataset stream returned only {len(sample)} rows for the storage estimate; "
            "check the pinned revision, split and network access.")
    estimate = estimate_storage(sample, limits, row_limit)
    required = check_disk_space(work, estimate, limits, row_limit)
    print(f"Storage estimate from {estimate['sample_rows']} rows: "
          f"avg image {human_bytes(estimate['average_image_bytes'])}, "
          f"avg TikZ {human_bytes(estimate['average_tikz_bytes'])}, "
          f"projected {human_bytes(estimate['projected_bytes'])}, "
          f"required {human_bytes(required)}")
    model_path = deps.download_model(args, model_revision) if args.download_model else ""
    if model_path:
        print(f"Model snapshot ready: {model_path}")
    meta_extra = {
        "model_id": args.model,
        "model_revision": model_revision,
        "model_path": model_path,
        "storage": dict(estimate, required_bytes=required),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    print(f"Freezing source rows 0-{row_limit - 1} from {args.dataset}@{dataset_revision} ...")
    meta = freeze_source(deps.row_source(args, dataset_revision), work=work,
                         dataset_id=args.dataset, revision=dataset_revision,
                         split=DATASET_SPLIT, limits=limits, meta_extra=meta_extra,
                         row_limit=row_limit, progress_every=5000)
    print(f"Frozen manifest complete: {manifest_path(work)}")
    print(f"  rows={meta['rows_frozen']} valid={meta['valid_rows']} rejected={meta['rejected_rows']}")
    if meta["rejection_counts"]:
        print(f"  rejection_counts={meta['rejection_counts']}")
    print(f"  manifest_sha256={meta['manifest_sha256']}")
    return 0


# ---------------------------------------------------------------------------
# Durable state ledger
# ---------------------------------------------------------------------------

LEDGER_FILE = "ledger.sqlite3"
LEDGER_BACKUP_DIR = "ledger-backups"

# Transient categories consume the bounded retry budget; interruption categories
# do not, because they say nothing about the row itself.
TRANSIENT_CATEGORIES = frozenset({
    "engine_transient", "engine_fatal", "timeout", "io", "transport", "decode",
    "worker_lost", "worker_stall",
})
NON_COUNTING_CATEGORIES = frozenset({"stale_claim", "run_interrupted"})
PERMANENT_CATEGORIES = frozenset({
    "input_invalid", "integrity", "input_too_long", "truncated_at_ceiling",
    "transient_exhausted", "invalid_instruction",
})

LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rows (
    row_id TEXT PRIMARY KEY,
    source_row_index INTEGER NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK (state IN ('pending','running','retryable','complete','rejected')),
    attempt_count INTEGER NOT NULL DEFAULT 0,
    claimed_at REAL,
    worker TEXT,
    not_before REAL NOT NULL DEFAULT 0,
    last_error_category TEXT,
    last_error_detail TEXT,
    rejection_reason TEXT,
    completed_at REAL,
    instruction TEXT,
    finish_reason TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    model_revision TEXT,
    prompt_sha256 TEXT,
    config_hash TEXT,
    image_sha256 TEXT NOT NULL,
    tikz_sha256 TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    row_id TEXT NOT NULL REFERENCES rows(row_id),
    attempt_no INTEGER NOT NULL,
    attempt_kind TEXT NOT NULL,
    state TEXT NOT NULL,
    worker TEXT,
    max_tokens INTEGER,
    started_at REAL NOT NULL,
    ended_at REAL,
    finish_reason TEXT,
    error_category TEXT,
    error_detail TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    instruction TEXT,
    model_revision TEXT,
    prompt_sha256 TEXT,
    config_hash TEXT
);
CREATE INDEX IF NOT EXISTS attempts_by_row ON attempts(row_id);
CREATE INDEX IF NOT EXISTS rows_by_state ON rows(state, source_row_index);
"""

LEDGER_META_KEYS = ("schema_version", "run_id", "identity_sha256", "manifest_sha256",
                    "dataset_revision", "model_revision", "prompt_sha256")


class LedgerError(RuntimeError):
    """Raised on invalid ledger transitions or identity drift."""


@dataclass(frozen=True)
class RetryPolicy:
    max_transient_attempts: int = 3
    backoff_base_seconds: float = 30.0
    backoff_cap_seconds: float = 600.0
    normal_max_tokens: int = 256
    truncation_max_tokens: int = 384

    @classmethod
    def from_inference(cls, inference) -> "RetryPolicy":
        return cls(normal_max_tokens=inference.max_output_tokens,
                   truncation_max_tokens=inference.truncation_retry_tokens)


@dataclass(frozen=True)
class Decision:
    state: str
    reason: str
    category: str
    backoff_seconds: float = 0.0
    detail: str = ""


def next_max_tokens(attempts, policy: RetryPolicy) -> int:
    """Escalate to the retry ceiling once a 256-token attempt truncated."""
    if any((attempt.get("max_tokens") or 0) >= policy.truncation_max_tokens
           for attempt in attempts):
        return policy.truncation_max_tokens
    if any(attempt.get("finish_reason") == "length" for attempt in attempts):
        return policy.truncation_max_tokens
    return policy.normal_max_tokens


def decide_result(result: dict, attempts, policy: RetryPolicy) -> Decision:
    """Map a finished attempt (and the row's history) to the next row state.

    Pure function: no I/O, fully unit-testable.  ``attempts`` includes the
    attempt that just finished.
    """
    if (result.get("ok") and result.get("finish_reason") == "stop"
            and (result.get("instruction") or "").strip()):
        return Decision(STATE_COMPLETE, "ok", "none")
    category = result.get("error_category") or "engine_transient"
    detail = result.get("error_detail") or ""
    if category in ("integrity", "input_too_long"):
        return Decision(STATE_REJECTED, category, category, detail=detail)
    if result.get("finish_reason") == "length":
        retry_attempted = any((attempt.get("max_tokens") or 0) >= policy.truncation_max_tokens
                              for attempt in attempts)
        if retry_attempted:
            return Decision(STATE_REJECTED, "truncated_at_ceiling", "truncation",
                            detail=f"output still truncated at {policy.truncation_max_tokens} tokens")
        return Decision(STATE_RETRYABLE, "truncated", "truncation",
                        detail=f"retry at {policy.truncation_max_tokens} tokens")
    counted = [attempt for attempt in attempts
               if attempt.get("error_category") in TRANSIENT_CATEGORIES]
    if len(counted) >= policy.max_transient_attempts:
        return Decision(STATE_REJECTED, "transient_exhausted", "transient",
                        detail=f"{len(counted)} transient attempts: {detail}")
    if category in NON_COUNTING_CATEGORIES:
        return Decision(STATE_RETRYABLE, category, "transient", detail=detail)
    backoff = min(policy.backoff_cap_seconds,
                  policy.backoff_base_seconds * 2 ** max(0, len(counted) - 1))
    return Decision(STATE_RETRYABLE, category, "transient", backoff_seconds=backoff,
                    detail=detail)


class Ledger:
    """SQLite state ledger.  Only the controller process writes to it."""

    def __init__(self, work, read_only: bool = False):
        self.work = Path(work)
        self.path = self.work / LEDGER_FILE
        self.read_only = read_only
        self.conn: sqlite3.Connection | None = None

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> "Ledger":
        if self.conn is not None:
            return self
        if self.read_only:
            if not self.path.is_file():
                raise LedgerError(f"No ledger at {self.path}")
            self.conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=30)
        else:
            self.work.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
            self.conn.execute("PRAGMA journal_mode=DELETE")
            self.conn.execute("PRAGMA synchronous=FULL")
            self.conn.execute("PRAGMA busy_timeout=30000")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.executescript(LEDGER_SCHEMA)
        self.conn.row_factory = sqlite3.Row
        return self

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc_info):
        self.close()

    def transaction(self):
        ledger = self

        class _Transaction:
            def __enter__(self):
                ledger.conn.execute("BEGIN IMMEDIATE")
                return ledger

            def __exit__(self, exc_type, exc, tb):
                if exc_type is None:
                    ledger.conn.execute("COMMIT")
                else:
                    ledger.conn.execute("ROLLBACK")
                return False
        if self.read_only:
            raise LedgerError("Read-only ledger cannot start a write transaction")
        return _Transaction()

    # -- identity ----------------------------------------------------------

    def initialize(self, values: dict) -> None:
        """Create or extend ledger metadata; refuses conflicting values."""
        missing = [key for key in LEDGER_META_KEYS if key not in values]
        if missing:
            raise LedgerError(f"Ledger identity missing keys: {missing}")
        with self.transaction():
            for key in LEDGER_META_KEYS:
                self.conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
                                  (key, str(values[key])))
            self.conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('tool_version', ?)",
                              (TOOL_VERSION,))
            self.conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('created_at', ?)",
                              (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),))
        self.verify_identity(values)

    def meta(self) -> dict:
        return {row["key"]: row["value"] for row in self.conn.execute("SELECT key, value FROM meta")}

    def verify_identity(self, values: dict) -> dict:
        stored = self.meta()
        differences = [f"{key}: ledger {stored.get(key)!r} != requested {values[key]!r}"
                       for key in LEDGER_META_KEYS
                       if stored.get(key) != str(values[key])]
        if differences:
            raise LedgerError(
                "Ledger belongs to a different run identity:\n  " + "\n  ".join(differences))
        return stored

    # -- seeding -----------------------------------------------------------

    def seed(self, entries, *, rows_frozen: int) -> dict:
        """Idempotently load frozen manifest rows into the ledger.

        Valid rows become pending; manifest-invalid rows become rejected with
        their recorded reason.  Existing rows are verified, never overwritten.
        """
        counts = {"inserted": 0, "existing": 0, "rejected": 0}
        now = time.time()
        with self.transaction():
            for entry in entries:
                state = STATE_PENDING if entry["status"] == "valid" else STATE_REJECTED
                reason = entry.get("rejection_reason")
                existing = self.conn.execute(
                    "SELECT row_id, source_row_index, image_sha256, tikz_sha256, state "
                    "FROM rows WHERE row_id = ?", (entry["row_id"],)).fetchone()
                if existing is not None:
                    if (existing["source_row_index"] != entry["source_row_index"]
                            or existing["image_sha256"] != entry["image_sha256"]
                            or existing["tikz_sha256"] != entry["tikz_sha256"]):
                        raise LedgerError(
                            f"Ledger row {entry['row_id']} does not match the frozen manifest")
                    counts["existing"] += 1
                    continue
                self.conn.execute(
                    "INSERT INTO rows (row_id, source_row_index, state, rejection_reason, "
                    "last_error_category, last_error_detail, image_sha256, tikz_sha256, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (entry["row_id"], entry["source_row_index"], state, reason,
                     "input_invalid" if state == STATE_REJECTED else None,
                     reason, entry["image_sha256"], entry["tikz_sha256"], now))
                counts["inserted"] += 1
                if state == STATE_REJECTED:
                    counts["rejected"] += 1
            total = self.conn.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
            if total != rows_frozen:
                raise LedgerError(
                    f"Ledger holds {total} rows but the manifest froze {rows_frozen}; "
                    "the ledger and manifest do not belong together")
        return counts

    # -- queries -----------------------------------------------------------

    def get(self, row_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM rows WHERE row_id = ?", (row_id,)).fetchone()
        return dict(row) if row else None

    def eligible(self, *, start_index: int = 0, end_index: int = ROW_LIMIT - 1,
                 limit: int | None = None, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        sql = ("SELECT row_id, source_row_index, attempt_count FROM rows "
               "WHERE state IN (?, ?) AND not_before <= ? AND source_row_index BETWEEN ? AND ? "
               "ORDER BY source_row_index")
        parameters: list = [STATE_PENDING, STATE_RETRYABLE, now, start_index, end_index]
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(limit)
        return [dict(row) for row in self.conn.execute(sql, parameters)]

    def counts(self) -> dict:
        states = {state: 0 for state in STATES}
        for row in self.conn.execute("SELECT state, COUNT(*) AS n FROM rows GROUP BY state"):
            states[row["state"]] = row["n"]
        attempts = self.conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
        truncations = self.conn.execute(
            "SELECT COUNT(*) FROM attempts WHERE finish_reason = 'length'").fetchone()[0]
        errors = {row["error_category"]: row["n"] for row in self.conn.execute(
            "SELECT error_category, COUNT(*) AS n FROM attempts "
            "WHERE error_category IS NOT NULL GROUP BY error_category")}
        reasons = {row["rejection_reason"]: row["n"] for row in self.conn.execute(
            "SELECT rejection_reason, COUNT(*) AS n FROM rows WHERE state = ? "
            "GROUP BY rejection_reason", (STATE_REJECTED,))}
        tokens = self.conn.execute(
            "SELECT COALESCE(SUM(prompt_tokens),0) AS p, COALESCE(SUM(completion_tokens),0) AS c "
            "FROM rows WHERE state = ?", (STATE_COMPLETE,)).fetchone()
        window = self.conn.execute(
            "SELECT MIN(claimed_at) AS first_claim, MAX(completed_at) AS last_complete, "
            "MIN(completed_at) AS first_complete FROM rows").fetchone()
        return {
            "total": sum(states.values()),
            "states": states,
            "attempts": attempts,
            "truncation_attempts": truncations,
            "error_categories": errors,
            "rejection_reasons": reasons,
            "prompt_tokens": tokens["p"],
            "completion_tokens": tokens["c"],
            "first_claim_at": window["first_claim"],
            "first_complete_at": window["first_complete"],
            "last_complete_at": window["last_complete"],
        }

    def attempts_for(self, row_ids) -> dict:
        result: dict[str, list[dict]] = {row_id: [] for row_id in row_ids}
        row_ids = list(row_ids)
        for start in range(0, len(row_ids), 500):
            chunk = row_ids[start:start + 500]
            placeholders = ",".join("?" * len(chunk))
            for row in self.conn.execute(
                    f"SELECT * FROM attempts WHERE row_id IN ({placeholders}) "
                    "ORDER BY attempt_id", chunk):
                result[row["row_id"]].append(dict(row))
        return result

    def all_attempts(self) -> list[dict]:
        return [dict(row) for row in self.conn.execute(
            "SELECT * FROM attempts ORDER BY row_id, attempt_id")]

    def complete_rows(self) -> list[dict]:
        return [dict(row) for row in self.conn.execute(
            "SELECT * FROM rows WHERE state = ? ORDER BY source_row_index", (STATE_COMPLETE,))]

    def rejected_rows(self) -> list[dict]:
        return [dict(row) for row in self.conn.execute(
            "SELECT * FROM rows WHERE state = ? ORDER BY source_row_index", (STATE_REJECTED,))]

    # -- claims and transitions -------------------------------------------

    def claim(self, assignments: dict, *, policy: RetryPolicy,
              now: float | None = None) -> dict:
        """Atomically move rows to running and open one attempt per row."""
        now = time.time() if now is None else now
        claimed: dict[str, list[dict]] = {}
        with self.transaction():
            for worker, row_ids in assignments.items():
                claimed[worker] = []
                for row_id in row_ids:
                    row = self.conn.execute(
                        "SELECT * FROM rows WHERE row_id = ?", (row_id,)).fetchone()
                    if row is None:
                        raise LedgerError(f"Claim for unknown row {row_id}")
                    if row["state"] not in (STATE_PENDING, STATE_RETRYABLE):
                        raise LedgerError(
                            f"Claim conflict for {row_id}: state is {row['state']}")
                    if row["not_before"] > now:
                        raise LedgerError(
                            f"Claim conflict for {row_id}: not_before {row['not_before']} > {now}")
                    attempts = self._attempts(row_id)
                    max_tokens = next_max_tokens(attempts, policy)
                    attempt_kind = ("truncation_retry"
                                    if max_tokens > policy.normal_max_tokens else "normal")
                    attempt_no = row["attempt_count"] + 1
                    self.conn.execute(
                        "UPDATE rows SET state = ?, claimed_at = ?, worker = ?, "
                        "attempt_count = ?, updated_at = ? WHERE row_id = ?",
                        (STATE_RUNNING, now, worker, attempt_no, now, row_id))
                    self.conn.execute(
                        "INSERT INTO attempts (row_id, attempt_no, attempt_kind, state, worker, "
                        "max_tokens, started_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (row_id, attempt_no, attempt_kind, STATE_RUNNING, worker, max_tokens, now))
                    claimed[worker].append({
                        "row_id": row_id,
                        "source_row_index": row["source_row_index"],
                        "attempt_no": attempt_no,
                        "attempt_kind": attempt_kind,
                        "max_tokens": max_tokens,
                    })
        return claimed

    def _attempts(self, row_id: str) -> list[dict]:
        return [dict(row) for row in self.conn.execute(
            "SELECT * FROM attempts WHERE row_id = ? ORDER BY attempt_id", (row_id,))]

    def _open_attempt(self, row_id: str, worker: str | None = None):
        sql = "SELECT * FROM attempts WHERE row_id = ? AND state = ?"
        parameters: list = [row_id, STATE_RUNNING]
        if worker is not None:
            sql += " AND worker = ?"
            parameters.append(worker)
        sql += " ORDER BY attempt_id DESC LIMIT 1"
        return self.conn.execute(sql, parameters).fetchone()

    def _close_attempt(self, attempt_id: int, result: dict, state: str, now: float) -> None:
        self.conn.execute(
            "UPDATE attempts SET state = ?, ended_at = ?, finish_reason = ?, error_category = ?, "
            "error_detail = ?, prompt_tokens = ?, completion_tokens = ?, instruction = ?, "
            "model_revision = ?, prompt_sha256 = ?, config_hash = ? WHERE attempt_id = ?",
            (state, now, result.get("finish_reason"), result.get("error_category"),
             result.get("error_detail"), result.get("prompt_tokens"),
             result.get("completion_tokens"), result.get("instruction"),
             result.get("model_revision"), result.get("prompt_sha256"),
             result.get("config_hash"), attempt_id))

    def record_result(self, result: dict, *, policy: RetryPolicy,
                      now: float | None = None) -> str:
        """Commit one worker result transactionally and return the new row state."""
        now = time.time() if now is None else now
        row_id = result["row_id"]
        with self.transaction():
            row = self.conn.execute("SELECT * FROM rows WHERE row_id = ?", (row_id,)).fetchone()
            if row is None:
                raise LedgerError(f"Result for unknown row {row_id}")
            if row["state"] == STATE_COMPLETE:
                # Crash between commit and result-file consumption: verify and no-op.
                if row["instruction"] != result.get("instruction"):
                    raise LedgerError(
                        f"Row {row_id} is already complete with a different instruction; "
                        "refusing to overwrite a completed caption")
                return STATE_COMPLETE
            if row["state"] == STATE_REJECTED:
                return STATE_REJECTED  # terminal; late results are ignored
            attempt = self._open_attempt(row_id, result.get("worker"))
            if attempt is not None:
                # Represent the just-finished attempt by its own result so the
                # retry budget and truncation escalation see it immediately.
                attempts = [dict(result) if row["attempt_id"] == attempt["attempt_id"] else row
                            for row in self._attempts(row_id)]
                decision = decide_result(result, attempts, policy)
                self._close_attempt(attempt["attempt_id"], result, decision.state, now)
            else:
                # Recovered result from a crashed run: no open attempt remains.
                attempts = self._attempts(row_id) + [dict(result)]
                decision = decide_result(result, attempts, policy)
                attempt_no = row["attempt_count"] + 1
                self.conn.execute(
                    "INSERT INTO attempts (row_id, attempt_no, attempt_kind, state, worker, "
                    "max_tokens, started_at, ended_at) VALUES (?, ?, 'recovered', ?, ?, ?, ?, ?)",
                    (row_id, attempt_no, decision.state, result.get("worker"),
                     result.get("max_tokens"), now, now))
                self._close_attempt(self.conn.execute("SELECT last_insert_rowid()").fetchone()[0],
                                    result, decision.state, now)
                self.conn.execute(
                    "UPDATE rows SET attempt_count = ? WHERE row_id = ?", (attempt_no, row_id))
            if decision.state == STATE_COMPLETE:
                self.conn.execute(
                    "UPDATE rows SET state = ?, instruction = ?, finish_reason = ?, "
                    "prompt_tokens = ?, completion_tokens = ?, model_revision = ?, "
                    "prompt_sha256 = ?, config_hash = ?, completed_at = ?, "
                    "last_error_category = NULL, last_error_detail = NULL, updated_at = ? "
                    "WHERE row_id = ?",
                    (STATE_COMPLETE, result.get("instruction"), result.get("finish_reason"),
                     result.get("prompt_tokens"), result.get("completion_tokens"),
                     result.get("model_revision"), result.get("prompt_sha256"),
                     result.get("config_hash"), now, now, row_id))
            elif decision.state == STATE_RETRYABLE:
                self.conn.execute(
                    "UPDATE rows SET state = ?, not_before = ?, last_error_category = ?, "
                    "last_error_detail = ?, updated_at = ? WHERE row_id = ?",
                    (STATE_RETRYABLE, now + decision.backoff_seconds, decision.reason,
                     decision.detail, now, row_id))
            else:
                self.conn.execute(
                    "UPDATE rows SET state = ?, rejection_reason = ?, last_error_category = ?, "
                    "last_error_detail = ?, updated_at = ? WHERE row_id = ?",
                    (STATE_REJECTED, decision.reason, decision.category, decision.detail,
                     now, row_id))
        return decision.state

    def reclaim_stale(self, stale_seconds: float, *, policy: RetryPolicy,
                      now: float | None = None) -> int:
        """Return rows whose worker vanished to retryable without consuming budget."""
        now = time.time() if now is None else now
        return self._release(category="stale_claim", policy=policy, now=now,
                             older_than=now - stale_seconds)

    def release_running(self, *, worker: str | None = None, category: str,
                        policy: RetryPolicy, now: float | None = None) -> int:
        """Release a dead worker's rows (or all running rows) with retry accounting."""
        if category not in TRANSIENT_CATEGORIES and category not in NON_COUNTING_CATEGORIES:
            raise LedgerError(f"Unknown release category {category!r}")
        now = time.time() if now is None else now
        return self._release(category=category, policy=policy, now=now, worker=worker)

    def _release(self, *, category: str, policy: RetryPolicy, now: float,
                 worker: str | None = None, older_than: float | None = None) -> int:
        sql = "SELECT row_id FROM rows WHERE state = ?"
        parameters: list = [STATE_RUNNING]
        if worker is not None:
            sql += " AND worker = ?"
            parameters.append(worker)
        if older_than is not None:
            sql += " AND claimed_at < ?"
            parameters.append(older_than)
        row_ids = [row["row_id"] for row in self.conn.execute(sql, parameters)]
        released = 0
        with self.transaction():
            for row_id in row_ids:
                row = self.conn.execute("SELECT * FROM rows WHERE row_id = ?", (row_id,)).fetchone()
                attempts = self._attempts(row_id)
                placeholder = {"error_category": category, "error_detail": f"released as {category}"}
                synthetic = attempts + [dict(placeholder, max_tokens=None)]
                decision = decide_result(dict(placeholder, ok=False), synthetic, policy)
                open_attempt = self._open_attempt(row_id)
                if open_attempt is not None:
                    self._close_attempt(open_attempt["attempt_id"], placeholder, "lost", now)
                if decision.state == STATE_REJECTED:
                    self.conn.execute(
                        "UPDATE rows SET state = ?, rejection_reason = ?, last_error_category = ?, "
                        "last_error_detail = ?, updated_at = ? WHERE row_id = ?",
                        (STATE_REJECTED, decision.reason, decision.category, decision.detail,
                         now, row_id))
                else:
                    self.conn.execute(
                        "UPDATE rows SET state = ?, not_before = ?, last_error_category = ?, "
                        "last_error_detail = ?, updated_at = ? WHERE row_id = ?",
                        (STATE_RETRYABLE, now + decision.backoff_seconds, category,
                         decision.detail, now, row_id))
                released += 1
        return released

    def reprocess_rejected(self, *, policy: RetryPolicy, now: float | None = None) -> int:
        """Explicitly move rejected rows back to pending, preserving attempt history."""
        now = time.time() if now is None else now
        with self.transaction():
            cursor = self.conn.execute(
                "UPDATE rows SET state = ?, rejection_reason = NULL, not_before = 0, "
                "updated_at = ? WHERE state = ? AND COALESCE(last_error_category, '') "
                "!= 'input_invalid'", (STATE_PENDING, now, STATE_REJECTED))
            return cursor.rowcount

    # -- backup ------------------------------------------------------------

    def checkpoint(self, destination: Path | None = None) -> Path:
        directory = Path(destination) if destination else self.work / LEDGER_BACKUP_DIR
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        target = directory / f"ledger-{stamp}.sqlite3"
        suffix = 0
        while target.exists():
            suffix += 1
            target = directory / f"ledger-{stamp}-{suffix}.sqlite3"
        backup = sqlite3.connect(str(target))
        try:
            self.conn.backup(backup)
        finally:
            backup.close()
        return target


def ledger_identity_from_config(config: RunConfig, meta: dict) -> dict:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "run_id": config.run_id(),
        "identity_sha256": config.identity_sha256(),
        "manifest_sha256": meta["manifest_sha256"],
        "dataset_revision": config.dataset.revision,
        "model_revision": config.model.revision,
        "prompt_sha256": config.prompt.sha256,
    }


# ---------------------------------------------------------------------------
# Slurm guard
# ---------------------------------------------------------------------------


def require_slurm(args, *, gpus: int | None = None) -> None:
    """Refuse heavy stages outside a Slurm allocation (synthetic fixtures excepted)."""
    if args.allow_non_slurm:
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit(
            "Refusing to run outside a Slurm allocation. Generate a script with "
            "`build_dataset.py slurm-script` and submit it on Alex. "
            "(--allow-non-slurm exists for synthetic fixtures in tests only.)")
    if gpus is not None:
        visible = [part for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part]
        if len(visible) != gpus:
            raise SystemExit(
                f"Expected exactly {gpus} visible GPUs for this stage, got {visible!r}")


# ---------------------------------------------------------------------------
# Commands: plan
# ---------------------------------------------------------------------------


def cmd_plan(args) -> int:
    prompt = load_prompt(args.prompt_version)
    config = config_from_args(args, prompt=prompt)
    plan = {
        "tool_version": TOOL_VERSION,
        "work": str(Path(args.work).resolve()),
        "run_id_provisional": config.run_id(),
        "identity_sha256_provisional": config.identity_sha256(),
        "identity_provisional": config.identity(),
        "stages": ["prepare", "run", "status", "validate", "audit", "export", "checkpoint"],
        "source": {
            "dataset": config.dataset.dataset_id,
            "split": config.dataset.split,
            "rows": f"{config.dataset.row_start}-{config.dataset.row_start + config.dataset.row_limit - 1}",
            "count": config.dataset.row_limit,
        },
        "model": config.model.model_id,
        "baseline": {
            "hardware": "2 x NVIDIA RTX PRO 6000 Blackwell 96GB",
            "measured_successful_samples_per_hour": MEASURED_SAMPLES_PER_HOUR,
            "estimated_generation_hours_at_baseline": round(
                config.dataset.row_limit / MEASURED_SAMPLES_PER_HOUR, 2),
            "note": "Measured baseline is a throughput reference, not a promise; "
                    "startup, retries, validation and export add overhead.",
        },
        "notes": [
            "Revisions and the container hash resolve to real values during prepare/run.",
            "The run identity includes prompt, model, dataset, code, container and inference settings.",
            "Use a fresh --work directory to start a new run without contaminating an old one.",
        ],
    }
    print(json.dumps(plan, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Commands: checkpoint
# ---------------------------------------------------------------------------


def cmd_checkpoint(args) -> int:
    work = Path(args.work).resolve()
    with Ledger(work, read_only=True) as ledger:
        target = ledger.checkpoint()
    print(f"Ledger checkpoint: {target}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work", default="./tikz-production",
                        help="Work directory holding manifest, ledger and outputs")
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--dataset-revision", default="",
                        help="Pinned dataset commit sha (resolved from the Hub when omitted)")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default="",
                        help="Pinned model commit sha (resolved from the Hub when omitted)")
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--concurrency", type=int, default=64, help="Aggregate in-flight requests")
    parser.add_argument("--mtp", type=int, choices=(0, 1, 2, 3), default=1)
    parser.add_argument("--batch-token-budget", type=int, default=16_384)
    parser.add_argument("--context", type=int, default=32_768)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--truncation-retry-tokens", type=int, default=384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--nccl-p2p", choices=("disabled", "auto"), default="disabled")
    parser.add_argument("--min-instruction-chars", type=int, default=20)
    parser.add_argument("--max-instruction-chars", type=int, default=2000)
    parser.add_argument("--max-instruction-words", type=int, default=120)
    parser.add_argument("--allow-non-slurm", action="store_true", help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_dataset.py",
        description="Production dataset builder (see cleaning/PRODUCTION.md)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Print resolved configuration and identity")
    add_common_arguments(plan)
    plan.set_defaults(handler=cmd_plan)

    prepare = subparsers.add_parser(
        "prepare", help="Freeze exactly the first 100,000 source rows into a manifest")
    add_common_arguments(prepare)
    prepare.add_argument("--model-cache-dir", default="",
                         help="Hugging Face cache for the model snapshot (default: WORK/hf)")
    prepare.add_argument("--download-model", action=argparse.BooleanOptionalAction,
                         default=True, help="Download the pinned model snapshot during prepare")
    prepare.add_argument("--max-tikz-chars", type=int, default=200_000)
    prepare.add_argument("--max-image-bytes", type=int, default=20_000_000)
    prepare.add_argument("--estimate-rows", type=int, default=200,
                         help="Bounded sample used for the disk-space estimate")
    prepare.add_argument("--row-limit", type=int, default=ROW_LIMIT, help=argparse.SUPPRESS)
    prepare.set_defaults(handler=cmd_prepare)

    checkpoint = subparsers.add_parser("checkpoint", help="Copy the ledger to a backup file")
    add_common_arguments(checkpoint)
    checkpoint.set_defaults(handler=cmd_checkpoint)
    return parser


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Accept the historical `--slurm-script` spelling as a subcommand alias.
    if argv and argv[0] == "--slurm-script":
        argv[0] = "slurm-script"
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ConfigError, IdentityMismatch, PromptError, ManifestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
