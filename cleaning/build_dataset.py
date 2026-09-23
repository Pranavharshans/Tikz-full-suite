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
import signal
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
            "validation": dict(dataclasses.asdict(self.validation),
                               rules_sha256=validation_rules_sha256()),
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
# Instruction validation (policy v1)
# ---------------------------------------------------------------------------

# LaTeX/TikZ commands that betray copied source rather than a natural instruction.
TIKZ_COMMANDS = (
    "\\begin{", "\\end{", "\\draw", "\\node", "\\path", "\\fill", "\\coordinate",
    "\\foreach", "\\documentclass", "\\usepackage", "\\usetikzlibrary", "\\tikz",
    "\\pgf",
)
# Lowercase phrases that reference the task inputs instead of describing the diagram.
FORBIDDEN_REFERENCES = (
    "supplied image", "supplied source", "supplied input",
    "provided input", "provided image", "provided source",
    "source code", "input image", "reference image",
    "this task", "the task above", "markdown fence",
)


def validation_rules_sha256() -> str:
    """Hash of the rule tables so editing them cannot silently keep an old identity."""
    return canonical_digest(dict(
        version=VALIDATION_POLICY_VERSION, tikz_commands=TIKZ_COMMANDS,
        forbidden_references=FORBIDDEN_REFERENCES))


def validate_instruction(text, policy: ValidationPolicy) -> tuple[bool, str, str]:
    """Validate one accepted instruction.  Returns (ok, rule, detail)."""
    if not isinstance(text, str):
        return False, "not_text", f"instruction is {type(text).__name__}"
    stripped = text.strip()
    if not stripped:
        return False, "empty", "instruction is empty after stripping"
    try:
        stripped.encode("utf-8")
    except UnicodeEncodeError as exc:
        return False, "invalid_utf8", str(exc)
    if "\x00" in stripped:
        return False, "control_characters", "instruction contains a NUL byte"
    lowered = stripped.lower()
    if "<think" in lowered or "</think" in lowered:
        return False, "thinking_leak", "instruction contains thinking tags"
    if "<source" in lowered or "</source" in lowered:
        return False, "source_tags", "instruction contains source tags"
    if "```" in stripped:
        return False, "markdown_fence", "instruction contains a code fence"
    for command in TIKZ_COMMANDS:
        if command in stripped:
            return False, "tikz_code", f"instruction contains {command!r}"
    for phrase in FORBIDDEN_REFERENCES:
        if phrase in lowered:
            return False, "task_reference", f"instruction references {phrase!r}"
    if len(stripped) < policy.min_chars:
        return False, "too_short", f"{len(stripped)} chars < {policy.min_chars}"
    if len(stripped) > policy.max_chars:
        return False, "too_long", f"{len(stripped)} chars > {policy.max_chars}"
    words = len(stripped.split())
    if words > policy.max_words:
        return False, "too_many_words", f"{words} words > {policy.max_words}"
    return True, "ok", ""


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
    if category in ("integrity", "input_too_long", "invalid_instruction"):
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

    def next_retry_time(self) -> float | None:
        row = self.conn.execute(
            "SELECT MIN(not_before) AS t FROM rows WHERE state = ?", (STATE_RETRYABLE,)).fetchone()
        return row["t"] if row and row["t"] is not None else None

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
        with self.transaction():
            return self._record_result_locked(result, policy, time.time() if now is None else now)

    def record_results(self, results, *, policy: RetryPolicy,
                       now: float | None = None) -> list[str]:
        """Commit a batch of results in one transaction."""
        moment = time.time() if now is None else now
        with self.transaction():
            return [self._record_result_locked(result, policy, moment) for result in results]

    def _record_result_locked(self, result: dict, policy: RetryPolicy, now: float) -> str:
        row_id = result["row_id"]
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
# Inference worker (runs inside the engine container, one per GPU)
# ---------------------------------------------------------------------------


class WorkerError(RuntimeError):
    """Raised when the engine cannot be trusted to associate outputs with rows."""


@dataclass
class EngineOutput:
    prompt: str | None
    text: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None


def build_messages(prompt_text: str, task: dict) -> list:
    """System prompt plus the image and the TikZ source wrapped as data."""
    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": [
            {"type": "image", "image": task["image"]},
            {"type": "text", "text": "<source>\n" + task["tikz_code"] + "\n</source>"},
        ]},
    ]


class ProcessorPromptBuilder:
    """Real prompt builder: chat template plus multimodal token accounting."""

    def __init__(self, job: dict):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(job["config"]["model_path"])

    def __call__(self, task: dict, config: dict):
        from PIL import Image
        prompt = self.processor.apply_chat_template(
            build_messages(config["prompt"], task), tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        with Image.open(task["image"]) as image:
            pixels = image.convert("RGB")
            encoded = self.processor(text=[prompt], images=[pixels], return_tensors="pt")
        return prompt, pixels, int(encoded["input_ids"].shape[-1])


class VllmOfflineEngine:
    """Real engine: vLLM offline, TP1, greedy, MTP, prefix caching disabled."""

    def __init__(self, job: dict):
        from vllm import LLM
        config = job["config"]
        options = dict(model=config["model_path"], tensor_parallel_size=1,
                       max_model_len=config["context"],
                       max_num_seqs=config["per_replica_concurrency"],
                       max_num_batched_tokens=config["batch_token_budget"],
                       enable_chunked_prefill=True, enable_prefix_caching=False,
                       mm_processor_cache_gb=0,
                       gpu_memory_utilization=config["gpu_memory_utilization"])
        options.update(bench.vllm_runtime_options())
        if config["mtp"]:
            options["speculative_config"] = dict(method="mtp",
                                                 num_speculative_tokens=config["mtp"])
        bench.dump(Path(job["engine_options_path"]), options)
        self.llm = LLM(**options)
        import vllm
        bench.dump(Path(job["versions_path"]),
                   dict(python=sys.version, engine=vllm.__version__))

    def generate(self, items: list) -> list[EngineOutput]:
        from vllm import SamplingParams
        inputs = [dict(prompt=item["prompt"],
                       multi_modal_data={"image": item["image"]}) for item in items]
        params = [SamplingParams(max_tokens=item["max_tokens"], temperature=0.0)
                  for item in items]
        outputs = self.llm.generate(inputs, params, use_tqdm=True)
        results = []
        for output in outputs:
            completion = output.outputs[0]
            results.append(EngineOutput(
                prompt=getattr(output, "prompt", None), text=completion.text,
                finish_reason=completion.finish_reason,
                prompt_tokens=len(output.prompt_token_ids) if output.prompt_token_ids else None,
                completion_tokens=len(completion.token_ids)))
        return results


def write_json_atomic(path, value) -> None:
    bench.dump(path, value)


def write_text_atomic(path, text: str) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def worker_base_result(task: dict, config: dict, worker: str, batch_index: int) -> dict:
    return dict(row_id=task["row_id"], source_row_index=task["source_row_index"],
                worker=worker, batch_index=batch_index, max_tokens=task["max_tokens"],
                ok=False, instruction=None, text=None, finish_reason=None,
                prompt_tokens=None, completion_tokens=None,
                error_category=None, error_detail=None,
                image_sha256=task["image_sha256"], tikz_sha256=task["tikz_sha256"],
                model_revision=config["model_revision"],
                prompt_sha256=config["prompt_sha256"], config_hash=config["config_hash"])


def error_result(task: dict, config: dict, worker: str, batch_index: int,
                 category: str, detail: str) -> dict:
    result = worker_base_result(task, config, worker, batch_index)
    result.update(error_category=category, error_detail=detail)
    return result


def prepare_task(task: dict, builder, config: dict, worker: str, batch_index: int):
    """Verify source identity, then build the prompt.  Returns a result dict on failure."""
    try:
        raw = Path(task["image"]).read_bytes()
    except OSError as exc:
        return error_result(task, config, worker, batch_index, "io",
                            f"image read failed: {exc!r}")
    if sha256_bytes(raw) != task["image_sha256"]:
        return error_result(task, config, worker, batch_index, "integrity",
                            "image checksum mismatch")
    if sha256_text(task["tikz_code"]) != task["tikz_sha256"]:
        return error_result(task, config, worker, batch_index, "integrity",
                            "tikz checksum mismatch")
    try:
        prompt, image, length = builder(task, config)
    except Exception as exc:
        return error_result(task, config, worker, batch_index, "decode",
                            f"prompt build failed: {exc!r}")
    if length >= config["context"]:
        return error_result(task, config, worker, batch_index, "input_too_long",
                            f"input {length} tokens >= context {config['context']}")
    return dict(task=task, prompt=prompt, image=image, length=length,
                max_tokens=task["max_tokens"])


def generated_result(item: dict, output: EngineOutput, config: dict, worker: str,
                     batch_index: int) -> dict:
    task = item["task"]
    text = output.text or ""
    ok = output.finish_reason == "stop" and bool(text.strip())
    result = worker_base_result(task, config, worker, batch_index)
    result.update(ok=ok, instruction=text.strip() if ok else None, text=text,
                  finish_reason=output.finish_reason,
                  prompt_tokens=output.prompt_tokens,
                  completion_tokens=output.completion_tokens)
    return result


def validate_outputs(outputs: list, items: list) -> None:
    """Refuse to trust engine outputs whose association with inputs is unclear."""
    if len(outputs) != len(items):
        raise WorkerError(f"engine returned {len(outputs)} outputs for {len(items)} inputs")
    for item, output in zip(items, outputs):
        if output.prompt is not None and output.prompt != item["prompt"]:
            raise WorkerError(
                f"engine prompt for {item['task']['row_id']} does not match the request")


def write_heartbeat(job: dict, batch_index: int, rows_done: int) -> None:
    write_json_atomic(job["heartbeat_path"], dict(
        worker=job["worker_index"], generation=job["generation"],
        batch_index=batch_index, rows_done=rows_done, updated_at=time.time()))


def process_batch(*, job: dict, batch: list, batch_index: int, engine, builder,
                  results_dir: Path, rows_done: int):
    """Run one microbatch.  Returns (rows_done, engine_error)."""
    config = job["config"]
    worker = f"worker-{job['worker_index']}"
    results, items = [], []
    for task in batch:
        prepared = prepare_task(task, builder, config, worker, batch_index)
        if isinstance(prepared, dict) and "task" not in prepared:
            results.append(prepared)
        else:
            items.append(prepared)
    engine_error = None
    if items:
        try:
            outputs = engine.generate([dict(prompt=item["prompt"], image=item["image"],
                                            max_tokens=item["max_tokens"],
                                            row_id=item["task"]["row_id"]) for item in items])
            validate_outputs(outputs, items)
        except Exception as exc:
            engine_error = exc
            for item in items:
                results.append(error_result(item["task"], config, worker, batch_index,
                                            "engine_transient", f"generate failed: {exc!r}"))
        else:
            for item, output in zip(items, outputs):
                results.append(generated_result(item, output, config, worker, batch_index))
    payload = dict(run_id=job["run_id"], worker=worker, worker_index=job["worker_index"],
                   generation=job["generation"], batch_index=batch_index, results=results)
    write_json_atomic(results_dir / f"batch-{job['generation']}-{batch_index:05d}.json", payload)
    rows_done += len(results)
    write_heartbeat(job, batch_index, rows_done)
    return rows_done, engine_error


def read_first_task(tasks_path) -> dict | None:
    with Path(tasks_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    return None


def run_worker(job_path, *, engine_factory=None, prompt_builder_factory=None) -> int:
    """Worker entry point.  Writes one atomic result file per microbatch."""
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    config = job["config"]
    worker = f"worker-{job['worker_index']}"
    results_dir = Path(job["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    stop_path = Path(job["stop_path"])
    tasks_path = Path(job["tasks_path"])

    def log(message):
        print(f"[{worker} gen{job['generation']}] {message}", flush=True)

    try:
        builder = (prompt_builder_factory or ProcessorPromptBuilder)(job)
        engine = (engine_factory or VllmOfflineEngine)(job)
    except Exception as exc:
        log(f"engine startup failed: {exc!r}")
        return 3
    first = read_first_task(tasks_path)
    if first is not None and config.get("warmup_samples", 1) > 0:
        try:
            prepared = prepare_task(first, builder, config, worker, -1)
            if "task" in prepared:
                outputs = engine.generate([dict(prompt=prepared["prompt"],
                                                image=prepared["image"],
                                                max_tokens=prepared["max_tokens"],
                                                row_id=prepared["task"]["row_id"])])
                if not outputs or not (outputs[0].text or "").strip():
                    log("warmup produced no output")
                    return 3
                log("warmup complete")
            else:
                log(f"warmup row skipped: {prepared.get('error_detail')}")
        except Exception as exc:
            log(f"warmup failed: {exc!r}")
            return 3
    batch_index = 0
    rows_done = 0
    engine_failed = False
    try:
        with tasks_path.open("r", encoding="utf-8") as handle:
            batch = []
            for line in handle:
                if not line.strip():
                    continue
                batch.append(json.loads(line))
                if len(batch) >= config["per_replica_concurrency"]:
                    if stop_path.exists():
                        log("stop requested; exiting before next batch")
                        return 0
                    rows_done, engine_error = process_batch(
                        job=job, batch=batch, batch_index=batch_index, engine=engine,
                        builder=builder, results_dir=results_dir, rows_done=rows_done)
                    batch_index += 1
                    if engine_error is not None:
                        log(f"engine failure; exiting for restart: {engine_error!r}")
                        engine_failed = True
                        break
                    batch = []
            if batch and not engine_failed and not stop_path.exists():
                rows_done, engine_error = process_batch(
                    job=job, batch=batch, batch_index=batch_index, engine=engine,
                    builder=builder, results_dir=results_dir, rows_done=rows_done)
                if engine_error is not None:
                    log(f"engine failure; exiting for restart: {engine_error!r}")
                    engine_failed = True
    except Exception as exc:
        log(f"worker crashed: {exc!r}")
        return 3
    log(f"finished {rows_done} rows")
    return 3 if engine_failed else 0


# ---------------------------------------------------------------------------
# Inference controller
# ---------------------------------------------------------------------------


class PipelineIdentityError(RuntimeError):
    """Raised when a result cannot be trusted to belong to this run and row."""


class StopFlag:
    """Cooperative SIGTERM/SIGINT handling for the controller."""

    def __init__(self):
        self.requested = False
        self.signal_name = None

    def handle(self, signum, frame):
        self.requested = True
        self.signal_name = signal.Signals(signum).name

    class _Installer:
        def __init__(self, flag):
            self.flag = flag
            self.previous = {}

        def __enter__(self):
            for sig in (signal.SIGTERM, signal.SIGINT):
                self.previous[sig] = signal.getsignal(sig)
                signal.signal(sig, self.flag.handle)
            return self.flag

        def __exit__(self, *exc_info):
            for sig, handler in self.previous.items():
                signal.signal(sig, handler)
            return False

    def installed(self):
        return StopFlag._Installer(self)


class SubprocessHandle:
    """A running engine worker process."""

    def __init__(self, process, log_handle):
        self.process = process
        self.log = log_handle

    def poll(self):
        return self.process.poll()

    @property
    def returncode(self):
        return self.process.returncode

    def terminate(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)

    def kill(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)

    def wait(self, timeout=None):
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def close(self):
        self.log.close()


def container_sha256(path, cache_path=None) -> str:
    """Hash the immutable SIF bytes, cached by path/size/mtime."""
    path = Path(path).resolve()
    stat = path.stat()
    key = f"{path}|{stat.st_size}|{int(stat.st_mtime)}"
    if cache_path is not None and Path(cache_path).is_file():
        cached = json.loads(Path(cache_path).read_text())
        if cached.get("key") == key:
            return cached["sha256"]
    digest = sha256_file(path)
    if cache_path is not None:
        bench.dump(cache_path, dict(key=key, sha256=digest))
    return digest


class PipelineDeps:
    """Injectable seams for the controller (tests replace spawner and probes)."""

    def container_sha256(self, path, work) -> str:
        return container_sha256(path, Path(work) / "runtime" / "container-sha256.json")

    def probe_gpus(self) -> list[str]:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise SystemExit(f"nvidia-smi failed: {result.stderr.strip()}")
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def spawn_worker(self, *, job_path, worker_index, device, args, work):
        env = dict(os.environ)
        env["APPTAINERENV_CUDA_VISIBLE_DEVICES"] = device
        env["APPTAINERENV_NCCL_P2P_DISABLE"] = "1" if args.nccl_p2p == "disabled" else "0"
        env["APPTAINERENV_no_proxy"] = "127.0.0.1,localhost,::1"
        env["APPTAINERENV_NO_PROXY"] = env["APPTAINERENV_no_proxy"]
        env["APPTAINERENV_HF_HUB_OFFLINE"] = "1"
        env["APPTAINERENV_TRANSFORMERS_OFFLINE"] = "1"
        binds = {str(Path(work).resolve()), str(HERE)}
        model_path = Path(args.model_path).resolve()
        binds.add(str(model_path))
        command = ["apptainer", "exec", "--nv"]
        for path in sorted(binds):
            command += ["--bind", path]
        command += [str(Path(args.vllm_sif).resolve()), "python3",
                    str(Path(__file__).resolve()), "worker", "--job", str(job_path)]
        log_path = Path(work) / "runtime" / "logs" / f"worker-{worker_index}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w")
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        return SubprocessHandle(process, log)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def balanced_assignment(eligible, index, workers: int) -> dict:
    """Deterministic length-balanced assignment; per-worker rows stay index-ordered."""
    ordered = sorted(eligible, key=lambda row: (-index[row["row_id"]]["tikz_chars"],
                                                row["source_row_index"]))
    loads = [0] * workers
    buckets: list[list] = [[] for _ in range(workers)]
    for row in ordered:
        target = min(range(workers), key=lambda worker: (loads[worker], worker))
        buckets[target].append(row)
        loads[target] += index[row["row_id"]]["tikz_chars"] + 1024
    for bucket in buckets:
        bucket.sort(key=lambda row: row["source_row_index"])
    return {f"worker-{worker}": [row["row_id"] for row in buckets[worker]]
            for worker in range(workers)}


def write_task_files(work, claims: dict, generation: int) -> dict:
    """Stream the manifest once and write one task file per worker."""
    work = Path(work)
    runtime = work / "runtime" / "tasks"
    runtime.mkdir(parents=True, exist_ok=True)
    worker_of = {}
    claim_of = {}
    for worker, rows in claims.items():
        for row in rows:
            worker_of[row["row_id"]] = worker
            claim_of[row["row_id"]] = row
    paths, handles = {}, {}
    for worker in claims:
        paths[worker] = runtime / f"{worker}-gen{generation}.jsonl"
        handles[worker] = paths[worker].with_suffix(".jsonl.tmp").open("w", encoding="utf-8")
    try:
        for entry in iter_manifest(work):
            row_id = entry["row_id"]
            worker = worker_of.get(row_id)
            if worker is None:
                continue
            claim = claim_of[row_id]
            task = dict(row_id=row_id, source_row_index=entry["source_row_index"],
                        image=str(work / entry["image"]),
                        image_sha256=entry["image_sha256"], tikz_code=entry["tikz_code"],
                        tikz_sha256=entry["tikz_sha256"], max_tokens=claim["max_tokens"],
                        attempt_no=claim["attempt_no"])
            handles[worker].write(json.dumps(task, ensure_ascii=False) + "\n")
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        for handle in handles.values():
            handle.close()
    for worker, path in paths.items():
        os.replace(path.with_suffix(".jsonl.tmp"), path)
    return paths


def write_worker_job(path, *, args, config, work, worker_index: int, generation: int,
                     tasks_path) -> Path:
    work = Path(work)
    job = dict(
        run_id=config.run_id(), worker_index=worker_index, generation=generation,
        tasks_path=str(tasks_path),
        results_dir=str(work / "runtime" / "results" / f"worker-{worker_index}"),
        heartbeat_path=str(work / "runtime" / "heartbeats" / f"worker-{worker_index}.json"),
        stop_path=str(work / "runtime" / "stop"),
        engine_options_path=str(work / "runtime" / "logs" / f"worker-{worker_index}-engine-options.json"),
        versions_path=str(work / "runtime" / "logs" / f"worker-{worker_index}-versions.json"),
        config=dict(
            model_path=config.model.path, model_revision=config.model.revision,
            prompt=config.prompt.text, prompt_version=config.prompt.version,
            prompt_sha256=config.prompt.sha256, context=config.inference.context,
            per_replica_concurrency=config.inference.per_replica_concurrency(worker_index),
            batch_token_budget=config.inference.batch_token_budget, mtp=config.inference.mtp,
            gpu_memory_utilization=config.inference.gpu_memory_utilization,
            max_output_tokens=config.inference.max_output_tokens,
            truncation_retry_tokens=config.inference.truncation_retry_tokens,
            warmup_samples=args.warmup_samples, config_hash=config.identity_sha256()))
    bench.dump(path, job)
    return Path(path)


def validate_result_identity(result: dict, config, index: dict) -> str | None:
    """Fatal identity checks raise; source checksum drift returns an integrity reason."""
    row_id = result.get("row_id")
    meta_row = index.get(row_id)
    if meta_row is None:
        raise PipelineIdentityError(f"Result references unknown row {row_id!r}")
    if result.get("source_row_index") != meta_row["source_row_index"]:
        raise PipelineIdentityError(f"Result source index mismatch for row {row_id}")
    if result.get("config_hash") != config.identity_sha256():
        raise PipelineIdentityError(f"Result config hash mismatch for row {row_id}")
    if result.get("prompt_sha256") != config.prompt.sha256:
        raise PipelineIdentityError(f"Result prompt hash mismatch for row {row_id}")
    if result.get("model_revision") != config.model.revision:
        raise PipelineIdentityError(f"Result model revision mismatch for row {row_id}")
    if result.get("image_sha256") != meta_row["image_sha256"]:
        return "image checksum mismatch"
    if result.get("tikz_sha256") != meta_row["tikz_sha256"]:
        return "tikz checksum mismatch"
    return None


def ingest_result_files(*, ledger, work, config, index, policy) -> dict:
    """Commit every unconsumed result file, then move it to consumed/."""
    work = Path(work)
    results_root = work / "runtime" / "results"
    consumed_root = work / "runtime" / "consumed"
    seen: dict[str, set] = {}
    if not results_root.is_dir():
        return seen
    for path in sorted(results_root.glob("worker-*/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("run_id") != config.run_id():
            raise PipelineIdentityError(
                f"Result file {path} belongs to run {data.get('run_id')!r}, not {config.run_id()!r}")
        worker = data.get("worker")
        if worker != f"worker-{data.get('worker_index')}":
            raise PipelineIdentityError(f"Result file {path} has an inconsistent worker label")
        results, row_ids = [], []
        for result in data.get("results", []):
            if result.get("worker") != worker:
                raise PipelineIdentityError(
                    f"Result for {result.get('row_id')} claims worker {result.get('worker')!r} "
                    f"but arrived in {worker}'s file")
            if result["row_id"] in row_ids:
                raise PipelineIdentityError(
                    f"Duplicate result for {result['row_id']} inside {path}")
            reason = validate_result_identity(result, config, index)
            if reason is not None:
                result = dict(result, ok=False, instruction=None, finish_reason=None,
                              error_category="integrity", error_detail=reason)
            elif result.get("ok"):
                # Semantic validation is authoritative here and never retried:
                # greedy decoding would reproduce the same invalid text.
                valid, rule, detail = validate_instruction(result.get("instruction"),
                                                           config.validation)
                if not valid:
                    result = dict(result, ok=False, instruction=None,
                                  error_category="invalid_instruction",
                                  error_detail=f"{rule}: {detail}" if detail else rule)
            row = ledger.get(result["row_id"])
            if row is None:
                raise PipelineIdentityError(f"Result for unknown row {result['row_id']}")
            if row["state"] == STATE_RUNNING and row["worker"] != worker:
                raise PipelineIdentityError(
                    f"Row {result['row_id']} is claimed by {row['worker']!r} but a result "
                    f"arrived from {worker!r}")
            results.append(result)
            row_ids.append(result["row_id"])
        ledger.record_results(results, policy=policy)
        destination = consumed_root / worker / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)
        seen.setdefault(worker, set()).update(row_ids)
    return seen


def filter_task_file(source, destination, keep_row_ids) -> Path:
    """Rewrite an existing task file keeping only the rows still outstanding."""
    source, destination = Path(source), Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("r", encoding="utf-8") as reader, temporary.open("w", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            task = json.loads(line)
            if task["row_id"] in keep_row_ids:
                writer.write(json.dumps(task, ensure_ascii=False) + "\n")
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, destination)
    return destination


def _worker_last_activity(handle, heartbeat_path) -> float:
    latest = getattr(handle, "started_at", 0.0)
    try:
        latest = max(latest, Path(heartbeat_path).stat().st_mtime)
    except OSError:
        pass
    return latest


def run_wave(*, args, config, work, deps, ledger, index, policy, wave, stop_flag,
             deadline, claimed_so_far: int) -> int:
    """Claim rows, run both replicas to completion, reclaim what they left.

    Returns the number of rows claimed in this wave.
    """
    work = Path(work)
    remaining = None
    if args.max_rows_this_run:
        remaining = args.max_rows_this_run - claimed_so_far
        if remaining <= 0:
            return 0
    eligible = ledger.eligible(start_index=args.start_index, end_index=args.end_index,
                               limit=remaining)
    if not eligible:
        return 0
    assignments = balanced_assignment(eligible, index, workers=config.inference.replicas)
    claims = ledger.claim(assignments, policy=policy)
    claimed_count = sum(len(rows) for rows in claims.values())
    print(f"wave {wave}: claimed {claimed_count} rows", flush=True)
    task_paths = write_task_files(work, claims, wave)
    handles, outstanding, restarts, stall_deadline = {}, {}, {}, {}
    devices = [part for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part]
    worker_indexes = sorted(int(worker.split("-")[1]) for worker in claims)

    def start(worker_index):
        worker = f"worker-{worker_index}"
        if restarts[worker_index] == 0:
            tasks_for_worker = task_paths[worker]
        else:
            tasks_for_worker = work / "runtime" / "tasks" / (
                f"{worker}-gen{wave}-r{restarts[worker_index]}.jsonl")
            filter_task_file(task_paths[worker], tasks_for_worker, outstanding[worker_index])
        job_path = work / "runtime" / "jobs" / f"{worker}-gen{wave}-r{restarts[worker_index]}.json"
        job_path.parent.mkdir(parents=True, exist_ok=True)
        write_worker_job(job_path, args=args, config=config, work=work,
                         worker_index=worker_index, generation=wave,
                         tasks_path=tasks_for_worker)
        handle = deps.spawn_worker(job_path=job_path, worker_index=worker_index,
                                   device=devices[worker_index], args=args, work=work)
        handle.started_at = time.time()
        handles[worker_index] = handle

    for worker_index in worker_indexes:
        worker = f"worker-{worker_index}"
        outstanding[worker_index] = {row["row_id"] for row in claims[worker]}
        restarts[worker_index] = 0
        start(worker_index)

    def ingest():
        ingested = ingest_result_files(ledger=ledger, work=work, config=config,
                                       index=index, policy=policy)
        for worker_index in handles:
            worker = f"worker-{worker_index}"
            if worker in ingested:
                outstanding[worker_index] -= ingested[worker]
        return ingested

    def release_outstanding(category):
        for worker_index, rows in outstanding.items():
            if rows:
                ledger.release_running(worker=f"worker-{worker_index}",
                                       category=category, policy=policy)
                outstanding[worker_index] = set()

    try:
        while True:
            ingest()
            if stop_flag.requested:
                _request_stop(work)
                _drain_workers(args, handles)
                ingest()
                release_outstanding("run_interrupted")
                return claimed_count
            if deadline is not None and time.time() > deadline:
                _request_stop(work)
                print("Runtime budget reached; stopping workers", flush=True)
                _drain_workers(args, handles)
                ingest()
                release_outstanding("run_interrupted")
                return claimed_count
            now = time.time()
            for worker_index, handle in list(handles.items()):
                worker = f"worker-{worker_index}"
                heartbeat = work / "runtime" / "heartbeats" / f"{worker}.json"
                if handle.poll() is None:
                    last = _worker_last_activity(handle, heartbeat)
                    if worker_index not in stall_deadline and now - last > args.worker_timeout:
                        print(f"{worker} stalled for {args.worker_timeout}s; terminating",
                              flush=True)
                        handle.terminate()
                        stall_deadline[worker_index] = now + 60
                    elif worker_index in stall_deadline and now > stall_deadline[worker_index]:
                        print(f"{worker} did not exit; killing", flush=True)
                        handle.kill()
                    continue
                # Worker exited: sweep its final results, then restart or release.
                ingest()
                if outstanding[worker_index]:
                    if restarts[worker_index] >= args.worker_restarts:
                        print(f"{worker} exited rc={handle.returncode} with "
                              f"{len(outstanding[worker_index])} rows left; releasing",
                              flush=True)
                        ledger.release_running(worker=worker, category="worker_lost",
                                               policy=policy)
                        outstanding[worker_index] = set()
                    else:
                        restarts[worker_index] += 1
                        print(f"{worker} exited rc={handle.returncode}; restart "
                              f"{restarts[worker_index]}/{args.worker_restarts}", flush=True)
                        handle.close()
                        start(worker_index)
            if all(handle.poll() is not None for handle in handles.values()) and \
                    not any(outstanding.values()):
                return claimed_count
            deps.sleep(args.poll_seconds)
    finally:
        for handle in handles.values():
            handle.close()


def _request_stop(work) -> None:
    stop = Path(work) / "runtime" / "stop"
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.touch()


def _drain_workers(args, handles) -> None:
    deadline = time.time() + args.shutdown_grace_seconds
    while time.time() < deadline:
        if all(handle.poll() is not None for handle in handles.values()):
            return
        time.sleep(1)
    for handle in handles.values():
        if handle.poll() is None:
            handle.terminate()
    deadline = time.time() + 30
    while time.time() < deadline:
        if all(handle.poll() is not None for handle in handles.values()):
            return
        time.sleep(1)
    for handle in handles.values():
        if handle.poll() is None:
            handle.kill()


def run_pipeline(args, config, meta, deps, stop_flag=None) -> dict:
    """Wave-based controller: claim, run, ingest, reclaim, repeat."""
    work = Path(args.work).resolve()
    stop_flag = stop_flag or StopFlag()
    policy = RetryPolicy(max_transient_attempts=args.max_transient_attempts,
                         backoff_base_seconds=args.retry_backoff_base_seconds,
                         backoff_cap_seconds=args.retry_backoff_cap_seconds,
                         normal_max_tokens=config.inference.max_output_tokens,
                         truncation_max_tokens=config.inference.truncation_retry_tokens)
    index = manifest_index(work)
    started = time.monotonic()
    deadline = started + args.max_runtime_minutes * 60 if args.max_runtime_minutes else None
    ledger = Ledger(work).open()
    claimed_so_far = 0
    try:
        # A leftover stop marker from an interrupted run must not stop this one.
        stop_path = work / "runtime" / "stop"
        if stop_path.exists():
            stop_path.unlink()
        ledger.initialize(ledger_identity_from_config(config, meta))
        seeded = ledger.seed(iter_manifest(work), rows_frozen=meta["rows_frozen"])
        if seeded["inserted"]:
            print(f"Ledger seeded: {seeded['inserted']} rows "
                  f"({seeded['rejected']} manifest-rejected)", flush=True)
        resumed = ingest_result_files(ledger=ledger, work=work, config=config, index=index,
                                      policy=policy)
        if resumed:
            print(f"Recovered results from a previous run: "
                  f"{sum(len(rows) for rows in resumed.values())} rows", flush=True)
        reclaimed = ledger.reclaim_stale(args.stale_claim_seconds, policy=policy)
        if reclaimed:
            print(f"Reclaimed {reclaimed} stale running rows", flush=True)
        if args.reprocess_rejected:
            moved = ledger.reprocess_rejected(policy=policy)
            print(f"Reprocessing {moved} rejected rows at the operator's request", flush=True)
        wave = 0
        while True:
            if stop_flag.requested:
                print(f"Stop requested ({stop_flag.signal_name}); no new claims", flush=True)
                break
            if deadline is not None and time.monotonic() > deadline:
                print("Runtime budget reached; no new claims", flush=True)
                break
            if args.max_rows_this_run and claimed_so_far >= args.max_rows_this_run:
                print("Row budget reached; no new claims", flush=True)
                break
            if not ledger.eligible(start_index=args.start_index, end_index=args.end_index,
                                   limit=1):
                next_retry = ledger.next_retry_time()
                now = time.time()
                if (next_retry is not None and next_retry - now <= args.retry_wait_seconds
                        and (deadline is None or now + (next_retry - now) < deadline)):
                    wait = max(0.0, next_retry - now)
                    print(f"Waiting {wait:.0f}s for retry backoff", flush=True)
                    deps.sleep(wait)
                    continue
                break
            claimed_so_far += run_wave(
                args=args, config=config, work=work, deps=deps, ledger=ledger, index=index,
                policy=policy, wave=wave + 1, stop_flag=stop_flag, deadline=deadline,
                claimed_so_far=claimed_so_far)
            wave += 1
        counts = ledger.counts()
        states = counts["states"]
        remaining = states["pending"] + states["retryable"] + states["running"]
        elapsed = time.monotonic() - started
        print(f"Run finished in {elapsed / 60:.1f} min: {states}", flush=True)
        if counts["last_complete_at"] and counts["first_complete_at"]:
            active = max(1.0, counts["last_complete_at"] - counts["first_complete_at"])
            print(f"Throughput: {states['complete'] / active * 3600:.1f} successful samples/hour "
                  f"(generation window)", flush=True)
        checkpoint = ledger.checkpoint()
        print(f"Ledger checkpoint: {checkpoint}", flush=True)
        return dict(states=states, remaining=remaining, elapsed_s=elapsed,
                    exit_code=0 if remaining == 0 else 3)
    finally:
        ledger.close()


def validate_run_args(args) -> None:
    if not 0 <= args.start_index <= args.end_index <= ROW_LIMIT - 1:
        raise ConfigError(
            f"Chunk bounds must satisfy 0 <= start <= end <= {ROW_LIMIT - 1}; got "
            f"{args.start_index}..{args.end_index}")
    if args.max_rows_this_run is not None and args.max_rows_this_run < 1:
        raise ConfigError("--max-rows-this-run must be positive")
    if args.max_runtime_minutes is not None and args.max_runtime_minutes <= 0:
        raise ConfigError("--max-runtime-minutes must be positive")
    if args.worker_restarts < 0:
        raise ConfigError("--worker-restarts cannot be negative")
    if args.warmup_samples < 0:
        raise ConfigError("--warmup-samples cannot be negative")
    for name in ("stale_claim_seconds", "worker_timeout", "poll_seconds"):
        if getattr(args, name) <= 0:
            raise ConfigError(f"--{name.replace('_', '-')} must be positive")
    for name in ("shutdown_grace_seconds", "retry_wait_seconds"):
        if getattr(args, name) < 0:
            raise ConfigError(f"--{name.replace('_', '-')} cannot be negative")
    if args.max_transient_attempts < 1:
        raise ConfigError("--max-transient-attempts must be positive")
    for name in ("retry_backoff_base_seconds", "retry_backoff_cap_seconds"):
        if getattr(args, name) < 0:
            raise ConfigError(f"--{name.replace('_', '-')} cannot be negative")


def cmd_run(args, deps=None) -> int:
    require_slurm(args, gpus=2)
    validate_run_args(args)
    deps = deps or PipelineDeps()
    work = Path(args.work).resolve()
    meta = verify_manifest(work, quick=True)
    if meta.get("row_limit") != ROW_LIMIT and not args.allow_non_slurm:
        raise ConfigError(
            f"Manifest freezes {meta.get('row_limit')} rows; production requires {ROW_LIMIT}")
    if not args.vllm_sif:
        raise ConfigError("run requires --vllm-sif (the engine container)")
    container = deps.container_sha256(args.vllm_sif, work)
    config = config_from_args(args, manifest_meta=meta, container_sha256=container,
                              require_pinned=True)
    if not config.model.path:
        raise ConfigError(
            "The frozen manifest has no model snapshot path; rerun prepare with "
            "--download-model (or place the snapshot and re-run prepare)")
    args.model_path = config.model.path
    if run_record_path(work).is_file():
        verify_run_record(work, config)
    else:
        record = write_run_record(work, config)
        print(f"Run identity recorded: {record['run_id']}", flush=True)
    gpus = deps.probe_gpus()
    if len(gpus) != 2:
        raise SystemExit(f"Expected exactly 2 visible GPUs, found {len(gpus)}: {gpus}")
    if not all("RTX PRO 6000" in name for name in gpus):
        raise SystemExit(f"Expected RTX PRO 6000 GPUs for the production baseline, found {gpus}")
    stop_flag = StopFlag()
    with stop_flag.installed():
        summary = run_pipeline(args, config, meta, deps, stop_flag)
    if summary["exit_code"]:
        print(f"{summary['remaining']} rows still need work; rerun to resume", flush=True)
    return summary["exit_code"]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

EXPORT_SCHEMA_VERSION = "export-v1"


class ExportError(RuntimeError):
    """Raised when the export cannot guarantee a consistent, complete dataset."""


def require_pyarrow():
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
        return pyarrow
    except ImportError as exc:
        raise ExportError(
            "pyarrow is required for export. Run export inside the vLLM container "
            "(the image provides pyarrow through datasets) or install pyarrow.") from exc


def dataset_arrow_schema():
    pyarrow = require_pyarrow()
    return pyarrow.schema([
        pyarrow.field("id", pyarrow.string(), nullable=False),
        pyarrow.field("source_row_index", pyarrow.int32(), nullable=False),
        pyarrow.field("file_id", pyarrow.string()),
        pyarrow.field("png_image", pyarrow.binary(), nullable=False),
        pyarrow.field("tikz_code", pyarrow.string(), nullable=False),
        pyarrow.field("instruction", pyarrow.string(), nullable=False),
        pyarrow.field("source_dataset", pyarrow.string(), nullable=False),
        pyarrow.field("source_revision", pyarrow.string(), nullable=False),
        pyarrow.field("caption_model", pyarrow.string(), nullable=False),
        pyarrow.field("caption_model_revision", pyarrow.string(), nullable=False),
        pyarrow.field("prompt_version", pyarrow.string(), nullable=False),
        pyarrow.field("image_sha256", pyarrow.string(), nullable=False),
        pyarrow.field("tikz_sha256", pyarrow.string(), nullable=False),
    ])


def rejected_arrow_schema():
    pyarrow = require_pyarrow()
    return pyarrow.schema([
        pyarrow.field("id", pyarrow.string(), nullable=False),
        pyarrow.field("source_row_index", pyarrow.int32(), nullable=False),
        pyarrow.field("rejection_reason", pyarrow.string()),
        pyarrow.field("error_category", pyarrow.string()),
        pyarrow.field("error_detail", pyarrow.string()),
        pyarrow.field("attempt_count", pyarrow.int32(), nullable=False),
        pyarrow.field("updated_at", pyarrow.float64()),
        pyarrow.field("image_sha256", pyarrow.string(), nullable=False),
        pyarrow.field("tikz_sha256", pyarrow.string(), nullable=False),
    ])


def attempts_arrow_schema():
    pyarrow = require_pyarrow()
    return pyarrow.schema([
        pyarrow.field("row_id", pyarrow.string(), nullable=False),
        pyarrow.field("attempt_no", pyarrow.int32(), nullable=False),
        pyarrow.field("attempt_kind", pyarrow.string(), nullable=False),
        pyarrow.field("state", pyarrow.string(), nullable=False),
        pyarrow.field("worker", pyarrow.string()),
        pyarrow.field("max_tokens", pyarrow.int32()),
        pyarrow.field("started_at", pyarrow.float64()),
        pyarrow.field("ended_at", pyarrow.float64()),
        pyarrow.field("finish_reason", pyarrow.string()),
        pyarrow.field("error_category", pyarrow.string()),
        pyarrow.field("error_detail", pyarrow.string()),
        pyarrow.field("prompt_tokens", pyarrow.int32()),
        pyarrow.field("completion_tokens", pyarrow.int32()),
        pyarrow.field("instruction", pyarrow.string()),
        pyarrow.field("model_revision", pyarrow.string()),
        pyarrow.field("prompt_sha256", pyarrow.string()),
        pyarrow.field("config_hash", pyarrow.string()),
    ])


def write_parquet_atomic(path, rows: list, schema, identity_column: str = "id") -> dict:
    """Write a parquet file with a tmp -> validate -> fsync -> rename lifecycle."""
    import pyarrow.parquet as parquet
    pyarrow = require_pyarrow()
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table = pyarrow.Table.from_pylist(rows, schema=schema)
    parquet.write_table(table, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    check = parquet.read_table(temporary)
    if check.num_rows != len(rows):
        temporary.unlink(missing_ok=True)
        raise ExportError(f"Shard {path.name} failed read-back validation")
    if len(rows) and check.column(identity_column)[0].as_py() != rows[0][identity_column]:
        temporary.unlink(missing_ok=True)
        raise ExportError(f"Shard {path.name} failed identity validation")
    os.replace(temporary, path)
    fsync_dir(path.parent)
    return dict(rows=len(rows), file_sha256=sha256_file(path))


def logical_row(row: dict) -> str:
    return json.dumps(dict(id=row["id"], source_row_index=row["source_row_index"],
                           image_sha256=row["image_sha256"], tikz_sha256=row["tikz_sha256"],
                           instruction=row["instruction"], prompt_version=row["prompt_version"]),
                      sort_keys=True, ensure_ascii=False)


def logical_checksum(rows: list) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(logical_row(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _merge_stream(work, cursor, row_key):
    """Yield (manifest_entry, ledger_row) pairs by index order; strict about gaps."""
    ledger_row = cursor.fetchone()
    for entry in iter_manifest(work):
        if ledger_row is None:
            break
        if entry["source_row_index"] < ledger_row["source_row_index"]:
            continue
        if entry["source_row_index"] > ledger_row["source_row_index"]:
            raise ExportError(
                f"Ledger row {ledger_row[row_key]} has no manifest entry at index "
                f"{ledger_row['source_row_index']}")
        yield entry, dict(ledger_row)
        ledger_row = cursor.fetchone()
    if ledger_row is not None:
        raise ExportError(
            f"Ledger row {ledger_row[row_key]} is beyond the frozen manifest")


def export_dataset(work, *, export_dir=None, shard_size: int = 1000) -> dict:
    """Deterministically export complete rows; idempotent for unchanged input."""
    work = Path(work).resolve()
    root = Path(export_dir).resolve() if export_dir else work / "export"
    meta = verify_manifest(work, quick=True)
    record = load_run_record(work)
    if record is None:
        raise ExportError(f"No run record at {run_record_path(work)}; nothing to export")
    identity = record["identity"]
    shard_root = root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    schema = dataset_arrow_schema()
    source = identity["dataset"]
    model = identity["model"]
    prompt = identity["prompt"]

    with Ledger(work, read_only=True) as ledger:
        stored = ledger.meta()
        if stored.get("identity_sha256") != record["identity_sha256"]:
            raise ExportError("Ledger identity does not match run.json; refusing to export")
        if stored.get("manifest_sha256") != meta["manifest_sha256"]:
            raise ExportError("Ledger does not belong to this frozen manifest")
        cursor = ledger.conn.execute(
            "SELECT row_id, source_row_index, instruction, image_sha256, tikz_sha256, "
            "finish_reason, completed_at FROM rows WHERE state = ? "
            "ORDER BY source_row_index", (STATE_COMPLETE,))
        shards, buffer, shard_index = [], [], 0
        validation_failures = []
        total_rows = 0

        def flush_shard():
            nonlocal buffer, shard_index
            if not buffer:
                return
            path = shard_root / f"shard-{shard_index:05d}.parquet"
            written = write_parquet_atomic(path, buffer, schema)
            written.update(logical_sha256=logical_checksum(buffer),
                           first_index=buffer[0]["source_row_index"],
                           last_index=buffer[-1]["source_row_index"],
                           name=path.name)
            shards.append(written)
            shard_index += 1
            buffer = []

        for entry, row in _merge_stream(work, cursor, "row_id"):
            if entry["status"] != "valid":
                raise ExportError(f"Complete row {row['row_id']} maps to a rejected manifest row")
            if sha256_text(entry["tikz_code"]) != entry["tikz_sha256"]:
                raise ExportError(f"TikZ checksum drift for {row['row_id']}")
            image_path = work / entry["image"]
            raw = image_path.read_bytes()
            if sha256_bytes(raw) != entry["image_sha256"]:
                raise ExportError(f"Image checksum drift for {row['row_id']}; refusing to export")
            valid, rule, detail = validate_instruction(row["instruction"], ValidationPolicy(
                min_chars=identity["validation"]["min_chars"],
                max_chars=identity["validation"]["max_chars"],
                max_words=identity["validation"]["max_words"]))
            if not valid:
                validation_failures.append(dict(id=row["row_id"], rule=rule, detail=detail))
                continue
            buffer.append(dict(
                id=row["row_id"], source_row_index=row["source_row_index"],
                file_id=entry.get("file_id"), png_image=raw,
                tikz_code=entry["tikz_code"], instruction=row["instruction"],
                source_dataset=source["dataset_id"], source_revision=source["revision"],
                caption_model=model["model_id"], caption_model_revision=model["revision"],
                prompt_version=prompt["version"], image_sha256=entry["image_sha256"],
                tikz_sha256=entry["tikz_sha256"]))
            total_rows += 1
            if len(buffer) >= shard_size:
                flush_shard()
        flush_shard()
        if validation_failures:
            raise ExportError(
                f"{len(validation_failures)} complete rows fail the validation policy; "
                "run validate and audit before exporting")
        # Remove shards from a previous export that this run no longer produces.
        for stale in shard_root.glob("shard-*.parquet"):
            if stale.name not in {shard["name"] for shard in shards}:
                stale.unlink()

        rejected_rows = []
        cursor = ledger.conn.execute(
            "SELECT row_id, source_row_index, rejection_reason, last_error_category, "
            "last_error_detail, attempt_count, updated_at, image_sha256, tikz_sha256 "
            "FROM rows WHERE state = ? ORDER BY source_row_index", (STATE_REJECTED,))
        for entry, row in _merge_stream(work, cursor, "row_id"):
            rejected_rows.append(dict(
                id=row["row_id"], source_row_index=row["source_row_index"],
                rejection_reason=row["rejection_reason"],
                error_category=row["last_error_category"],
                error_detail=row["last_error_detail"], attempt_count=row["attempt_count"],
                updated_at=row["updated_at"], image_sha256=row["image_sha256"],
                tikz_sha256=row["tikz_sha256"]))
        rejected_path = root / "rejected.parquet"
        rejected_written = write_parquet_atomic(rejected_path, rejected_rows,
                                                rejected_arrow_schema())
        attempts = ledger.all_attempts()
        attempts_path = root / "attempts.parquet"
        attempts_written = write_parquet_atomic(attempts_path, attempts,
                                                attempts_arrow_schema(),
                                                identity_column="row_id")
        counts = ledger.counts()

    export_meta = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool_version": TOOL_VERSION,
        "run_id": record["run_id"],
        "identity_sha256": record["identity_sha256"],
        "manifest_sha256": meta["manifest_sha256"],
        "rows": total_rows,
        "rejected_rows": len(rejected_rows),
        "shards": len(shards),
        "shard_size": shard_size,
        "dataset_logical_sha256": canonical_digest([shard["logical_sha256"] for shard in shards]),
        "complete_rows": counts["states"]["complete"],
    }
    provenance = {
        "run": record,
        "manifest": meta,
        "export": export_meta,
        "counts": counts,
        "shards": shards,
        "rejected": rejected_written,
        "attempts": attempts_written,
    }
    bench.dump(root / "run-metadata.json", provenance)
    bench.dump(root / "stats.json", dict(
        states=counts["states"], attempts=counts["attempts"],
        truncation_attempts=counts["truncation_attempts"],
        error_categories=counts["error_categories"],
        rejection_reasons=counts["rejection_reasons"],
        prompt_tokens=counts["prompt_tokens"],
        completion_tokens=counts["completion_tokens"],
        first_complete_at=counts["first_complete_at"],
        last_complete_at=counts["last_complete_at"]))
    bench.dump(root / "checksums.json", dict(
        shards=[dict(name=shard["name"], rows=shard["rows"],
                     logical_sha256=shard["logical_sha256"],
                     file_sha256=shard["file_sha256"]) for shard in shards],
        rejected=rejected_written, attempts=attempts_written,
        dataset_logical_sha256=export_meta["dataset_logical_sha256"]))
    bench.dump(root / "validation-report.json", dict(
        status="pass", checked=total_rows, failures=[],
        rules_sha256=validation_rules_sha256()))
    write_text_atomic(root / "dataset-card.md", render_dataset_card(provenance))
    bench.dump(root / "export.meta.json", export_meta)
    return export_meta


def render_dataset_card(provenance: dict) -> str:
    identity = provenance["run"]["identity"]
    source, model, prompt = identity["dataset"], identity["model"], identity["prompt"]
    export = provenance["export"]
    counts = provenance["counts"]
    return "\n".join([
        "# TikZ instruction dataset (DRAFT - not uploaded)",
        "",
        f"Auto-generated text-to-TikZ instructions for {export['rows']} diagrams, "
        "derived from a frozen slice of the source dataset.",
        "",
        "## Source",
        "",
        f"- Dataset: `{source['dataset_id']}` at revision `{source['revision']}` "
        f"(split `{source['split']}`, rows {source['row_start']}-"
        f"{source['row_start'] + source['row_limit'] - 1})",
        f"- Caption model: `{model['model_id']}` at revision `{model['revision']}`",
        f"- Prompt: `{prompt['version']}` (SHA256 `{prompt['sha256'][:16]}...`)",
        f"- Run: `{provenance['run']['run_id']}`",
        "",
        "## Contents",
        "",
        "- `shards/shard-*.parquet`: accepted rows sorted by source index",
        "- `rejected.parquet`: rejected rows with explicit reasons",
        "- `attempts.parquet`: full attempt history",
        "- `checksums.json`, `run-metadata.json`, `stats.json`",
        "",
        "## Schema",
        "",
        "`id`, `source_row_index`, `file_id`, `png_image`, `tikz_code`, `instruction`, "
        "`source_dataset`, `source_revision`, `caption_model`, `caption_model_revision`, "
        "`prompt_version`, `image_sha256`, `tikz_sha256`",
        "",
        "## Counts",
        "",
        f"- complete: {counts['states']['complete']}",
        f"- rejected: {counts['states']['rejected']}",
        f"- attempts: {counts['attempts']}",
        f"- truncated attempts: {counts['truncation_attempts']}",
        "",
        "## Intended use",
        "",
        "Supervised fine-tuning for text-to-TikZ generation: the instruction is the model "
        "input and the TikZ code is the target output.",
        "",
        "## Limitations",
        "",
        "- Captions are model-generated and have not been human-reviewed.",
        "- The source slice is the first rows of the split, not a random sample.",
        "- Rejected rows are excluded from the training shards; inspect them before "
        "assuming the dataset is complete.",
        "- This card is a draft; publishing requires separate authorization.",
        "",
    ])


def cmd_export(args) -> int:
    require_slurm(args)
    meta = export_dataset(args.work, export_dir=args.export_dir, shard_size=args.shard_size)
    print(f"Export complete: {meta['rows']} rows in {meta['shards']} shards, "
          f"{meta['rejected_rows']} rejected")
    print(f"  dataset_logical_sha256={meta['dataset_logical_sha256']}")
    return 0


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

    run = subparsers.add_parser(
        "run", help="Run identity-safe inference until rows reach a terminal state")
    add_common_arguments(run)
    run.add_argument("--vllm-sif", default="", help="vLLM Apptainer image for engine workers")
    run.add_argument("--max-rows-this-run", type=int, default=None,
                     help="Bounded production chunk: stop claiming after this many rows")
    run.add_argument("--max-runtime-minutes", type=float, default=None,
                     help="Stop claiming and drain workers after this many minutes")
    run.add_argument("--start-index", type=int, default=ROW_START)
    run.add_argument("--end-index", type=int, default=ROW_LIMIT - 1)
    run.add_argument("--worker-restarts", type=int, default=2,
                     help="In-run restarts per worker before its rows are released")
    run.add_argument("--stale-claim-seconds", type=float, default=3600,
                     help="Running rows older than this are reclaimed at startup")
    run.add_argument("--worker-timeout", type=float, default=1800,
                     help="Kill a worker with no heartbeat for this many seconds")
    run.add_argument("--poll-seconds", type=float, default=2.0)
    run.add_argument("--shutdown-grace-seconds", type=float, default=120)
    run.add_argument("--retry-wait-seconds", type=float, default=300,
                     help="Wait this long for short retry backoffs before ending the run")
    run.add_argument("--warmup-samples", type=int, default=1,
                     help="Untimed warmup generations per worker (0 disables)")
    run.add_argument("--reprocess-rejected", action="store_true",
                     help="Explicitly move rejected rows back to pending")
    run.add_argument("--max-transient-attempts", type=int, default=3,
                     help="Bounded transient retry budget per row")
    run.add_argument("--retry-backoff-base-seconds", type=float, default=30.0)
    run.add_argument("--retry-backoff-cap-seconds", type=float, default=600.0)
    run.set_defaults(handler=cmd_run)

    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--job", required=True)
    worker.set_defaults(handler=lambda args: run_worker(args.job))

    export = subparsers.add_parser(
        "export", help="Write deterministic atomic Parquet shards for complete rows")
    add_common_arguments(export)
    export.add_argument("--export-dir", default="", help="Default: WORK/export")
    export.add_argument("--shard-size", type=int, default=1000)
    export.set_defaults(handler=cmd_export)
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
    except (ConfigError, IdentityMismatch, PromptError, ManifestError, LedgerError,
            PipelineIdentityError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
