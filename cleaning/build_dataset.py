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
import dataclasses
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
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
    except (ConfigError, IdentityMismatch, PromptError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
