"""Small, boring helpers shared across Stage 1.

Everything here is stdlib-only. ``canonical_digest`` reproduces the cleaning
pipeline's ``bench.digest`` byte for byte so that dataset logical checksums can
be re-derived independently (see ``tests/test_cleaning_consistency.py``).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from .errors import ConfigError

SHA256_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+!-]*")

# Distribution name -> import name for the environment identity snapshot.
DEPENDENCY_DISTRIBUTIONS = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("unsloth", "unsloth"),
    ("unsloth-zoo", "unsloth_zoo"),
    ("trl", "trl"),
    ("datasets", "datasets"),
    ("accelerate", "accelerate"),
    ("peft", "peft"),
    ("bitsandbytes", "bitsandbytes"),
    ("tokenizers", "tokenizers"),
    ("safetensors", "safetensors"),
    ("huggingface-hub", "huggingface_hub"),
    ("sentencepiece", "sentencepiece"),
    ("numpy", "numpy"),
    ("pyarrow", "pyarrow"),
    ("PyYAML", "yaml"),
    ("pillow", "PIL"),
    ("triton", "triton"),
    ("xformers", "xformers"),
)


def _normalize_distribution(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def distribution_import_name(name: str):
    """Map a PyPI distribution name to its import name, or None."""
    table = {_normalize_distribution(distribution): import_name
             for distribution, import_name in DEPENDENCY_DISTRIBUTIONS}
    return table.get(_normalize_distribution(name))


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
    """Byte-for-byte equivalent of ``cleaning/benchmark.py:digest``.

    The cleaning pipeline hashes JSON with sorted keys, default separators and
    default ``ensure_ascii=True``. Do not "improve" this function: dataset
    logical checksums must be reproducible against the cleaning export.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def is_sha256(value) -> bool:
    return bool(SHA256_RE.fullmatch(value or ""))


def is_pinned_revision(value) -> bool:
    return bool(REVISION_RE.fullmatch(value or ""))


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def human_bytes(value) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def fsync_dir(path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_text_atomic(path, text: str) -> None:
    """Durable writer: unique temp name, fsync, atomic rename, directory fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_dir(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def write_json_atomic(path, value) -> None:
    write_text_atomic(path, canonical_json(value))


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def require_absolute(path, what: str) -> Path:
    """Reject relative and empty paths early, with the caller's context."""
    if path is None or str(path) == "":
        raise ConfigError(f"{what} is required; pass an absolute path")
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ConfigError(f"{what} must be an absolute path, got {str(path)!r}")
    return candidate


def detect_repo_commit(repo_root) -> str:
    """Best-effort git commit for identity and provenance; never raises."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10)
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def source_tree_sha256(root) -> str:
    """Hash every ``*.py`` under ``root`` so mid-run code edits break resume."""
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        files[str(path.relative_to(root))] = sha256_file(path)
    if not files:
        raise ConfigError(f"No Python sources found under {root}")
    return canonical_digest(files)


def dependency_versions() -> dict:
    """Snapshot the environment identity; missing packages are explicit."""
    versions = {}
    for distribution, import_name in DEPENDENCY_DISTRIBUTIONS:
        try:
            versions[import_name] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[import_name] = "not-installed"
    versions["python"] = ".".join(str(part) for part in sys.version_info[:3])
    return versions


def parse_lock_file(path) -> dict:
    """Parse ``name==version`` requirement lines; anything else is a defect.

    Interpreter pins are rejected: the Python version lives in
    ``stage-1/.python-version`` so the lock stays a pip-installable
    requirements file.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Lock file not found: {path}")
    pins = {}
    for number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ConfigError(
                f"{path}:{number}: lock lines must be 'name==version', got {line!r}")
        name, _, version = line.partition("==")
        name, version = name.strip(), version.strip()
        if not NAME_RE.fullmatch(name):
            raise ConfigError(
                f"{path}:{number}: {name!r} is not a valid distribution name")
        if name.lower() == "python":
            raise ConfigError(
                f"{path}:{number}: interpreter pins do not belong in a "
                "requirements file; put the Python version in .python-version")
        if not VERSION_RE.fullmatch(version):
            raise ConfigError(
                f"{path}:{number}: {version!r} is not a valid version pin")
        if name in pins:
            raise ConfigError(f"{path}:{number}: duplicate pin for {name!r}")
        pins[name] = version
    if not pins:
        raise ConfigError(f"Lock file {path} contains no pins")
    return pins


def python_version_pin() -> str:
    """The repository's ``stage-1/.python-version`` interpreter pin, or ''."""
    path = Path(__file__).resolve().parents[2] / ".python-version"
    if not path.is_file():
        return ""
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return ""


def verify_lock_versions(pins: dict, installed: dict | None = None) -> list:
    """Return human-readable differences between a lock and the environment."""
    installed = dependency_versions() if installed is None else installed
    differences = []
    for distribution, expected in pins.items():
        import_name = distribution_import_name(distribution)
        if import_name is None:
            differences.append(
                f"{distribution}: pinned {expected}, cannot verify "
                "(unknown distribution name)")
            continue
        actual = installed.get(import_name)
        if actual != expected:
            differences.append(f"{distribution}: pinned {expected}, installed {actual}")
    return differences


def lock_sha256(path) -> str:
    return sha256_file(path)


def parse_wall_time(value: str) -> int:
    """Parse Slurm wall time into whole minutes (same semantics as cleaning).

    Accepts minutes, ``MM:SS``, ``HH:MM:SS`` and ``D-HH[:MM[:SS]]``.
    """
    text = str(value).strip()
    days = 0
    if "-" in text:
        day_part, text = text.split("-", 1)
        if not day_part.isdigit():
            raise ConfigError(f"Invalid wall time {value!r}")
        days = int(day_part)
    if not text or not all(part.isdigit() for part in text.split(":")):
        raise ConfigError(
            f"Invalid wall time {value!r}; use minutes, MM:SS, HH:MM:SS or D-HH:MM:SS")
    parts = [int(part) for part in text.split(":")]
    hours = minutes = seconds = 0
    if len(parts) == 1:
        minutes = parts[0]
    elif len(parts) == 2:
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ConfigError(f"Invalid wall time {value!r}")
    if len(parts) > 1 and (minutes > 59 or seconds > 59 or (days and hours > 23)):
        raise ConfigError(f"Invalid wall time {value!r}")
    total_seconds = days * 24 * 3600 + hours * 3600 + minutes * 60 + seconds
    if total_seconds <= 0:
        raise ConfigError("Wall time must be positive")
    return max(1, total_seconds // 60)
