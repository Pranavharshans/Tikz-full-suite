"""Immutable run identities, run records and gate evidence.

Identity inputs (requirement: dataset logical hash, split-manifest hash, model
id and exact revision, tokenizer/chat-template hashes, resolved configuration,
dependency versions, repository commit, seed):

- the prepared data identity (which itself pins the cleaning export's
  ``dataset_logical_sha256`` and the split-manifest hash),
- the model id, revision, adapter and loader,
- the tokenizer fingerprint (file hashes, vocab, pad token, template hash),
- the resolved configuration payload (paths and other provenance excluded),
- the dependency version snapshot plus the lock file hash,
- the Stage 1 source hash and the repository commit,
- the seed.

Gate overrides (row caps, epochs, checkpoint frequency) are recorded in gate
evidence but deliberately excluded from the run identity: gates are stages of
one run, not separate runs.
"""
from __future__ import annotations

from pathlib import Path

from .errors import DataError, GateError, IdentityMismatch
from .util import (canonical_digest, read_json, source_tree_sha256, utc_now_iso,
                   write_json_atomic)

RUN_RECORD_NAME = "run.json"
PREFLIGHT_REPORT_NAME = "preflight.json"
GATES_DIRNAME = "gates"
METRICS_DIRNAME = "metrics"
EVALUATIONS_DIRNAME = "evaluations"
CHECKPOINTS_DIRNAME = "checkpoints"
FINAL_DIRNAME = "final"

RUN_RECORD_SCHEMA = "stage1-run-v1"
GATE_EVIDENCE_SCHEMA = "stage1-gate-v1"
PREFLIGHT_SCHEMA = "stage1-preflight-v1"

GATE_PREREQUISITES = {
    "overfit-100": ("preflight",),
    "smoke-1000": ("preflight", "overfit-100"),
    "full": ("preflight", "overfit-100", "smoke-1000"),
}


# ---------------------------------------------------------------------------
# Identity construction
# ---------------------------------------------------------------------------


def stage1_code_sha256() -> str:
    return source_tree_sha256(Path(__file__).resolve().parent)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_run_identity(*, data_identity: dict, config, model_fingerprint: dict,
                       dependencies: dict, code_sha256: str,
                       repo_commit: str,
                       dataset_report_sha256: str | None = None) -> dict:
    """Assemble the identity payload and its digest.

    ``dataset_report_sha256`` anchors the prepared model/token/quarantine
    artifacts to the data gate: the report records each artifact's hash, and
    this hash anchors the report itself.
    """
    if not data_identity.get("sha256"):
        raise DataError("data identity has no sha256")
    fingerprint_core = {
        key: value for key, value in model_fingerprint.items()
        if key not in ("chat_template_source",)
    }
    payload = {
        "tool_version": config.tool_version,
        "seed": config.seed,
        "data": {
            "data_identity_sha256": data_identity["sha256"],
            "dataset_logical_sha256": data_identity["dataset_logical_sha256"],
            "split_manifest_sha256": data_identity["split_manifest_sha256"],
            "dataset_report_sha256": dataset_report_sha256,
            "row_count": data_identity["row_count"],
            "source_dataset": data_identity["source_dataset"],
            "source_revision": data_identity["source_revision"],
        },
        "model": {
            "id": config.model.id,
            "revision": config.model.revision,
            "adapter": config.model.adapter,
            "loader": config.model.loader,
        },
        "tokenizer": fingerprint_core,
        "config": config.identity_payload(),
        "environment": {
            "lock_file": config.environment.lock_file,
            "lock_sha256": config.environment.lock_sha256,
            "python_version": config.environment.python_version,
            "packages": dict(sorted(dependencies.items())),
        },
        "code": {
            "sha256": code_sha256,
            "repo_commit": repo_commit,
        },
    }
    identity = dict(payload)
    identity["sha256"] = canonical_digest(payload)
    return identity


def identity_diff(left, right, path: str = "") -> list:
    """Human-readable differences between two identity payloads."""
    differences = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else key
            if key not in left:
                differences.append(f"{child}: missing on the existing run")
            elif key not in right:
                differences.append(f"{child}: missing in the requested run")
            else:
                differences.extend(identity_diff(left[key], right[key], child))
    elif left != right:
        differences.append(f"{path or '<root>'}: existing={_short(left)} requested={_short(right)}")
    return differences


def _short(value) -> str:
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------


def run_record_path(run_dir) -> Path:
    return Path(run_dir) / RUN_RECORD_NAME


def metrics_path(run_dir, gate: str) -> Path:
    return Path(run_dir) / METRICS_DIRNAME / f"{gate}.json"


def gate_evidence_path(run_dir, gate: str) -> Path:
    return Path(run_dir) / GATES_DIRNAME / f"{gate}.json"


def preflight_report_path(run_dir) -> Path:
    return Path(run_dir) / PREFLIGHT_REPORT_NAME


def evaluation_dir(run_dir) -> Path:
    return Path(run_dir) / EVALUATIONS_DIRNAME


def load_run_record(run_dir):
    path = run_record_path(run_dir)
    if not path.is_file():
        return None
    record = read_json(path)
    if record.get("schema_version") != RUN_RECORD_SCHEMA:
        raise IdentityMismatch([
            f"run.json schema {record.get('schema_version')!r} is not "
            f"{RUN_RECORD_SCHEMA!r}"])
    return record


def ensure_run_record(run_dir, identity: dict) -> dict:
    """Write the run record once, or refuse when it belongs to another run."""
    run_dir = Path(run_dir)
    record = load_run_record(run_dir)
    if record is not None:
        if record.get("identity_sha256") != identity["sha256"]:
            raise IdentityMismatch(
                identity_diff(record.get("identity", {}), identity))
        return record
    orphans = _orphan_artifacts(run_dir)
    if orphans:
        raise IdentityMismatch([
            "run directory contains artifacts but no run.json:",
            *[f"  {item}" for item in orphans],
        ])
    record = {
        "schema_version": RUN_RECORD_SCHEMA,
        "tool_version": identity["tool_version"],
        "created_at": utc_now_iso(),
        "identity_sha256": identity["sha256"],
        "identity": identity,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(run_record_path(run_dir), record)
    return record


def _orphan_artifacts(run_dir: Path) -> list:
    if not run_dir.exists():
        return []
    suspicious = []
    for relative in (CHECKPOINTS_DIRNAME, FINAL_DIRNAME, METRICS_DIRNAME,
                     GATES_DIRNAME, EVALUATIONS_DIRNAME, PREFLIGHT_REPORT_NAME):
        path = run_dir / relative
        if path.exists():
            suspicious.append(str(path))
    return suspicious


# ---------------------------------------------------------------------------
# Gate evidence
# ---------------------------------------------------------------------------


def load_gate_evidence(run_dir, gate: str):
    path = gate_evidence_path(run_dir, gate)
    if not path.is_file():
        return None
    evidence = read_json(path)
    if evidence.get("schema_version") != GATE_EVIDENCE_SCHEMA:
        raise GateError(
            f"{path} has schema {evidence.get('schema_version')!r}; expected "
            f"{GATE_EVIDENCE_SCHEMA!r}")
    return evidence


def write_gate_evidence(run_dir, gate: str, *, status: str, identity_sha256: str,
                        payload: dict) -> Path:
    if status not in ("passed", "failed"):
        raise GateError(f"Gate evidence status must be passed/failed, got {status!r}")
    evidence = {
        "schema_version": GATE_EVIDENCE_SCHEMA,
        "gate": gate,
        "status": status,
        "identity_sha256": identity_sha256,
        "created_at": utc_now_iso(),
        **payload,
    }
    path = gate_evidence_path(run_dir, gate)
    write_json_atomic(path, evidence)
    return path


def require_gate_start(run_dir, gate: str, identity_sha256: str,
                       *, allow_rerun: bool = False) -> dict | None:
    """Refuse to start a gate that already passed for this identity."""
    evidence = load_gate_evidence(run_dir, gate)
    if evidence is None:
        return None
    if evidence.get("identity_sha256") != identity_sha256:
        raise IdentityMismatch([
            f"gate {gate!r} evidence belongs to identity "
            f"{evidence.get('identity_sha256')}, requested {identity_sha256}"])
    if evidence.get("status") == "passed" and not allow_rerun:
        raise GateError(
            f"gate {gate!r} already passed for this identity. Stage 1 does not "
            "silently redo a passed gate; pass --rerun-gate to repeat it "
            "deliberately.")
    return evidence


def require_prerequisites(run_dir, gate: str, identity_sha256: str) -> None:
    """Enforce explicit gate order; never advance automatically."""
    if gate not in GATE_PREREQUISITES:
        raise GateError(f"No prerequisite chain defined for gate {gate!r}")
    problems = []
    for prerequisite in GATE_PREREQUISITES[gate]:
        if prerequisite == "preflight":
            path = preflight_report_path(run_dir)
            evidence = read_json(path) if path.is_file() else None
            if evidence is None:
                problems.append("preflight: not run")
                continue
            if evidence.get("identity_sha256") != identity_sha256:
                problems.append("preflight: identity mismatch")
                continue
            if evidence.get("status") != "passed":
                problems.append(f"preflight: status {evidence.get('status')}")
                continue
            if not evidence.get("complete", False):
                problems.append("preflight: incomplete (checks were skipped)")
            continue
        evidence = load_gate_evidence(run_dir, prerequisite)
        if evidence is None:
            problems.append(f"{prerequisite}: not run")
        elif evidence.get("identity_sha256") != identity_sha256:
            problems.append(f"{prerequisite}: identity mismatch")
        elif evidence.get("status") != "passed":
            problems.append(f"{prerequisite}: status {evidence.get('status')}")
    if problems:
        raise GateError(
            f"gate {gate!r} requires prerequisite evidence that is missing or "
            "unsatisfied:\n" + "\n".join(f"  - {item}" for item in problems) +
            "\nRun the prerequisite gates first; Stage 1 never advances gates "
            "automatically.")


def write_preflight_report(run_dir, *, identity_sha256: str, checks: list,
                           status: str, complete: bool, payload: dict | None = None) -> Path:
    report = {
        "schema_version": PREFLIGHT_SCHEMA,
        "created_at": utc_now_iso(),
        "identity_sha256": identity_sha256,
        "status": status,
        "complete": complete,
        "checks": checks,
        **(payload or {}),
    }
    path = preflight_report_path(run_dir)
    write_json_atomic(path, report)
    return path
