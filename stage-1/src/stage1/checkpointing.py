"""Checkpoint and final-artifact integrity, provenance and resume safety.

Artifacts are namespaced per gate:

    RUN/checkpoints/<gate>/checkpoint-<step>/   resumable training checkpoints
    RUN/final/<gate>/                           final model of a completed gate

Every artifact carries ``stage1-checkpoint.json`` (kind ``checkpoint``) or
``stage1-final.json`` (kind ``final``) with the run identity, gate, model id,
revision and adapter. Verification refuses:

- incomplete checkpoints (missing config/weights/optimizer/scheduler/state),
- foreign identities, models, revisions, adapters or gates,
- directories with no metadata at all,
- a newest checkpoint that is invalid (never falls back to an older one).

A gate resumes only its own namespace, and metadata must agree with the
requested gate, so a smoke run cannot resume an overfit checkpoint even if the
directory is copied.
"""
from __future__ import annotations

import re
from pathlib import Path

from .errors import CheckpointError
from .util import read_json, utc_now_iso, write_json_atomic

ARTIFACT_SCHEMA_VERSION = "stage1-artifact-v1"
CHECKPOINT_META_NAME = "stage1-checkpoint.json"
FINAL_META_NAME = "stage1-final.json"
CHECKPOINT_DIR_RE = re.compile(r"checkpoint-(\d+)")

TRAINING_GATES = ("overfit-100", "smoke-1000", "full")
TRAINING_METHODS = ("full", "lora")

REQUIRED_CHECKPOINT_FILES = ("config.json", "trainer_state.json", "optimizer.pt",
                             "scheduler.pt")
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin",
                "model.safetensors.index.json", "pytorch_model.bin.index.json")
ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json")

META_FILES = (CHECKPOINT_META_NAME, FINAL_META_NAME)

BASE_PROVENANCE_FIELDS = ("base_model_id", "base_model_revision")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def checkpoints_root(run_dir) -> Path:
    return Path(run_dir) / "checkpoints"


def gate_checkpoints_dir(run_dir, gate: str) -> Path:
    return checkpoints_root(run_dir) / gate


def final_dir(run_dir, gate: str) -> Path:
    return Path(run_dir) / "final" / gate


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def write_artifact_meta(directory, *, kind: str, identity_sha256: str,
                        model_id: str, model_revision: str, adapter: str,
                        gate: str | None, global_step: int,
                        supervised_tokens_seen: int, epochs_completed: float,
                        training_method: str = "full",
                        base_model_id: str | None = None,
                        base_model_revision: str | None = None,
                        lora: dict | None = None,
                        run_id: str | None = None) -> Path:
    if kind not in ("checkpoint", "final"):
        raise CheckpointError(f"Unknown artifact kind {kind!r}")
    if gate is not None and gate not in TRAINING_GATES:
        raise CheckpointError(f"Unknown gate {gate!r}")
    if training_method not in TRAINING_METHODS:
        raise CheckpointError(f"Unknown training method {training_method!r}")
    if training_method == "lora" and not (base_model_id and base_model_revision):
        raise CheckpointError(
            "LoRA artifacts must record the base model id and revision; an "
            "adapter is not a standalone model")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "kind": kind,
        "created_at": utc_now_iso(),
        "identity_sha256": identity_sha256,
        "run_id": run_id,
        "gate": gate,
        "model_id": model_id,
        "model_revision": model_revision,
        "adapter": adapter,
        "training_method": training_method,
        "base_model_id": base_model_id,
        "base_model_revision": base_model_revision,
        "lora": lora,
        "global_step": int(global_step),
        "supervised_tokens_seen": int(supervised_tokens_seen),
        "epochs_completed": float(epochs_completed),
    }
    name = CHECKPOINT_META_NAME if kind == "checkpoint" else FINAL_META_NAME
    path = directory / name
    write_json_atomic(path, meta)
    return path


def write_checkpoint_meta(directory, **kwargs) -> Path:
    return write_artifact_meta(directory, kind="checkpoint", **kwargs)


def write_final_meta(directory, **kwargs) -> Path:
    return write_artifact_meta(directory, kind="final", **kwargs)


def read_artifact_meta(directory, *, required: bool = True) -> dict | None:
    directory = Path(directory)
    present = [name for name in META_FILES if (directory / name).is_file()]
    if len(present) > 1:
        raise CheckpointError(
            f"{directory} contains both {present}; refusing an ambiguous artifact")
    if not present:
        if required:
            raise CheckpointError(
                f"{directory} has no Stage 1 artifact metadata "
                f"({CHECKPOINT_META_NAME} or {FINAL_META_NAME}); refusing to use "
                "an artifact whose provenance cannot be verified.")
        return None
    meta = read_json(directory / present[0])
    if meta.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise CheckpointError(
            f"{directory / present[0]} has schema "
            f"{meta.get('schema_version')!r}; expected {ARTIFACT_SCHEMA_VERSION!r}")
    if meta.get("kind") not in ("checkpoint", "final"):
        raise CheckpointError(f"{directory} has an invalid artifact kind")
    return meta


def verify_artifact_meta(meta: dict, *, path, identity_sha256: str | None = None,
                         model_id: str | None = None,
                         model_revision: str | None = None,
                         adapter: str | None = None, gate: str | None = None,
                         method: str | None = None,
                         kinds=("checkpoint", "final")) -> dict:
    """Validate provenance fields; raise ``CheckpointError`` on any mismatch."""
    if meta.get("kind") not in kinds:
        raise CheckpointError(
            f"{path} is a {meta.get('kind')!r} artifact; expected one of "
            f"{list(kinds)}")
    if not isinstance(meta.get("global_step"), int) or meta["global_step"] < 0:
        raise CheckpointError(f"{path} has an invalid global_step")
    if not isinstance(meta.get("supervised_tokens_seen"), int):
        raise CheckpointError(f"{path} has an invalid supervised_tokens_seen")
    artifact_method = meta.get("training_method")
    if artifact_method not in TRAINING_METHODS:
        raise CheckpointError(
            f"{path} has training_method={artifact_method!r}; expected one of "
            f"{list(TRAINING_METHODS)}")
    if method is not None and artifact_method != method:
        raise CheckpointError(
            f"{path} belongs to a {artifact_method!r} run, but this run uses "
            f"{method!r}; full and LoRA checkpoints never share or resume each "
            "other.")
    if artifact_method == "lora":
        missing = [field for field in BASE_PROVENANCE_FIELDS if not meta.get(field)]
        if missing:
            raise CheckpointError(
                f"{path} is a LoRA adapter without base-model provenance "
                f"({missing}); refusing to use an adapter as a standalone model.")
    expected = {
        "identity_sha256": identity_sha256,
        "model_id": model_id,
        "model_revision": model_revision,
        "adapter": adapter,
    }
    for field, wanted in expected.items():
        if wanted is None:
            continue
        actual = meta.get(field)
        if actual != wanted:
            raise CheckpointError(
                f"{path} has {field}={actual!r}, expected {wanted!r}; refusing "
                "to use an artifact from a different run, model or adapter.")
    if gate is not None and meta.get("gate") != gate:
        raise CheckpointError(
            f"{path} belongs to gate {meta.get('gate')!r}, not {gate!r}; "
            "refusing cross-gate resume or evaluation.")
    return meta


# ---------------------------------------------------------------------------
# Completeness and unified verification
# ---------------------------------------------------------------------------


def incomplete_reason(checkpoint_dir, *, method: str = "full") -> str | None:
    """Missing-file reason for a resumable checkpoint of the given method."""
    if method not in TRAINING_METHODS:
        return f"unknown training method {method!r}"
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return f"{checkpoint_dir} is not a directory"
    if method == "lora":
        required = (ADAPTER_CONFIG_FILE,) + REQUIRED_CHECKPOINT_FILES[1:]
        weight_files = ADAPTER_WEIGHT_FILES
        weight_hint = "no adapter weight file (adapter_model.safetensors)"
    else:
        required = REQUIRED_CHECKPOINT_FILES
        weight_files = WEIGHT_FILES
        weight_hint = "no model weight file (model.safetensors / pytorch_model.bin)"
    missing = [name for name in required if not (checkpoint_dir / name).is_file()]
    if missing:
        return f"missing files: {', '.join(missing)}"
    if not any((checkpoint_dir / name).is_file() for name in weight_files):
        return weight_hint
    return None


def verify_checkpoint(checkpoint_dir, identity_sha256: str, *,
                      gate: str | None = None, model_id: str | None = None,
                      model_revision: str | None = None,
                      adapter: str | None = None,
                      method: str | None = None,
                      require_meta: bool = True) -> dict:
    """Verify a resumable checkpoint: metadata/provenance first, then files.

    Provenance errors (foreign identity, model, gate or training method) are
    reported before layout errors, so a full artifact presented to a LoRA run
    fails with the cross-method message rather than a missing-adapter message.
    Completeness is then checked against the artifact's own method.
    """
    checkpoint_dir = Path(checkpoint_dir)
    meta = read_artifact_meta(checkpoint_dir, required=require_meta)
    if meta is not None:
        meta = verify_artifact_meta(
            meta, path=checkpoint_dir, identity_sha256=identity_sha256,
            model_id=model_id, model_revision=model_revision, adapter=adapter,
            gate=gate, method=method, kinds=("checkpoint",))
        artifact_method = meta["training_method"]
    else:
        artifact_method = method or "full"
    reason = incomplete_reason(checkpoint_dir, method=artifact_method)
    if reason:
        raise CheckpointError(
            f"Checkpoint {checkpoint_dir} is incomplete ({reason}); refusing to "
            "resume. Remove or repair it explicitly.")
    if meta is None:
        return {}
    return meta


def verify_final_artifact(directory, *, identity_sha256: str,
                          model_id: str, model_revision: str, adapter: str,
                          gate: str | None = None,
                          method: str | None = None) -> dict:
    """Verify a final artifact: metadata, weights and tokenizer files.

    Full artifacts are complete model directories; LoRA artifacts are native
    PEFT adapters that require the pinned base model, so they must carry base
    provenance and adapter files instead of full model weights.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise CheckpointError(f"Final artifact directory does not exist: {directory}")
    meta = read_artifact_meta(directory, required=True)
    meta = verify_artifact_meta(
        meta, path=directory, identity_sha256=identity_sha256, model_id=model_id,
        model_revision=model_revision, adapter=adapter, gate=gate, method=method,
        kinds=("final",))
    if meta.get("training_method") == "lora":
        if not (directory / ADAPTER_CONFIG_FILE).is_file():
            raise CheckpointError(
                f"LoRA final artifact {directory} has no {ADAPTER_CONFIG_FILE}; "
                "an adapter directory is not a standalone model")
        if not any((directory / name).is_file() for name in ADAPTER_WEIGHT_FILES):
            raise CheckpointError(
                f"LoRA final artifact {directory} has no adapter weights")
    else:
        if not (directory / "config.json").is_file():
            raise CheckpointError(f"Final artifact {directory} has no config.json")
        if not any((directory / name).is_file() for name in WEIGHT_FILES):
            raise CheckpointError(f"Final artifact {directory} has no model weights")
    if not any((directory / name).is_file() for name in TOKENIZER_FILES):
        raise CheckpointError(
            f"Final artifact {directory} has no tokenizer files; evaluation "
            "could not rebuild the chat template")
    return meta


def verify_evaluation_artifact(directory, *, identity_sha256: str, model_id: str,
                               model_revision: str, adapter: str,
                               gate: str | None = None,
                               method: str | None = None) -> dict:
    """Verify either a resumable checkpoint or a final artifact for evaluation."""
    directory = Path(directory)
    meta = read_artifact_meta(directory, required=True)
    if meta.get("kind") == "checkpoint":
        return verify_checkpoint(
            directory, identity_sha256, gate=gate, model_id=model_id,
            model_revision=model_revision, adapter=adapter, method=method)
    return verify_final_artifact(
        directory, identity_sha256=identity_sha256, model_id=model_id,
        model_revision=model_revision, adapter=adapter, gate=gate, method=method)


def load_artifact_weights(model, directory, *, method: str = "full",
                          loader=None) -> dict:
    """Load an artifact's weights into the model, exactly and method-aware.

    Full artifacts load the complete model state dict. LoRA artifacts load only
    adapter tensors: the saved keys must be adapter keys, must cover every
    adapter parameter of the live model, and must not touch base weights.
    ``loader`` is an injection seam for tests.
    """
    if method not in TRAINING_METHODS:
        raise CheckpointError(f"Unknown training method {method!r}")
    if loader is None:
        try:
            from safetensors.torch import load_file as loader
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise CheckpointError(
                "safetensors is required to load artifact weights") from exc
    weight_names = ADAPTER_WEIGHT_FILES if method == "lora" else WEIGHT_FILES
    saved = {}
    for name in weight_names:
        path = Path(directory) / name
        if path.is_file():
            saved.update(loader(str(path)))
    if not saved:
        raise CheckpointError(
            f"No {'adapter ' if method == 'lora' else ''}weight file found in "
            f"{directory}")
    if method == "lora":
        non_adapter = sorted(key for key in saved if "lora_" not in key)
        if non_adapter:
            raise CheckpointError(
                f"Adapter file in {directory} contains non-adapter tensors "
                f"({non_adapter[:3]}); refusing to treat it as a LoRA artifact")
        live = model.state_dict()
        expected = {key for key in live if "lora_" in key}
        if not expected:
            raise CheckpointError(
                "The live model has no adapter parameters; refusing to load a "
                "LoRA artifact into a non-LoRA model")
        missing = sorted(expected - set(saved))
        if missing:
            raise CheckpointError(
                f"Adapter in {directory} does not cover the model's adapter "
                f"parameters; missing {len(missing)} key(s), first "
                f"{missing[0]}")
    missing, unexpected = model.load_state_dict(saved, strict=False)
    if method == "full":
        if missing or unexpected:
            raise CheckpointError(
                f"Artifact {directory} does not match the model: "
                f"missing={list(missing)[:3]}, unexpected={list(unexpected)[:3]}")
    return {
        "path": str(directory),
        "method": method,
        "tensors": len(saved),
    }


# ---------------------------------------------------------------------------
# Resume-state verification (torch injected)
# ---------------------------------------------------------------------------


def _optimizer_states_equal(left: dict, right: dict) -> bool:
    if left.get("param_groups") != right.get("param_groups"):
        return False
    left_state, right_state = left.get("state", {}), right.get("state", {})
    if set(left_state) != set(right_state):
        return False
    for key in left_state:
        left_entry, right_entry = left_state[key], right_state[key]
        if set(left_entry) != set(right_entry):
            return False
        for field_name, value in left_entry.items():
            other = right_entry[field_name]
            if hasattr(value, "shape"):
                if getattr(other, "shape", None) != value.shape:
                    return False
                if not bool((value == other).all()):
                    return False
            elif value != other:
                return False
    return True


def lambda_schedule_parameters(saved_state) -> dict | None:
    """Extract the schedule parameters from a saved ``LambdaLR`` state.

    Transformers builds its cosine schedule with ``functools.partial`` wrappers
    around a module-level lambda function, so the saved state's ``lr_lambdas``
    carry ``num_warmup_steps``/``num_training_steps``/``num_cycles`` keywords
    that identify the exact schedule. Returns ``None`` when the lambdas are not
    introspectable (for example a locally defined closure).
    """
    if not isinstance(saved_state, dict):
        return None
    lambdas = saved_state.get("lr_lambdas")
    if not isinstance(lambdas, (list, tuple)) or not lambdas:
        return None
    parameters = {}
    for entry in lambdas:
        keywords = getattr(entry, "keywords", None)
        if not isinstance(keywords, dict):
            continue
        for key in ("num_warmup_steps", "num_training_steps", "num_cycles"):
            if key in keywords:
                parameters.setdefault(key, keywords[key])
    return parameters or None


def scheduler_restore_plan(saved_state, *, num_training_steps: int,
                           num_warmup_steps: int) -> dict:
    """Validate a saved scheduler state against the production schedule.

    Production uses ``transformers.get_scheduler("cosine", optimizer,
    num_warmup_steps=..., num_training_steps=...)``, which returns a
    ``LambdaLR``. A state that is not a lambda-based schedule (for example a
    ``CosineAnnealingLR`` state with ``T_max``) is refused rather than guessed
    at, and the saved schedule parameters must match the numbers this run
    reconstructs from its own training schedule.
    """
    if not isinstance(saved_state, dict):
        raise CheckpointError(
            "Scheduler state is not a dictionary; refusing to restore it")
    if "lr_lambdas" not in saved_state:
        raise CheckpointError(
            "Scheduler state is not a Transformers cosine (LambdaLR) state; "
            "production uses get_scheduler('cosine', ...). Refusing to guess "
            "a schedule from it.")
    if not isinstance(num_training_steps, int) or isinstance(num_training_steps, bool) \
            or num_training_steps < 1:
        raise CheckpointError(
            f"num_training_steps must be a positive integer, got "
            f"{num_training_steps!r}")
    if not isinstance(num_warmup_steps, int) or isinstance(num_warmup_steps, bool) \
            or num_warmup_steps < 0:
        raise CheckpointError(
            f"num_warmup_steps must be a non-negative integer, got "
            f"{num_warmup_steps!r}")
    parameters = lambda_schedule_parameters(saved_state)
    problems = []
    if parameters:
        if ("num_training_steps" in parameters
                and parameters["num_training_steps"] != num_training_steps):
            problems.append(
                f"num_training_steps: checkpoint={parameters['num_training_steps']} "
                f"this run={num_training_steps}")
        if ("num_warmup_steps" in parameters
                and parameters["num_warmup_steps"] != num_warmup_steps):
            problems.append(
                f"num_warmup_steps: checkpoint={parameters['num_warmup_steps']} "
                f"this run={num_warmup_steps}")
    if problems:
        raise CheckpointError(
            "Checkpoint schedule does not match this run's training schedule:\n" +
            "\n".join(f"  - {line}" for line in problems))
    return {
        "scheduler": "cosine",
        "num_training_steps": num_training_steps,
        "num_warmup_steps": num_warmup_steps,
        "schedule_parameters": parameters,
        "schedule_parameters_verified": bool(parameters),
    }


def check_saved_initial_lr(saved_state, expected_lr: float, *,
                           tolerance: float = 1e-12) -> dict:
    """Compare the configured initial LR against saved scheduler ``base_lrs``.

    This must run **before** the optimizer state is loaded: loading an
    optimizer state restores its param-group learning rates, which would mask a
    configuration mismatch. Returns the saved base learning rates.
    """
    if not isinstance(saved_state, dict):
        raise CheckpointError("Scheduler state is not a dictionary")
    saved_base_lrs = saved_state.get("base_lrs")
    if saved_base_lrs is None:
        return {"base_lrs": None, "initial_lr_verified": False}
    try:
        values = [float(value) for value in saved_base_lrs]
    except (TypeError, ValueError) as exc:
        raise CheckpointError(
            f"Saved scheduler base_lrs are not numeric: {saved_base_lrs!r}") from exc
    mismatched = [value for value in values
                  if abs(value - float(expected_lr)) > tolerance]
    if mismatched:
        raise CheckpointError(
            f"Saved scheduler base_lrs {values} do not match the configured "
            f"initial learning rate {float(expected_lr)}; refusing to resume "
            "with a different schedule")
    return {"base_lrs": values, "initial_lr_verified": True}


def verify_scheduler_state(saved_state: dict, restored_state: dict, *,
                           fresh_state: dict | None = None,
                           expected_base_lrs=None,
                           expected_warmup_steps: int | None = None,
                           expected_training_steps: int | None = None,
                           tolerance: float = 1e-9) -> dict:
    """Compare a restored scheduler state against the saved state.

    ``fresh_state`` is the state of the scheduler as reconstructed by this run
    *before* the checkpoint state was loaded. Its lambda values are probed at
    key steps (start, warmup boundary, midpoint, end) and compared against the
    saved lambdas, which proves that this run's total/warmup steps reproduce
    the checkpoint's schedule. If the lambdas cannot be called, the report
    records ``lambda_schedule_verified: False`` instead of claiming it.
    """
    problems = []
    for field_name in ("last_epoch", "_step_count"):
        if field_name in saved_state and restored_state.get(field_name) != saved_state[field_name]:
            problems.append(
                f"{field_name}: saved={saved_state.get(field_name)!r} "
                f"restored={restored_state.get(field_name)!r}")
    saved_base_lrs = saved_state.get("base_lrs")
    if expected_base_lrs is not None and saved_base_lrs is not None:
        if [float(value) for value in saved_base_lrs] != [float(value) for value in expected_base_lrs]:
            problems.append(
                f"base_lrs: checkpoint was created for learning rates "
                f"{list(saved_base_lrs)} but this run uses "
                f"{list(expected_base_lrs)}")

    parameters = lambda_schedule_parameters(saved_state)
    if parameters:
        if (expected_training_steps is not None
                and "num_training_steps" in parameters
                and parameters["num_training_steps"] != expected_training_steps):
            problems.append(
                f"num_training_steps: checkpoint={parameters['num_training_steps']} "
                f"this run={expected_training_steps}")
        if (expected_warmup_steps is not None
                and "num_warmup_steps" in parameters
                and parameters["num_warmup_steps"] != expected_warmup_steps):
            problems.append(
                f"num_warmup_steps: checkpoint={parameters['num_warmup_steps']} "
                f"this run={expected_warmup_steps}")

    lambda_verified = False
    probe_report = None
    if fresh_state is not None and expected_training_steps:
        probe_report = _compare_lambda_schedules(
            saved_state, fresh_state, expected_training_steps,
            expected_warmup_steps or 0, tolerance=tolerance)
        if probe_report["compared"]:
            lambda_verified = True
            if not probe_report["match"]:
                problems.append(
                    "lambda schedule values differ at steps "
                    f"{probe_report['mismatched_steps']}")

    if problems:
        raise CheckpointError(
            "Restored scheduler state does not match the checkpoint:\n" +
            "\n".join(f"  - {line}" for line in problems))
    return {
        "kind": "lambda" if "lr_lambdas" in saved_state else "unknown",
        "last_epoch": saved_state.get("last_epoch"),
        "step_count": saved_state.get("_step_count"),
        "base_lrs": list(saved_base_lrs) if saved_base_lrs is not None else None,
        "schedule_parameters": parameters,
        "schedule_parameters_verified": bool(parameters),
        "lambda_schedule_verified": lambda_verified,
        "lambda_probes": probe_report,
    }


def _compare_lambda_schedules(saved_state, fresh_state, num_training_steps,
                              num_warmup_steps, *, tolerance: float) -> dict:
    """Probe saved and freshly reconstructed lambdas at key steps."""
    saved_lambdas = saved_state.get("lr_lambdas")
    fresh_lambdas = fresh_state.get("lr_lambdas")
    if (not isinstance(saved_lambdas, (list, tuple)) or not saved_lambdas
            or not isinstance(fresh_lambdas, (list, tuple))
            or len(fresh_lambdas) != len(saved_lambdas)):
        return {"compared": False, "match": None, "mismatched_steps": [],
                "reason": "lambda lists are missing or have different lengths"}
    steps = sorted({0, max(0, num_warmup_steps - 1), num_warmup_steps,
                    num_training_steps // 2, num_training_steps - 1,
                    num_training_steps})
    mismatched = []
    try:
        for saved_lambda, fresh_lambda in zip(saved_lambdas, fresh_lambdas):
            for step in steps:
                saved_value = float(saved_lambda(step))
                fresh_value = float(fresh_lambda(step))
                if abs(saved_value - fresh_value) > tolerance:
                    mismatched.append(step)
    except Exception as exc:
        return {"compared": False, "match": None, "mismatched_steps": [],
                "reason": f"lambda probe failed: {type(exc).__name__}: {exc}"}
    return {"compared": True, "match": not mismatched,
            "mismatched_steps": sorted(set(mismatched)),
            "steps": steps, "lambdas": len(saved_lambdas)}


def verify_resume_state(checkpoint_dir, *, torch, model, learning_rate: float,
                        fused: bool, num_training_steps: int,
                        num_warmup_steps: int, device="cuda",
                        expected_optimizer_state: dict | None = None,
                        scheduler_factory=None) -> dict:
    """Restore optimizer, scheduler and trainer state from a checkpoint.

    The scheduler is reconstructed exactly as production builds it -
    ``transformers.get_scheduler("cosine", optimizer, num_warmup_steps=...,
    num_training_steps=...)`` - and its state is loaded and verified. The
    configured initial learning rate is compared against the saved scheduler
    ``base_lrs`` *before* the optimizer state is loaded, because loading that
    state restores its own param-group learning rates and would mask a
    configuration mismatch. When ``expected_optimizer_state`` is supplied, the
    restored optimizer state must equal it field for field.
    """
    import json as json_module

    checkpoint_dir = Path(checkpoint_dir)
    optimizer_path = checkpoint_dir / "optimizer.pt"
    scheduler_path = checkpoint_dir / "scheduler.pt"
    state_path = checkpoint_dir / "trainer_state.json"
    for path in (optimizer_path, scheduler_path, state_path):
        if not path.is_file():
            raise CheckpointError(f"Resume state file is missing: {path}")
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    if not trainable:
        raise CheckpointError("Model has no trainable parameters to resume")
    try:
        saved_optimizer_state = torch.load(optimizer_path, map_location=device)
        saved_scheduler_state = torch.load(scheduler_path, map_location=device)
    except Exception as exc:
        raise CheckpointError(
            f"Failed to read resume state from {checkpoint_dir}: "
            f"{type(exc).__name__}: {exc}") from exc

    # 1. Validate the saved schedule shape and this run's reconstruction plan.
    plan = scheduler_restore_plan(
        saved_scheduler_state, num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps)
    # 2. Compare the configured initial LR before the optimizer state is loaded.
    initial_lr = check_saved_initial_lr(saved_scheduler_state, learning_rate)

    # 3. Rebuild the production optimizer and scheduler, then load the states.
    try:
        restored_optimizer = torch.optim.AdamW(trainable, lr=learning_rate,
                                               fused=fused)
        restored_optimizer.load_state_dict(saved_optimizer_state)
        if scheduler_factory is None:
            scheduler_factory = _transformers_scheduler_factory()
        restored_scheduler = scheduler_factory(
            restored_optimizer, num_warmup_steps=plan["num_warmup_steps"],
            num_training_steps=plan["num_training_steps"])
        fresh_scheduler_state = restored_scheduler.state_dict()
        restored_scheduler.load_state_dict(saved_scheduler_state)
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(
            f"Resume state in {checkpoint_dir} does not match this model or "
            f"optimizer: {type(exc).__name__}: {exc}") from exc

    state = json_module.loads(state_path.read_text())
    global_step = state.get("global_step")
    if not isinstance(global_step, int) or global_step < 0:
        raise CheckpointError(
            f"{state_path} has an invalid global_step: {global_step!r}")

    shapes = {}
    for parameter, entry in zip(trainable, restored_optimizer.state.values()):
        for field_name, value in entry.items():
            if hasattr(value, "shape") and hasattr(parameter, "shape"):
                if field_name in ("exp_avg", "exp_avg_sq") and value.shape != parameter.shape:
                    raise CheckpointError(
                        f"Optimizer state {field_name} shape {tuple(value.shape)} "
                        f"does not match parameter shape {tuple(parameter.shape)}")
                shapes[field_name] = list(value.shape) if hasattr(value, "shape") else None
    if expected_optimizer_state is not None and not _optimizer_states_equal(
            expected_optimizer_state, restored_optimizer.state_dict()):
        raise CheckpointError(
            f"Restored optimizer state in {checkpoint_dir} does not match the "
            "expected state")
    restored_base_lrs = [group.get("lr") for group in restored_optimizer.param_groups]
    scheduler_report = verify_scheduler_state(
        saved_scheduler_state, restored_scheduler.state_dict(),
        fresh_state=fresh_scheduler_state, expected_base_lrs=restored_base_lrs,
        expected_warmup_steps=plan["num_warmup_steps"],
        expected_training_steps=plan["num_training_steps"])
    return {
        "verified": True,
        "global_step": global_step,
        "epoch": state.get("epoch"),
        "optimizer_params_with_state": len(restored_optimizer.state),
        "scheduler_last_epoch": scheduler_report["last_epoch"],
        "scheduler": scheduler_report,
        "scheduler_plan": plan,
        "initial_lr": initial_lr,
        "optimizer_state_matches_expected": expected_optimizer_state is not None,
    }


def _transformers_scheduler_factory():
    """The production scheduler factory: transformers cosine with warmup."""
    try:
        from transformers import get_scheduler
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise CheckpointError(
            "transformers is required to reconstruct the production cosine "
            "scheduler for resume verification") from exc

    def factory(optimizer, *, num_warmup_steps, num_training_steps):
        return get_scheduler("cosine", optimizer,
                             num_warmup_steps=num_warmup_steps,
                             num_training_steps=num_training_steps)

    return factory


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def list_checkpoints(run_dir, gate: str) -> list:
    """Checkpoint directories inside one gate namespace, ordered by step."""
    root = gate_checkpoints_dir(run_dir, gate)
    if not root.is_dir():
        return []
    found = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = CHECKPOINT_DIR_RE.fullmatch(path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort(key=lambda item: item[0])
    return [path for _, path in found]


def find_latest_checkpoint(run_dir, gate: str, identity_sha256: str, *,
                           model_id: str | None = None,
                           model_revision: str | None = None,
                           adapter: str | None = None,
                           method: str | None = None) -> tuple:
    """Newest valid checkpoint in this gate's namespace, or ``(None, {})``.

    The newest checkpoint is authoritative: if it is incomplete, foreign, from
    another gate or from another training method, the call raises instead of
    silently resuming an older one. Checkpoints of other gates and of the other
    training method are never visible here.
    """
    checkpoints = list_checkpoints(run_dir, gate)
    if not checkpoints:
        return None, {}
    latest = checkpoints[-1]
    meta = verify_checkpoint(
        latest, identity_sha256, gate=gate, model_id=model_id,
        model_revision=model_revision, adapter=adapter, method=method)
    return latest, meta
