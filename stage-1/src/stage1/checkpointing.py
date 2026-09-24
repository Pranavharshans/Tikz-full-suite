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

import math
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


def _tensors_equal(left, right) -> bool:
    """Duck-typed tensor/list equality for weight read-back checks."""
    if left is None or right is None:
        return False
    if hasattr(left, "shape"):
        if getattr(right, "shape", None) != left.shape:
            return False
        try:
            # Saved safetensors are read on CPU while the live PEFT adapter is
            # normally on CUDA. Compare values on the same device; treating a
            # cross-device comparison error as inequality creates a false
            # failure even when the numeric delta is exactly zero.
            left_value = left.detach().cpu() if hasattr(left, "detach") else left
            right_value = right.detach().cpu() if hasattr(right, "detach") else right
            return bool((left_value == right_value).all())
        except Exception:  # pragma: no cover - exotic tensor types
            return False
    return left == right


def normalize_adapter_key(key: str) -> str:
    """PEFT's adapter-key normalization: strip the PeftModel wrapper prefix.

    ``get_peft_model_state_dict`` returns keys relative to the wrapped base
    model while ``model.state_dict()`` keeps the ``base_model.model.`` prefix;
    both are valid spellings of the same adapter tensor. Normalizing both sides
    makes the comparison independent of which spelling a PEFT version emits.
    """
    for prefix in ("base_model.model.", "base_model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _tensor_dtype(value):
    dtype = getattr(value, "dtype", None)
    return None if dtype is None else str(dtype)


def compare_adapter_states(saved: dict, restored: dict) -> dict:
    """Compare normalized adapter states and report only genuine mismatches.

    Both mappings are keyed by PEFT's own adapter-state spelling. Base-model
    weights are absent from both by construction, so their omission can never
    be reported. Only adapter parameters that are missing from the artifact,
    tensors that are not part of the model's adapter, or shape/dtype/value
    differences are problems.
    """
    saved_norm = {normalize_adapter_key(key): value for key, value in saved.items()}
    restored_norm = {normalize_adapter_key(key): value
                     for key, value in restored.items()}
    problems = []
    only_restored = sorted(set(restored_norm) - set(saved_norm))
    only_saved = sorted(set(saved_norm) - set(restored_norm))
    for key in only_restored:
        problems.append(f"{key}: adapter parameter missing from the saved artifact")
    for key in only_saved:
        problems.append(f"{key}: saved tensor is not part of this model's adapter")
    compared = 0
    for key in sorted(set(saved_norm) & set(restored_norm)):
        left, right = saved_norm[key], restored_norm[key]
        left_shape = getattr(left, "shape", None)
        right_shape = getattr(right, "shape", None)
        if (left_shape is None) != (right_shape is None):
            problems.append(
                f"{key}: shape mismatch saved={left_shape} restored={right_shape}")
            continue
        if left_shape is not None and tuple(left_shape) != tuple(right_shape):
            problems.append(
                f"{key}: shape mismatch saved={tuple(left_shape)} "
                f"restored={tuple(right_shape)}")
            continue
        left_dtype, right_dtype = _tensor_dtype(left), _tensor_dtype(right)
        if left_dtype is not None and right_dtype is not None and left_dtype != right_dtype:
            problems.append(
                f"{key}: dtype mismatch saved={left_dtype} restored={right_dtype}")
            continue
        if not _tensors_equal(left, right):
            problems.append(f"{key}: values differ after the PEFT restore")
            continue
        compared += 1
    return {
        "problems": problems,
        "compared": compared,
        "adapter_parameters": len(restored_norm),
        "missing_from_artifact": only_restored,
        "not_part_of_adapter": only_saved,
    }


def _peft_adapter_apis():
    try:
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise CheckpointError(
            "peft is required to restore LoRA adapter state through its "
            "supported API; install the pinned environment from "
            "environment.lock_file") from exc
    return set_peft_model_state_dict, get_peft_model_state_dict


def load_artifact_weights(model, directory, *, method: str = "full",
                          loader=None, adapter_state_setter=None,
                          adapter_state_getter=None,
                          adapter_name: str = "default") -> dict:
    """Restore an artifact's weights into the model, exactly and method-aware.

    Full artifacts load the complete model state dict. LoRA artifacts are
    restored through PEFT's supported adapter-state APIs: the saved tensors are
    applied with ``set_peft_model_state_dict`` and then read back with
    ``get_peft_model_state_dict`` using the same ``adapter_name``; the
    normalized adapter tensors (keys, shapes, dtypes, values) must match.
    Base-model weight omissions and PEFT key-prefix normalization are expected
    and are never treated as errors - only genuine adapter incompatibilities
    fail. ``loader``, ``adapter_state_setter`` and ``adapter_state_getter`` are
    injection seams for tests.
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

    if method == "full":
        missing, unexpected = model.load_state_dict(saved, strict=False)
        if missing or unexpected:
            raise CheckpointError(
                f"Artifact {directory} does not match the model: "
                f"missing={list(missing)[:3]}, unexpected={list(unexpected)[:3]}")
        return {"path": str(directory), "method": method, "tensors": len(saved),
                "restore_api": "Module.load_state_dict"}

    if not any("lora_" in key for key in saved):
        raise CheckpointError(
            f"{directory} contains no adapter tensors; refusing to restore it "
            "as a LoRA artifact")
    if adapter_state_setter is None or adapter_state_getter is None:
        default_setter, default_getter = _peft_adapter_apis()
        adapter_state_setter = adapter_state_setter or default_setter
        adapter_state_getter = adapter_state_getter or default_getter
    try:
        result = adapter_state_setter(model, saved, adapter_name=adapter_name)
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(
            f"PEFT failed to restore the adapter in {directory} with "
            f"adapter_name={adapter_name!r} "
            f"({type(exc).__name__}: {exc})") from exc
    try:
        restored = adapter_state_getter(model, adapter_name=adapter_name)
    except Exception as exc:
        raise CheckpointError(
            f"PEFT could not read the adapter state back from {directory} with "
            f"adapter_name={adapter_name!r} "
            f"({type(exc).__name__}: {exc})") from exc
    comparison = compare_adapter_states(saved, restored)
    if comparison["problems"]:
        detail = "\n".join(f"  - {line}" for line in comparison["problems"][:10])
        raise CheckpointError(
            f"Adapter in {directory} is incompatible with this model:\n{detail}")
    # The setter's missing keys cover base-model weights that adapter files
    # never contain; they are recorded for diagnostics, never rejected.
    setter_missing = list(getattr(result, "missing_keys", []) or [])
    setter_unexpected = list(getattr(result, "unexpected_keys", []) or [])
    return {
        "path": str(directory),
        "method": method,
        "tensors": len(saved),
        "adapter_name": adapter_name,
        "restore_api": "peft.set_peft_model_state_dict",
        "readback_api": "peft.get_peft_model_state_dict",
        "readback_verified": True,
        "compared_tensors": comparison["compared"],
        "adapter_parameters": comparison["adapter_parameters"],
        "setter_missing_keys": len(setter_missing),
        "setter_unexpected_keys": len(setter_unexpected),
    }


# ---------------------------------------------------------------------------
# Resume-state verification (torch injected)
# ---------------------------------------------------------------------------

RESUME_PLAN_FILE = "stage1-resume-plan.json"

# The exact weight-decay filter Hugging Face Trainer applies in
# ``get_decay_parameter_names`` (transformers 4.57): bias and every norm
# spelling. The parity test compares this against the installed Trainer.
DECAY_FORBIDDEN_NAME_PATTERNS = (
    r"bias", r"layernorm", r"rmsnorm",
    r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)")


def decay_parameter_names(model) -> list:
    """Names of parameters that receive weight decay, per the Trainer filter."""
    try:
        from torch import nn
        from transformers.trainer_pt_utils import get_parameter_names
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise CheckpointError(
            "torch and transformers are required to reconstruct the production "
            "optimizer grouping") from exc
    return get_parameter_names(
        model, [nn.LayerNorm], list(DECAY_FORBIDDEN_NAME_PATTERNS))


def build_optimizer(torch, model, *, learning_rate, fused, decay_names=None,
                    weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8):
    """The same named, two-group AdamW layout used by the production Trainer.

    The groups keep Hugging Face Trainer's exact layout (decay first, no-decay
    second, both present even when empty). A ``state_dict`` post-hook stamps
    ``param_names`` into ``optimizer.pt``, so every serialized state entry can
    be anchored to a model parameter instead of the flattened order; building
    the groups from ``named_parameters()`` directly would make PyTorch reject
    the mixed named/unnamed case an empty group creates.
    """
    if decay_names is None:
        decay_names = decay_parameter_names(model)
    decay_names = set(decay_names)
    named = [(name, parameter) for name, parameter in model.named_parameters()
             if parameter.requires_grad]
    groups = [
        {"params": [parameter for name, parameter in named
                    if name in decay_names],
         "weight_decay": weight_decay},
        {"params": [parameter for name, parameter in named
                    if name not in decay_names],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=learning_rate, fused=fused,
                                  betas=betas, eps=eps)
    _register_parameter_names(optimizer, named)
    return optimizer


def _register_parameter_names(optimizer, named) -> None:
    """Record each optimizer group entry's parameter name in ``state_dict()``.

    PyTorch 2.12 stores ``param_names`` natively when groups are built from
    ``named_parameters()``; stamping the packed state dict avoids the
    named/unnamed constructor restriction while keeping the same information
    inside ``optimizer.pt``.
    """
    register = getattr(optimizer, "register_state_dict_post_hook", None)
    if register is None:  # pragma: no cover - older torch fallback
        return
    names_by_id = {id(parameter): name for name, parameter in named}

    def add_parameter_names(optimizer, state_dict):
        groups = state_dict.get("param_groups")
        if not isinstance(groups, list):
            return state_dict
        for group, live_group in zip(groups, optimizer.param_groups):
            group["param_names"] = [
                names_by_id[id(parameter)] for parameter in live_group["params"]
                if id(parameter) in names_by_id]
        return state_dict

    register(add_parameter_names)


def write_resume_plan(directory, *, model, optimizer, num_training_steps,
                      num_warmup_steps):
    """Persist information LambdaLR intentionally omits from state_dict()."""
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = [[names[id(parameter)] for parameter in group["params"]]
              for group in optimizer.param_groups]
    write_json_atomic(Path(directory) / RESUME_PLAN_FILE, {
        "schema_version": 1, "scheduler": "cosine",
        "num_training_steps": int(num_training_steps),
        "num_warmup_steps": int(num_warmup_steps),
        "parameter_groups": groups,
    })


def optimizer_group_names(state) -> list | None:
    """Parameter names serialized by a names-aware optimizer, if present.

    ``build_optimizer`` stamps ``param_names`` into every group inside
    ``optimizer.pt``. A checkpoint carrying them was written by this code and
    must also carry the resume plan, so a missing sidecar can never silently
    downgrade verification.
    """
    if not isinstance(state, dict):
        return None
    groups = state.get("param_groups")
    if not isinstance(groups, list) or not groups:
        return None
    names = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("param_names"), list):
            return None
        names.append(list(group["param_names"]))
    return names


def _optimizer_from_plan(torch, model, saved, plan, *, learning_rate, fused):
    """Bind checkpoint groups to names, never to coincidental tensor shapes."""
    named = {name: parameter for name, parameter in model.named_parameters()
             if parameter.requires_grad}
    groups = plan.get("parameter_groups")
    if not isinstance(groups, list) or len(groups) != len(saved["param_groups"]):
        raise CheckpointError("Resume parameter-group metadata does not match optimizer")
    flat = [name for group in groups for name in group]
    if len(flat) != len(set(flat)) or set(flat) != set(named):
        raise CheckpointError("Resume parameter names do not match trainable model parameters")
    rebuilt = []
    for names, saved_group in zip(groups, saved["param_groups"]):
        if len(names) != len(saved_group["params"]):
            raise CheckpointError("Resume parameter-group size does not match optimizer")
        saved_names = saved_group.get("param_names")
        if saved_names is not None and list(saved_names) != list(names):
            raise CheckpointError(
                "Resume parameter names disagree between the optimizer state "
                "and the resume plan")
        rebuilt.append({"params": [named[name] for name in names]})
    return torch.optim.AdamW(rebuilt, lr=learning_rate, fused=fused)


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
                if not _tensors_equal(value, other):
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
    """Probe saved and freshly reconstructed lambdas at key steps.

    Informational only: PyTorch does not serialize the lambdas (``LambdaLR``
    saves ``None`` for plain functions and the callable's ``__dict__`` for
    objects, which is ``{}`` for the ``functools.partial`` Transformers uses),
    so a real checkpoint cannot be probed this way. Schedule identity comes
    from the resume plan; this probe only compares when a caller passes
    in-memory schedulers.
    """
    saved_lambdas = saved_state.get("lr_lambdas")
    fresh_lambdas = fresh_state.get("lr_lambdas")
    if (not isinstance(saved_lambdas, (list, tuple)) or not saved_lambdas
            or not isinstance(fresh_lambdas, (list, tuple))
            or len(fresh_lambdas) != len(saved_lambdas)):
        return {"compared": False, "match": None, "mismatched_steps": [],
                "reason": "lambda lists are missing or have different lengths"}
    if not all(callable(function) for function in saved_lambdas):
        return {"compared": False, "match": None, "mismatched_steps": [],
                "reason": ("PyTorch did not serialize the scheduler lambdas "
                           "(None/__dict__ entries); schedule identity is "
                           f"verified from {RESUME_PLAN_FILE}")}
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


def scheduler_current_lrs(scheduler) -> list | None:
    """Recompute the LR the restored schedule assigns at its restored step.

    Uses the scheduler's own lambdas, its saved ``base_lrs`` and its saved
    ``last_epoch`` (both restored by ``load_state_dict``); returns ``None``
    when they are not usable, so the check is skipped rather than claimed.
    """
    lambdas = getattr(scheduler, "lr_lambdas", None)
    base_lrs = getattr(scheduler, "base_lrs", None)
    last_epoch = getattr(scheduler, "last_epoch", None)
    if (not isinstance(lambdas, (list, tuple))
            or not isinstance(base_lrs, (list, tuple))
            or len(lambdas) != len(base_lrs) or last_epoch is None
            or not all(callable(function) for function in lambdas)):
        return None
    try:
        return [float(function(last_epoch)) * float(base)
                for function, base in zip(lambdas, base_lrs)]
    except Exception:
        return None


def verify_resume_state(checkpoint_dir, *, torch, model, learning_rate: float,
                        fused: bool, num_training_steps: int,
                        num_warmup_steps: int, device="cuda",
                        expected_optimizer_state: dict | None = None,
                        scheduler_factory=None, continuation=None) -> dict:
    """Restore optimizer, scheduler and trainer state from a checkpoint.

    The scheduler is reconstructed exactly as production builds it -
    ``transformers.get_scheduler("cosine", optimizer, num_warmup_steps=...,
    num_training_steps=...)`` - and its state is loaded and verified. The
    restore order is PyTorch's required one: optimizer, then scheduler
    construction, then the optimizer state, then the scheduler state
    (constructing a scheduler overwrites ``param_group["lr"]`` and would
    otherwise reset the restored LR to warmup step zero). The configured
    initial LR is compared against the saved scheduler ``base_lrs`` *before*
    the optimizer state is loaded; afterwards the restored ``initial_lr`` and
    current ``lr`` are checked against the scheduler checkpoint and the
    reconstructed schedule. Optimizer state is mapped per parameter, must be
    complete after step >= 1, and a names-aware checkpoint must carry the
    resume plan that pins the schedule (PyTorch does not serialize the
    lambdas). When ``expected_optimizer_state`` is supplied, the restored
    optimizer state must equal it field for field; ``continuation`` runs one
    further step on the restored objects and may raise ``CheckpointError``.
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
    resume_plan_path = checkpoint_dir / RESUME_PLAN_FILE
    resume_plan = read_json(resume_plan_path) if resume_plan_path.is_file() else None
    saved_group_names = optimizer_group_names(saved_optimizer_state)
    if resume_plan is None and saved_group_names is not None:
        # Names-aware groups only come from the corrected builder; such a
        # checkpoint must carry the schedule sidecar. Refuse instead of
        # silently falling back to an unverified reconstruction.
        raise CheckpointError(
            f"{checkpoint_dir} has a names-aware optimizer but no "
            f"{RESUME_PLAN_FILE}; refusing to verify its schedule and "
            "parameter mapping without the resume plan")
    if resume_plan is not None:
        for name, expected in (("scheduler", "cosine"),
                               ("num_training_steps", num_training_steps),
                               ("num_warmup_steps", num_warmup_steps)):
            if resume_plan.get(name) != expected:
                raise CheckpointError(
                    f"Resume schedule {name}: checkpoint={resume_plan.get(name)!r}, "
                    f"this run={expected!r}")
        plan["schedule_parameters_verified"] = True
        plan["proof"] = RESUME_PLAN_FILE

    # 3. Rebuild the production optimizer and scheduler, then load the states.
    # PyTorch requires the scheduler to be initialized before
    # optimizer.load_state_dict(): constructing LambdaLR after restoring the
    # optimizer immediately rewrites its current LR to warmup step zero and
    # corrupts the state we are trying to verify.
    try:
        if resume_plan is not None:
            restored_optimizer = _optimizer_from_plan(
                torch, model, saved_optimizer_state, resume_plan,
                learning_rate=learning_rate, fused=fused)
        else:
            # Legacy single-group probes can still be inspected, with explicit
            # unverified schedule identity. Production checkpoints write names.
            restored_optimizer = torch.optim.AdamW(trainable, lr=learning_rate,
                                                   fused=fused)
        if scheduler_factory is None:
            scheduler_factory = _transformers_scheduler_factory()
        restored_scheduler = scheduler_factory(
            restored_optimizer, num_warmup_steps=plan["num_warmup_steps"],
            num_training_steps=plan["num_training_steps"])
        fresh_scheduler_state = restored_scheduler.state_dict()
        restored_optimizer.load_state_dict(saved_optimizer_state)
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

    # 4. Parameter-mapped optimizer state: every entry must belong to an
    # optimized parameter with the right shapes, and after at least one step
    # every optimized parameter must carry a complete Adam state.
    optimized_parameters = [parameter
                            for group in restored_optimizer.param_groups
                            for parameter in group["params"]]
    state_fields = set()
    required_adam_fields = {"step", "exp_avg", "exp_avg_sq"}
    for parameter, entry in restored_optimizer.state.items():
        missing_fields = required_adam_fields - set(entry)
        if global_step >= 1 and missing_fields:
            raise CheckpointError(
                f"Optimizer state in {checkpoint_dir} is missing field(s) "
                f"{sorted(missing_fields)} for an optimized parameter after "
                f"step {global_step}")
        for field_name, value in entry.items():
            if hasattr(value, "shape") and hasattr(parameter, "shape"):
                state_fields.add(field_name)
                if field_name in ("exp_avg", "exp_avg_sq") and value.shape != parameter.shape:
                    raise CheckpointError(
                        f"Optimizer state {field_name} shape {tuple(value.shape)} "
                        f"does not match parameter shape {tuple(parameter.shape)}")
    if global_step >= 1 and len(restored_optimizer.state) != len(optimized_parameters):
        raise CheckpointError(
            f"Optimizer state in {checkpoint_dir} covers "
            f"{len(restored_optimizer.state)} of {len(optimized_parameters)} "
            f"optimized parameter(s) at global step {global_step}; refusing a "
            "partial optimizer state")
    if expected_optimizer_state is not None and not _optimizer_states_equal(
            expected_optimizer_state, restored_optimizer.state_dict()):
        raise CheckpointError(
            f"Restored optimizer state in {checkpoint_dir} does not match the "
            "expected state")

    # 5. Explicit base-vs-current LR checks. The optimizer state load restores
    # the saved param-group learning rates; the scheduler state load restores
    # last_epoch without touching them. Both the restored base LR
    # (``initial_lr``) and the current LR are compared against the scheduler
    # checkpoint and against the reconstructed schedule.
    restored_base_lrs = []
    missing_base_lrs = []
    for index, group in enumerate(restored_optimizer.param_groups):
        value = group.get("initial_lr")
        if value is None:
            missing_base_lrs.append(f"param_groups[{index}]")
        else:
            restored_base_lrs.append(float(value))
    if missing_base_lrs:
        raise CheckpointError(
            "Restored optimizer state has no initial_lr for "
            f"{missing_base_lrs}; the base learning rates cannot be verified "
            "against the scheduler checkpoint")
    restored_current_lrs = [float(group.get("lr"))
                            for group in restored_optimizer.param_groups]
    saved_last_lrs = saved_scheduler_state.get("_last_lr")
    if saved_last_lrs is not None:
        saved_current_lrs = [float(value) for value in saved_last_lrs]
        if restored_current_lrs != saved_current_lrs:
            raise CheckpointError(
                "Restored optimizer learning rates do not match the scheduler "
                f"checkpoint: optimizer={restored_current_lrs}, "
                f"scheduler_last_lr={saved_current_lrs}")
    expected_current_lrs = scheduler_current_lrs(restored_scheduler)
    if expected_current_lrs is not None:
        mismatched_lrs = [
            index for index, (actual, expected) in
            enumerate(zip(restored_current_lrs, expected_current_lrs))
            if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)]
        if mismatched_lrs:
            raise CheckpointError(
                "Restored optimizer learning rates do not match the "
                "reconstructed cosine schedule at step "
                f"{restored_scheduler.last_epoch}: "
                f"optimizer={restored_current_lrs}, "
                f"schedule={expected_current_lrs}")
    scheduler_report = verify_scheduler_state(
        saved_scheduler_state, restored_scheduler.state_dict(),
        fresh_state=fresh_scheduler_state,
        expected_base_lrs=restored_base_lrs,
        expected_warmup_steps=plan["num_warmup_steps"],
        expected_training_steps=plan["num_training_steps"])
    continuation_report = continuation(restored_optimizer, restored_scheduler) \
        if continuation is not None else None
    return {
        "verified": True,
        "global_step": global_step,
        "epoch": state.get("epoch"),
        "optimizer_params_with_state": len(restored_optimizer.state),
        "optimized_parameters": len(optimized_parameters),
        "optimizer_state_fields": sorted(state_fields),
        "scheduler_last_epoch": scheduler_report["last_epoch"],
        "scheduler": scheduler_report,
        "scheduler_plan": plan,
        "initial_lr": initial_lr,
        "restored_base_lrs": restored_base_lrs,
        "restored_current_lrs": restored_current_lrs,
        "optimizer_state_matches_expected": expected_optimizer_state is not None,
        "parameter_names_verified": resume_plan is not None,
        "continuation": continuation_report,
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
