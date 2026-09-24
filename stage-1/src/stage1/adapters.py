"""Model-specific loading kept small and declarative.

Everything data-driven lives in configuration; this module only holds the
built-in adapter facts that cannot be expressed as plain data (context length,
which Unsloth loader to use, capability detection) plus the two loading entry
points used by preparation, preflight, training and evaluation.

Compatibility is never assumed: the preflight must load the exact checkpoint
with the pinned environment before any training gate can run. This module
raises with the model id, revision and loader in the message so a failed load
is diagnosable.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

from .errors import DataError
from .util import sha256_file

TOKENIZER_FINGERPRINT_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "chat_template.jinja", "vocab.json", "merges.txt", "added_tokens.json",
)


@dataclass(frozen=True)
class ModelAdapter:
    name: str
    context_length: int
    default_attn_implementation: str | None
    template_kwargs: dict
    notes: str

    def to_jsonable(self) -> dict:
        return {
            "name": self.name,
            "context_length": self.context_length,
            "default_attn_implementation": self.default_attn_implementation,
            "template_kwargs": dict(self.template_kwargs),
            "notes": self.notes,
        }


ADAPTERS = {
    "qwen3.5": ModelAdapter(
        name="qwen3.5",
        context_length=262144,
        default_attn_implementation=None,
        template_kwargs={"enable_thinking": False},
        notes=("Multimodal Qwen3_5ForConditionalGeneration with hybrid "
               "linear/full attention; Stage 1 trains it text-only."),
    ),
    "minicpm5": ModelAdapter(
        name="minicpm5",
        context_length=131072,
        default_attn_implementation=None,
        template_kwargs={"enable_thinking": False},
        notes="Text-only LlamaForCausalLM; the text-only comparison baseline.",
    ),
}


def get_adapter(name: str) -> ModelAdapter:
    if name not in ADAPTERS:
        raise DataError(
            f"Unknown adapter {name!r}; known adapters are {sorted(ADAPTERS)}")
    return ADAPTERS[name]


def validate_config_against_adapter(config) -> ModelAdapter:
    adapter = get_adapter(config.model.adapter)
    if config.data.max_seq_len > adapter.context_length:
        raise DataError(
            f"data.max_seq_len={config.data.max_seq_len} exceeds the context "
            f"length {adapter.context_length} of {config.model.id} "
            f"({config.model.adapter})")
    return adapter


def _source(config) -> str:
    if config.model.local_path is not None:
        return str(config.model.local_path)
    return config.model.id


def _hub_kwargs(config, *, local_files_only: bool, cache_dir) -> dict:
    kwargs = {
        "trust_remote_code": config.model.trust_remote_code,
        "local_files_only": bool(local_files_only),
    }
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    if config.model.local_path is None:
        kwargs["revision"] = config.model.revision
    return kwargs


def load_tokenizer(config, *, local_files_only: bool = False, cache_dir=None):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataError(
            "transformers is required to load a tokenizer; install the pinned "
            "environment from environment.lock_file") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            _source(config), **_hub_kwargs(config, local_files_only=local_files_only,
                                          cache_dir=cache_dir))
    except Exception as exc:
        raise DataError(
            f"Failed to load the tokenizer for {config.model.id}@"
            f"{config.model.revision} ({type(exc).__name__}: {exc}). Verify the "
            "revision, network access, or point model.local_path at a local "
            "snapshot.") from exc
    if config.tokenizer.pad_token:
        tokenizer.pad_token = config.tokenizer.pad_token
    if tokenizer.pad_token_id is None:
        raise DataError(
            f"The tokenizer for {config.model.id} has no pad token; set "
            "tokenizer.pad_token in the config explicitly")
    return tokenizer


def tokenizer_fingerprint(tokenizer, template, config) -> dict:
    """Identity-bearing description of the exact tokenizer and template."""
    files = {}
    base = Path(str(getattr(tokenizer, "name_or_path", "")))
    if base.is_dir():
        for name in TOKENIZER_FINGERPRINT_FILES:
            candidate = base / name
            if candidate.is_file():
                files[name] = sha256_file(candidate)
    return {
        "model_id": config.model.id,
        "revision": config.model.revision,
        "adapter": config.model.adapter,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "chat_template_sha256": template.sha256,
        "chat_template_source": template.source,
        "files": files,
    }


def fingerprint_core(fingerprint: dict) -> dict:
    return {key: value for key, value in fingerprint.items()
            if key not in ("chat_template_source",)}


def compare_fingerprints(prepared: dict, actual: dict) -> list:
    """Differences that must refuse training (empty means identical)."""
    problems = []
    left, right = fingerprint_core(prepared), fingerprint_core(actual)
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            problems.append(
                f"tokenizer fingerprint {key}: prepared={left.get(key)!r} "
                f"actual={right.get(key)!r}")
    return problems


def _callable_parameters(function):
    try:
        return inspect.signature(function).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return {}


def _split_kwargs(function, kwargs: dict) -> tuple:
    parameters = _callable_parameters(function)
    accepts_any = any(parameter.kind is inspect.Parameter.VAR_KEYWORD
                      for parameter in parameters.values())
    if accepts_any:
        return dict(kwargs), {}
    accepted, rejected = {}, {}
    for key, value in kwargs.items():
        if key in parameters:
            accepted[key] = value
        else:
            rejected[key] = value
    return accepted, rejected


def module_short_names(model) -> dict:
    """Count module short names (last dotted component) in the loaded model."""
    counts = {}
    for name, _module in model.named_modules():
        if not name:
            continue
        short = name.rsplit(".", 1)[-1]
        counts[short] = counts.get(short, 0) + 1
    return counts


def validate_target_modules(available: dict, targets) -> dict:
    """Verify every configured LoRA target exists in the loaded model.

    ``available`` maps module short names to occurrence counts. Returns a
    report with the matched count per target; raises with the available module
    names when a target does not exist, so a typo or an architecture mismatch
    can never be silently ignored.
    """
    missing = [name for name in targets if available.get(name, 0) <= 0]
    if missing:
        candidates = sorted(name for name in available
                            if any(part in name for part in ("proj", "gate", "up", "down")))
        raise DataError(
            "LoRA target module(s) not found in the loaded model: "
            f"{missing}. Available projection-like modules: {candidates[:40]}. "
            "Fix lora.target_modules for this architecture; Stage 1 never "
            "silently ignores a configured target.")
    return {
        "targets": list(targets),
        "matched_modules": {name: int(available.get(name, 0)) for name in targets},
        "total_matched_modules": sum(int(available.get(name, 0)) for name in targets),
    }


def verify_lora_trainables(named_parameters, lora_config) -> dict:
    """Assert base parameters are frozen and only adapter parameters train.

    ``named_parameters`` is any iterable of ``(name, parameter)`` pairs (for
    example ``model.named_parameters()``); each parameter needs
    ``requires_grad`` and ``numel()``.
    """
    total = 0
    trainable = 0
    base_trainable = []
    adapter_trainable = []
    adapter_targets = {}
    for name, parameter in named_parameters:
        count = int(parameter.numel())
        total += count
        if not parameter.requires_grad:
            continue
        trainable += count
        if "lora_" in name:
            adapter_trainable.append(name)
            for target in lora_config.target_modules:
                if f".{target}." in name or name.endswith(f".{target}"):
                    adapter_targets[target] = adapter_targets.get(target, 0) + 1
        elif lora_config.bias != "none" and name.endswith(".bias"):
            # bias: all/lora_only legitimately train biases.
            adapter_trainable.append(name)
        else:
            base_trainable.append(name)
    if base_trainable:
        raise DataError(
            f"{len(base_trainable)} base parameter(s) are trainable in LoRA mode; "
            f"only adapter parameters may train. First: {base_trainable[:3]}")
    if not adapter_trainable:
        raise DataError(
            "No trainable adapter parameters were found after applying LoRA; "
            "the adapter was not attached")
    absent = [target for target in lora_config.target_modules
              if adapter_targets.get(target, 0) == 0]
    if absent and lora_config.bias == "none":
        raise DataError(
            f"Configured LoRA targets without trainable adapter parameters: "
            f"{absent}; refusing to train a partially attached adapter")
    percentage = (100.0 * trainable / total) if total else 0.0
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": round(percentage, 6),
        "adapter_parameters": len(adapter_trainable),
        "adapter_parameter_names_sample": sorted(adapter_trainable)[:5],
        "adapter_targets": dict(sorted(adapter_targets.items())),
        "base_parameters_trainable": 0,
        "base_frozen": True,
    }


def _parameter_summary(model) -> dict:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    return {
        "parameter_count": total,
        "trainable_parameter_count": trainable,
        "trainable_percentage": round(100.0 * trainable / total, 6) if total else 0.0,
    }


def load_model_and_tokenizer(config, *, local_files_only: bool = False,
                             cache_dir=None, for_training: bool = False,
                             for_inference: bool = False,
                             source_override=None, method: str | None = None,
                             with_adapter: bool = True):
    """Load the exact checkpoint through Unsloth; never a silent fallback.

    ``method`` defaults to ``config.training.method``. ``"full"`` keeps the
    existing full-parameter BF16 path; ``"lora"`` loads the same unquantized
    BF16 base model and applies adapters through Unsloth's native PEFT
    integration (``get_peft_model``). With ``with_adapter=False`` the pinned
    base checkpoint is returned without any adapter - use
    :func:`load_base_model_and_tokenizer` for that so callers never create a
    fresh adapter that would then collide with a saved one. Returns
    ``(model, tokenizer, report)``.
    """
    validate_config_against_adapter(config)
    method = method or config.training.method
    if method not in ("full", "lora"):
        raise DataError(f"Unknown training method {method!r}")
    if with_adapter and method == "lora" and config.lora is None:
        raise DataError(
            "training.method is 'lora' but no lora configuration is present")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataError(
            "torch is required to load a model; install the pinned environment "
            "from environment.lock_file") from exc
    try:
        import unsloth
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataError(
            "unsloth is required for Stage 1 training; install the pinned "
            "environment from environment.lock_file") from exc

    adapter = get_adapter(config.model.adapter)
    attn = config.model.attn_implementation or adapter.default_attn_implementation
    source = str(source_override) if source_override is not None else _source(config)
    hub_kwargs = _hub_kwargs(config, local_files_only=local_files_only,
                             cache_dir=cache_dir)
    if source_override is not None:
        hub_kwargs.pop("revision", None)

    load_kwargs = dict(hub_kwargs)
    load_kwargs.update({
        "max_seq_length": config.data.max_seq_len,
        "dtype": torch.bfloat16,
        "load_in_4bit": False,
        "load_in_8bit": False,
    })
    if attn is not None:
        load_kwargs["attn_implementation"] = attn

    if config.model.loader == "unsloth-vision-model":
        try:
            from unsloth import FastVisionModel as Loader
        except ImportError as exc:
            raise DataError(
                "unsloth.FastVisionModel is not available in the installed "
                f"unsloth ({getattr(unsloth, '__version__', 'unknown')}); "
                f"{config.model.id} requires model.loader: unsloth-vision-model") from exc
        loader_name = "FastVisionModel"
    else:
        from unsloth import FastLanguageModel as Loader
        loader_name = "FastLanguageModel"

    if method == "full" and with_adapter:
        full_finetuning_supported = "full_finetuning" in _callable_parameters(
            Loader.from_pretrained)
        if not full_finetuning_supported:
            raise DataError(
                f"Unsloth {loader_name}.from_pretrained in unsloth "
                f"{getattr(unsloth, '__version__', 'unknown')} does not expose "
                "full_finetuning; Stage 1 requires full-parameter training and "
                "will not fall back to PEFT/QLoRA silently")
        load_kwargs["full_finetuning"] = True

    accepted, rejected = _split_kwargs(Loader.from_pretrained, load_kwargs)
    critical = {"revision", "local_files_only", "cache_dir"}
    if method == "full" and with_adapter:
        critical.add("full_finetuning")
    dropped_critical = sorted(critical & set(rejected))
    if dropped_critical:
        raise DataError(
            f"Unsloth {loader_name}.from_pretrained does not accept "
            f"{dropped_critical}; refusing to load an unpinned or unverifiable "
            "checkpoint")
    try:
        model, tokenizer = Loader.from_pretrained(source, **accepted)
    except Exception as exc:
        raise DataError(
            f"Failed to load {config.model.id}@{config.model.revision} through "
            f"Unsloth {loader_name} ({type(exc).__name__}: {exc}). The preflight "
            "must prove this checkpoint loads before any training gate.") from exc

    embedding = model.get_input_embeddings()
    embedding_dtype = str(embedding.weight.dtype) if embedding is not None else "unknown"
    if embedding is not None and embedding.weight.dtype != torch.bfloat16:
        raise DataError(
            f"Model embedding dtype is {embedding_dtype}, expected bfloat16. "
            "Stage 1 is BF16 training; fix the environment instead of training "
            "in another precision.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise DataError(f"No pad token available for {config.model.id}")

    report = {
        "loader": loader_name,
        "loader_module_version": getattr(unsloth, "__version__", "unknown"),
        "training_method": method if with_adapter else "base",
        "adapter_applied": bool(with_adapter and method == "lora"),
        "full_finetuning": method == "full" and with_adapter,
        "quantization": "none",
        "load_in_4bit": False,
        "load_in_8bit": False,
        "attn_implementation": attn or "backend-default",
        "dtype": embedding_dtype,
        "source": source,
        "revision": config.model.revision if source_override is None else None,
        "base_model_id": config.model.id,
        "base_model_revision": config.model.revision,
        "rejected_kwargs": sorted(rejected),
    }

    if with_adapter and method == "lora":
        target_report = validate_target_modules(
            module_short_names(model), config.lora.target_modules)
        model, lora_applied = _apply_lora(model, Loader, config, report)
        report["lora"] = config.lora.to_jsonable()
        report["lora_applied"] = lora_applied
        report["target_modules"] = target_report
        report.update(verify_lora_trainables(model.named_parameters(), config.lora))

    training_patch_stage = None
    for_training_applied = False
    if for_training:
        for_training_fn = getattr(Loader, "for_training", None)
        if for_training_fn is None:
            raise DataError(
                f"Unsloth {loader_name}.for_training is not available in the "
                "installed unsloth; refusing to start training without the "
                "training-time patches")
        model = for_training_fn(model)
        for_training_applied = True
        training_patch_stage = "after_adapter" if method == "lora" else "base_model"

    for_inference_applied = False
    if for_inference:
        for_inference_fn = getattr(Loader, "for_inference", None)
        if for_inference_fn is None:
            raise DataError(
                f"Unsloth {loader_name}.for_inference is not available in the "
                "installed unsloth; refusing to evaluate without the "
                "inference-time patches")
        model = for_inference_fn(model)
        for_inference_applied = True

    report.update(_parameter_summary(model))
    report["for_training_applied"] = for_training_applied
    report["for_training_patch_stage"] = training_patch_stage
    report["for_inference_applied"] = for_inference_applied
    return model, tokenizer, report


def _apply_lora(model, Loader, config, report):
    """Attach LoRA through Unsloth's native PEFT integration."""
    get_peft_model = getattr(Loader, "get_peft_model", None)
    if get_peft_model is None:
        raise DataError(
            f"Unsloth {report['loader']}.get_peft_model is not available in the "
            "installed unsloth; Stage 1 uses the native PEFT integration and "
            "will not build adapter layers manually")
    lora = config.lora
    requested = {
        "r": lora.rank,
        "lora_alpha": lora.alpha,
        "lora_dropout": lora.dropout,
        "bias": lora.bias,
        "target_modules": list(lora.target_modules),
        "random_state": config.seed,
    }
    if config.training.gradient_checkpointing:
        requested["use_gradient_checkpointing"] = "unsloth"
    accepted, rejected = _split_kwargs(get_peft_model, requested)
    required = {"r", "lora_alpha", "lora_dropout", "bias", "target_modules"}
    missing = sorted(required - set(accepted))
    if missing:
        raise DataError(
            f"Unsloth {report['loader']}.get_peft_model does not accept "
            f"{missing}; refusing to attach a partially configured adapter")
    if config.training.gradient_checkpointing and "use_gradient_checkpointing" not in accepted:
        raise DataError(
            f"Unsloth {report['loader']}.get_peft_model does not accept "
            "use_gradient_checkpointing but training.gradient_checkpointing is "
            "true; refusing to train without gradient checkpointing")
    try:
        model = get_peft_model(model, **accepted)
    except Exception as exc:
        raise DataError(
            f"Failed to attach LoRA adapters for {config.model.id} through "
            f"Unsloth {report['loader']}.get_peft_model "
            f"({type(exc).__name__}: {exc})") from exc
    return model, {
        "api": f"{report['loader']}.get_peft_model",
        "requested_kwargs": sorted(accepted),
        "dropped_kwargs": sorted(rejected),
        "quantized": False,
    }


def load_base_model_and_tokenizer(config, **kwargs):
    """Load the pinned BF16 base checkpoint with no adapter.

    Evaluation, preflight reload and merge must never create a fresh adapter
    (a later ``attach_lora_adapter`` would then collide with it). This loader
    is the single entry point for that: it disables the LoRA attachment path
    and records ``training_method: "base"`` in the report.
    """
    kwargs["with_adapter"] = False
    return load_model_and_tokenizer(config, **kwargs)


def attach_lora_adapter(model, adapter_dir):
    """Attach a saved PEFT adapter to an already-loaded base model.

    Refuses a model that already carries an adapter, so a saved adapter is
    attached exactly once. The guard runs before the peft import so it holds
    even for a model from another adapter stack.
    """
    if getattr(model, "peft_config", None) is not None:
        raise DataError(
            "The loaded model already carries a PEFT adapter; refusing to "
            "attach a second one. Use load_base_model_and_tokenizer for the "
            "base checkpoint.")
    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataError(
            "peft is required to attach a saved LoRA adapter; install the "
            "pinned environment from environment.lock_file") from exc
    if isinstance(model, PeftModel):
        raise DataError(
            "The loaded model already carries a PEFT adapter; refusing to "
            "attach a second one. Use load_base_model_and_tokenizer for the "
            "base checkpoint.")
    adapter_dir = Path(adapter_dir).resolve()
    if not adapter_dir.is_dir():
        raise DataError(f"Adapter directory does not exist: {adapter_dir}")
    try:
        return PeftModel.from_pretrained(model, str(adapter_dir))
    except Exception as exc:
        raise DataError(
            f"Failed to attach the adapter at {adapter_dir} "
            f"({type(exc).__name__}: {exc})") from exc


def load_model_for_evaluation(config, *, checkpoint_dir=None, artifact_meta=None,
                              local_files_only: bool = False, cache_dir=None,
                              base_loader=None):
    """Load the pinned base model, optionally with a saved Stage 1 artifact.

    ``"full"`` artifacts are complete model directories. ``"lora"`` artifacts
    are PEFT adapters: the exact pinned BF16 base model is loaded **base-only**
    (no fresh adapter is created) and the saved adapter is attached exactly
    once. ``base_loader`` is an injection seam for tests; production uses
    :func:`load_base_model_and_tokenizer`.
    """
    base_loader = base_loader or load_base_model_and_tokenizer
    if checkpoint_dir is None:
        model, tokenizer, report = base_loader(
            config, local_files_only=local_files_only, cache_dir=cache_dir,
            for_inference=True)
        report["source_kind"] = "base"
        return model, tokenizer, report

    checkpoint_dir = Path(checkpoint_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise DataError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if artifact_meta is None:
        from . import checkpointing
        artifact_meta = checkpointing.read_artifact_meta(checkpoint_dir, required=True)
    method = artifact_meta.get("training_method")
    if method == "lora":
        model, tokenizer, report = base_loader(
            config, local_files_only=local_files_only, cache_dir=cache_dir,
            for_inference=True)
        model = attach_lora_adapter(model, checkpoint_dir)
        report.update({
            "source_kind": "lora_adapter",
            "training_method": "lora",
            "adapter_attached": "once",
            "checkpoint_dir": str(checkpoint_dir),
            "adapter": str(checkpoint_dir),
            "base_model_id": artifact_meta.get("base_model_id"),
            "base_model_revision": artifact_meta.get("base_model_revision"),
            **_parameter_summary(model),
        })
        return model, tokenizer, report
    if method != "full":
        raise DataError(
            f"Artifact {checkpoint_dir} has training_method={method!r}; "
            "refusing to guess how to load it")
    model, tokenizer, report = load_model_and_tokenizer(
        config, local_files_only=True, cache_dir=cache_dir, for_inference=True,
        source_override=checkpoint_dir, method="full")
    report["source_kind"] = "checkpoint"
    report["checkpoint_dir"] = str(checkpoint_dir)
    return model, tokenizer, report
