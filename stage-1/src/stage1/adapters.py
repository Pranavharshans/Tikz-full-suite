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


def load_model_and_tokenizer(config, *, local_files_only: bool = False,
                             cache_dir=None, for_training: bool = False,
                             for_inference: bool = False,
                             source_override=None):
    """Load the exact checkpoint through Unsloth; never a silent fallback.

    Returns ``(model, tokenizer, load_report)``. The report records which
    loader and which full-finetuning path were used so the preflight evidence
    and run metadata can state it explicitly.
    """
    validate_config_against_adapter(config)
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
    critical = {"revision", "local_files_only", "cache_dir", "full_finetuning"}
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
            "Stage 1 is full-parameter BF16 SFT; fix the environment instead of "
            "training in another precision.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise DataError(f"No pad token available for {config.model.id}")

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

    report = {
        "loader": loader_name,
        "loader_module_version": getattr(unsloth, "__version__", "unknown"),
        "full_finetuning": True,
        "for_training_applied": for_training_applied,
        "for_inference_applied": for_inference_applied,
        "attn_implementation": attn or "backend-default",
        "dtype": embedding_dtype,
        "source": source,
        "revision": config.model.revision if source_override is None else None,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()
                                         if parameter.requires_grad),
        "rejected_kwargs": sorted(rejected),
    }
    return model, tokenizer, report


def load_model_for_evaluation(config, *, checkpoint_dir=None,
                              local_files_only: bool = False, cache_dir=None):
    """Load either the pinned base checkpoint or a full-FT checkpoint directory."""
    if checkpoint_dir is None:
        model, tokenizer, report = load_model_and_tokenizer(
            config, local_files_only=local_files_only, cache_dir=cache_dir,
            for_inference=True)
        report["source_kind"] = "base"
        return model, tokenizer, report
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise DataError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    model, tokenizer, report = load_model_and_tokenizer(
        config, local_files_only=True, cache_dir=cache_dir, for_inference=True,
        source_override=checkpoint_dir)
    report["source_kind"] = "checkpoint"
    report["checkpoint_dir"] = str(checkpoint_dir)
    return model, tokenizer, report
