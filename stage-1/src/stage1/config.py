"""Strict Stage 1 configuration.

Design rules:

- Every unknown key is an error. There is no silent acceptance of typos.
- User-specific data/model paths must be absolute and are supplied by CLIs.
  Repository auxiliary files such as locks and chat templates may be relative
  and are resolved deterministically against the config location.
- Identity-bearing settings are separated from provenance-only settings. A
  resolved config exposes ``identity_payload()``; local paths are excluded
  because content hashes (dataset logical hash, split-manifest hash, lock file
  hash, tokenizer fingerprints) pin the actual inputs.
- Gate overrides are limited to run-shape knobs (row counts, epochs, logging
  and checkpoint frequencies, success criteria). Learning rate, batch sizes,
  sequence length, optimizer and splits are not gate-overridable, so the run
  identity stays meaningful across gates.
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ConfigError
from .util import canonical_digest, is_pinned_revision, require_absolute

SUPPORTED_ADAPTERS = ("qwen3.5", "minicpm5")
SUPPORTED_LOADERS = ("unsloth-language-model", "unsloth-vision-model")
TRAINING_METHODS = ("full", "lora")
LORA_BIASES = ("none", "all", "lora_only")
TRAINING_GATES = ("overfit-100", "smoke-1000", "full")
GATE_OVERRIDE_KEYS = (
    "max_rows", "epochs", "save_steps", "eval_steps", "logging_steps",
    "max_steps", "success_eval_loss", "min_loss_reduction",
    "generation_samples", "check_gradients",
)
OPTIMIZERS = ("adamw_torch_fused", "adamw_torch")
COMPILE_ENGINES = ("pdflatex", "lualatex", "xelatex")
MODEL_ID_RE = re.compile(r"[^/\s]+/[^/\s]+")

TOP_LEVEL_KEYS = (
    "inherit", "tool_version", "stage", "seed", "model", "tokenizer", "data",
    "training", "lora", "gates", "evaluation", "hardware", "environment",
)


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------


def _kind(value) -> str:
    return type(value).__name__


def _expect_mapping(value, path: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: expected a mapping, got {_kind(value)}")
    return value


def _expect_scalar(value, path: str, types, *, allow_none: bool = False):
    if value is None:
        if allow_none:
            return None
        raise ConfigError(f"{path}: value is required")
    if isinstance(value, bool) and bool not in types:
        raise ConfigError(f"{path}: expected {'/'.join(t.__name__ for t in types)}, got bool")
    if not isinstance(value, types):
        expected = "/".join(t.__name__ for t in types)
        raise ConfigError(f"{path}: expected {expected}, got {_kind(value)}")
    return value


def _expect_int(value, path: str, *, minimum=None, allow_none=False):
    if value is None and allow_none:
        return None
    value = _expect_scalar(value, path, (int,))
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path}: must be >= {minimum}, got {value}")
    return value


def _expect_float(value, path: str, *, minimum=None, maximum=None,
                  exclusive_minimum=None, allow_none=False):
    if value is None and allow_none:
        return None
    value = _expect_scalar(value, path, (int, float))
    value = float(value)
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path}: must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{path}: must be <= {maximum}, got {value}")
    if exclusive_minimum is not None and value <= exclusive_minimum:
        raise ConfigError(f"{path}: must be > {exclusive_minimum}, got {value}")
    return value


def _expect_bool(value, path: str):
    return bool(_expect_scalar(value, path, (bool,)))


def _expect_str(value, path: str, *, allow_none=False, allow_empty=False):
    if value is None and allow_none:
        return None
    value = _expect_scalar(value, path, (str,))
    if not allow_empty and not value.strip():
        raise ConfigError(f"{path}: must not be empty")
    return value


def _check_keys(mapping: dict, allowed, path: str) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ConfigError(
            f"{path}: unknown key(s) {unknown}; allowed keys are {sorted(allowed)}")


def _optional_absolute(value, path: str):
    if value is None:
        return None
    return require_absolute(value, path)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    id: str
    revision: str
    adapter: str
    loader: str = "unsloth-language-model"
    local_path: Path | None = None
    trust_remote_code: bool = False
    attn_implementation: str | None = None

    def identity_payload(self) -> dict:
        return {
            "id": self.id,
            "revision": self.revision,
            "adapter": self.adapter,
            "loader": self.loader,
            "trust_remote_code": self.trust_remote_code,
            "attn_implementation": self.attn_implementation,
        }

    def resolved_attn_implementation(self, adapter_default: str | None) -> str | None:
        return self.attn_implementation or adapter_default


@dataclass
class TokenizerConfig:
    chat_template_file: Path | None = None
    chat_template_kwargs: dict = field(default_factory=dict)
    pad_token: str | None = None

    def identity_payload(self) -> dict:
        return {
            "chat_template_kwargs": dict(sorted(self.chat_template_kwargs.items())),
            "pad_token": self.pad_token,
        }


@dataclass
class SplitConfig:
    validation_fraction: float = 0.01
    test_fraction: float = 0.01


@dataclass
class DataConfig:
    export_dir: Path | None = None
    prepared_dir: Path | None = None
    max_seq_len: int = 4096
    require_complete_export: bool = True
    max_quarantined_fraction: float = 0.02
    splits: SplitConfig = field(default_factory=SplitConfig)

    def identity_payload(self) -> dict:
        return {
            "max_seq_len": self.max_seq_len,
            "require_complete_export": self.require_complete_export,
            "max_quarantined_fraction": self.max_quarantined_fraction,
            "splits": dataclasses.asdict(self.splits),
        }


@dataclass
class TrainingConfig:
    method: str = "full"
    epochs: int = 1
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    optim: str = "adamw_torch_fused"
    bf16: bool = True
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int | None = 3
    logging_steps: int = 10
    dataloader_num_workers: int = 2
    max_steps: int = -1
    validate_batches: bool = True
    eval_max_rows: int = 500
    report_to: list = field(default_factory=list)

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


@dataclass
class LoraConfig:
    """Native PEFT LoRA settings (BF16 base weights, no quantization)."""

    rank: int = 64
    alpha: int = 64
    dropout: float = 0.0
    bias: str = "none"
    target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    def to_jsonable(self) -> dict:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "bias": self.bias,
            "target_modules": list(self.target_modules),
        }


@dataclass
class GateConfig:
    """Resolved settings for one gate: base training config + overrides."""

    name: str
    max_rows: int | None
    epochs: int
    save_steps: int
    eval_steps: int
    logging_steps: int
    max_steps: int
    success_eval_loss: float | None
    min_loss_reduction: float | None
    generation_samples: int
    check_gradients: bool

    def to_jsonable(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class CompileConfig:
    engine: str = "pdflatex"
    timeout_seconds: int = 20
    render: bool = True
    keep_artifacts: bool = False


@dataclass
class RenderSimilarityConfig:
    enabled: bool = True
    size: int = 64


@dataclass
class EvaluationConfig:
    max_new_tokens: int = 1024
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 0.95
    batch_size: int = 8
    max_examples: int | None = 200
    duplicate_ngram_size: int = 8
    duplicate_overlap_threshold: float = 0.8
    memorization_sample: int = 5000
    compile: CompileConfig = field(default_factory=CompileConfig)
    render_similarity: RenderSimilarityConfig = field(default_factory=RenderSimilarityConfig)


HARDWARE_PROFILES = {
    # Full-parameter BF16 SFT on one RTX PRO 6000 Blackwell 96GB.
    "rtxpro6000-full": {
        "expected_gpu_name_regex": "RTX PRO 6000",
        "min_vram_gib": 88.0,
        "min_free_disk_gib": 250.0,
    },
    # BF16 LoRA on one RTX PRO 6000 Blackwell 96GB: adapters need far less
    # VRAM and disk than full finetuning.
    "rtxpro6000-lora": {
        "expected_gpu_name_regex": "RTX PRO 6000",
        "min_vram_gib": 60.0,
        "min_free_disk_gib": 60.0,
    },
    # BF16 LoRA on one A40 48GB, strictly within the 44-48 GiB range.
    "a40-lora": {
        "expected_gpu_name_regex": r"\bA40\b",
        "min_vram_gib": 44.0,
        "min_free_disk_gib": 60.0,
    },
}


@dataclass
class HardwareConfig:
    expected_gpu_name_regex: str = "RTX PRO 6000"
    min_vram_gib: float = 88.0
    min_free_disk_gib: float = 250.0
    device: str = "cuda:0"
    profile: str = "rtxpro6000-full"

    def to_jsonable(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class EnvironmentConfig:
    lock_file: str | None = None
    python_version: str = "3.12"
    lock_path: Path | None = None   # resolved; provenance-only
    lock_sha256: str | None = None  # content hash; identity-bearing

    def identity_payload(self) -> dict:
        if self.lock_file is None:
            raise ConfigError("environment.lock_file is required")
        return {
            "lock_file": self.lock_file,
            "lock_sha256": self.lock_sha256,
            "python_version": self.python_version,
        }


@dataclass
class Stage1Config:
    tool_version: str
    stage: int
    seed: int
    model: ModelConfig
    tokenizer: TokenizerConfig
    data: DataConfig
    training: TrainingConfig
    gates: dict
    evaluation: EvaluationConfig
    hardware: HardwareConfig
    environment: EnvironmentConfig
    lora: LoraConfig | None = None
    config_path: Path | None = None
    source_sha256: str | None = None

    def identity_payload(self) -> dict:
        return {
            "tool_version": self.tool_version,
            "stage": self.stage,
            "seed": self.seed,
            "model": self.model.identity_payload(),
            "tokenizer": self.tokenizer.identity_payload(),
            "data": self.data.identity_payload(),
            "training": dataclasses.asdict(self.training),
            "lora": self.lora.to_jsonable() if self.lora is not None else None,
            "evaluation": _evaluation_payload(self.evaluation),
            "environment": self.environment.identity_payload(),
        }

    def gate(self, name: str) -> GateConfig:
        return resolve_gate(self, name)

    def to_jsonable(self) -> dict:
        payload = _jsonable(dataclasses.asdict(self))
        payload["config_path"] = str(self.config_path) if self.config_path else None
        payload["source_sha256"] = self.source_sha256
        return payload


def _evaluation_payload(evaluation: EvaluationConfig) -> dict:
    payload = _jsonable(dataclasses.asdict(evaluation))
    return payload


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_config(raw: dict, *, source: str = "<dict>") -> Stage1Config:
    raw = _expect_mapping(raw, source)
    _check_keys(raw, TOP_LEVEL_KEYS, source)

    tool_version = _expect_str(raw.get("tool_version", "stage1-sft-v1"),
                               f"{source}.tool_version")
    stage = _expect_int(raw.get("stage", 1), f"{source}.stage", minimum=1)
    seed = _expect_int(raw.get("seed"), f"{source}.seed", minimum=0)

    model = _parse_model(_expect_mapping(raw.get("model"), f"{source}.model"), source)
    tokenizer = _parse_tokenizer(
        _expect_mapping(raw.get("tokenizer", {}), f"{source}.tokenizer"), source)
    data = _parse_data(_expect_mapping(raw.get("data", {}), f"{source}.data"), source)
    training = _parse_training(
        _expect_mapping(raw.get("training", {}), f"{source}.training"), source)
    gates = _parse_gates(_expect_mapping(raw.get("gates", {}), f"{source}.gates"), source)
    evaluation = _parse_evaluation(
        _expect_mapping(raw.get("evaluation", {}), f"{source}.evaluation"), source)
    hardware = _parse_hardware(
        _expect_mapping(raw.get("hardware", {}), f"{source}.hardware"), source)
    environment = _parse_environment(
        _expect_mapping(raw.get("environment", {}), f"{source}.environment"), source)
    lora = _parse_lora(raw.get("lora"), source)

    if training.method == "lora" and lora is None:
        raise ConfigError(
            f"{source}.lora: a LoRA configuration is required when "
            "training.method is 'lora'")
    if training.method == "full" and lora is not None:
        raise ConfigError(
            f"{source}.lora: a LoRA configuration is only valid with "
            "training.method: lora; remove it or set training.method: lora")
    expected_batch = 16
    if training.effective_batch_size != expected_batch:
        raise ConfigError(
            f"{source}.training: effective batch size is "
            f"per_device_train_batch_size * gradient_accumulation_steps = "
            f"{training.effective_batch_size}, but this task requires "
            f"{expected_batch} on one A40. Adjust the two values accordingly "
            "(for example 2 x 8).")

    return Stage1Config(
        tool_version=tool_version, stage=stage, seed=seed, model=model,
        tokenizer=tokenizer, data=data, training=training, gates=gates,
        evaluation=evaluation, hardware=hardware, environment=environment,
        lora=lora)


def _parse_model(raw: dict, source: str) -> ModelConfig:
    path = f"{source}.model"
    _check_keys(raw, ("id", "revision", "adapter", "loader", "local_path",
                      "trust_remote_code", "attn_implementation"), path)
    model_id = _expect_str(raw.get("id"), f"{path}.id")
    if not MODEL_ID_RE.fullmatch(model_id):
        raise ConfigError(f"{path}.id: expected 'owner/name', got {model_id!r}")
    revision = _expect_str(raw.get("revision"), f"{path}.revision")
    if not is_pinned_revision(revision):
        raise ConfigError(
            f"{path}.revision: expected a 40-character commit SHA, got {revision!r}. "
            "Stage 1 refuses branch names: pin the exact revision.")
    adapter = _expect_str(raw.get("adapter"), f"{path}.adapter")
    if adapter not in SUPPORTED_ADAPTERS:
        raise ConfigError(
            f"{path}.adapter: {adapter!r} is not supported; choose one of "
            f"{list(SUPPORTED_ADAPTERS)}")
    loader = _expect_str(raw.get("loader", "unsloth-language-model"), f"{path}.loader")
    if loader not in SUPPORTED_LOADERS:
        raise ConfigError(
            f"{path}.loader: {loader!r} is not supported; choose one of "
            f"{list(SUPPORTED_LOADERS)}")
    local_path = _optional_absolute(raw.get("local_path"), f"{path}.local_path")
    trust_remote_code = _expect_bool(raw.get("trust_remote_code", False),
                                     f"{path}.trust_remote_code")
    attn = raw.get("attn_implementation")
    if attn is not None:
        attn = _expect_str(attn, f"{path}.attn_implementation")
    return ModelConfig(
        id=model_id, revision=revision, adapter=adapter, loader=loader,
        local_path=local_path, trust_remote_code=trust_remote_code,
        attn_implementation=attn)


def _parse_tokenizer(raw: dict, source: str) -> TokenizerConfig:
    path = f"{source}.tokenizer"
    _check_keys(raw, ("chat_template_file", "chat_template_kwargs", "pad_token"), path)
    template_value = raw.get("chat_template_file")
    template = None
    if template_value is not None:
        template = Path(_expect_str(
            template_value, f"{path}.chat_template_file"))
    kwargs = _expect_mapping(raw.get("chat_template_kwargs", {}),
                             f"{path}.chat_template_kwargs")
    for key in kwargs:
        if not isinstance(key, str):
            raise ConfigError(f"{path}.chat_template_kwargs: keys must be strings")
    pad_token = raw.get("pad_token")
    if pad_token is not None:
        pad_token = _expect_str(pad_token, f"{path}.pad_token")
    return TokenizerConfig(chat_template_file=template,
                           chat_template_kwargs=dict(kwargs), pad_token=pad_token)


def _parse_data(raw: dict, source: str) -> DataConfig:
    path = f"{source}.data"
    _check_keys(raw, ("export_dir", "prepared_dir", "max_seq_len",
                      "require_complete_export", "max_quarantined_fraction",
                      "splits"), path)
    export_dir = _optional_absolute(raw.get("export_dir"), f"{path}.export_dir")
    prepared_dir = _optional_absolute(raw.get("prepared_dir"), f"{path}.prepared_dir")
    max_seq_len = _expect_int(raw.get("max_seq_len", 4096), f"{path}.max_seq_len",
                              minimum=128)
    require_complete = _expect_bool(raw.get("require_complete_export", True),
                                    f"{path}.require_complete_export")
    max_quarantined = _expect_float(raw.get("max_quarantined_fraction", 0.02),
                                    f"{path}.max_quarantined_fraction",
                                    minimum=0.0, maximum=1.0)
    splits_raw = _expect_mapping(raw.get("splits", {}), f"{path}.splits")
    _check_keys(splits_raw, ("validation_fraction", "test_fraction"),
                f"{path}.splits")
    validation = _expect_float(splits_raw.get("validation_fraction", 0.01),
                               f"{path}.splits.validation_fraction",
                               minimum=0.0, maximum=0.99)
    test = _expect_float(splits_raw.get("test_fraction", 0.01),
                         f"{path}.splits.test_fraction", minimum=0.0, maximum=0.99)
    if validation + test >= 1.0:
        raise ConfigError(
            f"{path}.splits: validation_fraction + test_fraction must be < 1, "
            f"got {validation + test}")
    return DataConfig(export_dir=export_dir, prepared_dir=prepared_dir,
                      max_seq_len=max_seq_len,
                      require_complete_export=require_complete,
                      max_quarantined_fraction=max_quarantined,
                      splits=SplitConfig(validation_fraction=validation,
                                         test_fraction=test))


def _parse_training(raw: dict, source: str) -> TrainingConfig:
    path = f"{source}.training"
    allowed = ("method", "epochs", "learning_rate", "lr_scheduler_type",
               "warmup_ratio",
               "max_grad_norm", "per_device_train_batch_size",
               "per_device_eval_batch_size", "gradient_accumulation_steps",
               "optim", "bf16", "gradient_checkpointing",
               "gradient_checkpointing_use_reentrant", "packing", "save_steps",
               "eval_steps", "save_total_limit", "logging_steps",
               "dataloader_num_workers", "max_steps", "validate_batches",
               "eval_max_rows", "report_to")
    _check_keys(raw, allowed, path)
    method = _expect_str(raw.get("method", "full"), f"{path}.method")
    if method not in TRAINING_METHODS:
        raise ConfigError(
            f"{path}.method: {method!r} is not supported; choose one of "
            f"{list(TRAINING_METHODS)}")
    epochs = _expect_int(raw.get("epochs", 1), f"{path}.epochs", minimum=1)
    learning_rate = _expect_float(raw.get("learning_rate", 1e-5),
                                  f"{path}.learning_rate",
                                  exclusive_minimum=0.0, maximum=1e-2)
    scheduler = _expect_str(raw.get("lr_scheduler_type", "cosine"),
                            f"{path}.lr_scheduler_type")
    if scheduler != "cosine":
        raise ConfigError(
            f"{path}.lr_scheduler_type: Stage 1 pins the cosine schedule; "
            f"got {scheduler!r}")
    warmup = _expect_float(raw.get("warmup_ratio", 0.03), f"{path}.warmup_ratio",
                           minimum=0.0, maximum=0.5)
    max_grad_norm = _expect_float(raw.get("max_grad_norm", 1.0),
                                  f"{path}.max_grad_norm", exclusive_minimum=0.0)
    train_batch = _expect_int(raw.get("per_device_train_batch_size", 2),
                              f"{path}.per_device_train_batch_size", minimum=1)
    eval_batch = _expect_int(raw.get("per_device_eval_batch_size", 2),
                             f"{path}.per_device_eval_batch_size", minimum=1)
    accumulation = _expect_int(raw.get("gradient_accumulation_steps", 8),
                               f"{path}.gradient_accumulation_steps", minimum=1)
    optim = _expect_str(raw.get("optim", "adamw_torch_fused"), f"{path}.optim")
    if optim not in OPTIMIZERS:
        raise ConfigError(
            f"{path}.optim: {optim!r} is not supported; choose one of "
            f"{list(OPTIMIZERS)}")
    bf16 = _expect_bool(raw.get("bf16", True), f"{path}.bf16")
    if not bf16:
        raise ConfigError(
            f"{path}.bf16: full-parameter BF16 SFT requires bf16: true; "
            "QLoRA, fp16 and quantized training are out of scope for Stage 1")
    gradient_checkpointing = _expect_bool(raw.get("gradient_checkpointing", True),
                                          f"{path}.gradient_checkpointing")
    reentrant = _expect_bool(raw.get("gradient_checkpointing_use_reentrant", False),
                             f"{path}.gradient_checkpointing_use_reentrant")
    packing = _expect_bool(raw.get("packing", False), f"{path}.packing")
    if packing:
        raise ConfigError(
            f"{path}.packing: true is not supported. Stage 1 batches are "
            "right-padded only: resetting position_ids under a normal "
            "attention mask does not stop attention across packed examples, "
            "and a backend-specific varlen implementation has not been "
            "validated. Set training.packing: false.")
    save_steps = _expect_int(raw.get("save_steps", 200), f"{path}.save_steps",
                             minimum=1)
    eval_steps = _expect_int(raw.get("eval_steps", 200), f"{path}.eval_steps",
                             minimum=1)
    save_total_limit = raw.get("save_total_limit", 3)
    if save_total_limit is not None:
        save_total_limit = _expect_int(save_total_limit, f"{path}.save_total_limit",
                                       minimum=1)
    logging_steps = _expect_int(raw.get("logging_steps", 10),
                                f"{path}.logging_steps", minimum=1)
    workers = _expect_int(raw.get("dataloader_num_workers", 2),
                          f"{path}.dataloader_num_workers", minimum=0)
    max_steps = _expect_int(raw.get("max_steps", -1), f"{path}.max_steps",
                            minimum=-1)
    validate_batches = _expect_bool(raw.get("validate_batches", True),
                                    f"{path}.validate_batches")
    eval_max_rows = _expect_int(raw.get("eval_max_rows", 500),
                                f"{path}.eval_max_rows", minimum=1)
    report_to = raw.get("report_to", [])
    if not isinstance(report_to, list) or any(not isinstance(item, str) for item in report_to):
        raise ConfigError(f"{path}.report_to: expected a list of strings")
    return TrainingConfig(
        method=method, epochs=epochs, learning_rate=learning_rate,
        lr_scheduler_type=scheduler,
        warmup_ratio=warmup, max_grad_norm=max_grad_norm,
        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=eval_batch,
        gradient_accumulation_steps=accumulation, optim=optim, bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_use_reentrant=reentrant,
        save_steps=save_steps, eval_steps=eval_steps,
        save_total_limit=save_total_limit, logging_steps=logging_steps,
        dataloader_num_workers=workers, max_steps=max_steps,
        validate_batches=validate_batches, eval_max_rows=eval_max_rows,
        report_to=list(report_to))


def _parse_lora(raw, source: str):
    """Parse the ``lora`` section; ``None`` means no LoRA configuration."""
    path = f"{source}.lora"
    if raw is None:
        return None
    raw = _expect_mapping(raw, path)
    _check_keys(raw, ("rank", "alpha", "dropout", "bias", "target_modules"), path)
    rank = _expect_int(raw.get("rank", 64), f"{path}.rank", minimum=1)
    alpha = _expect_int(raw.get("alpha", 64), f"{path}.alpha", minimum=1)
    dropout = _expect_float(raw.get("dropout", 0.0), f"{path}.dropout",
                            minimum=0.0, maximum=0.5)
    bias = _expect_str(raw.get("bias", "none"), f"{path}.bias")
    if bias not in LORA_BIASES:
        raise ConfigError(
            f"{path}.bias: {bias!r} is not supported; choose one of "
            f"{list(LORA_BIASES)}")
    targets = raw.get("target_modules")
    if targets is None:
        targets = list(LoraConfig().target_modules)
    if not isinstance(targets, (list, tuple)) or not targets:
        raise ConfigError(
            f"{path}.target_modules: expected a non-empty list of module names")
    cleaned = []
    for index, value in enumerate(targets):
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(
                f"{path}.target_modules[{index}]: expected a non-empty string")
        cleaned.append(value.strip())
    duplicates = sorted({name for name in cleaned if cleaned.count(name) > 1})
    if duplicates:
        raise ConfigError(
            f"{path}.target_modules contains duplicate names: {duplicates}")
    return LoraConfig(rank=rank, alpha=alpha, dropout=dropout, bias=bias,
                      target_modules=tuple(cleaned))


def _parse_gates(raw: dict, source: str) -> dict:
    path = f"{source}.gates"
    unknown = sorted(set(raw) - set(TRAINING_GATES))
    if unknown:
        raise ConfigError(
            f"{path}: unknown gate(s) {unknown}; known gates are {list(TRAINING_GATES)}")
    gates = {}
    for name, overrides in raw.items():
        override_path = f"{path}.{name}"
        overrides = _expect_mapping(overrides, override_path)
        _check_keys(overrides, GATE_OVERRIDE_KEYS, override_path)
        gates[name] = dict(overrides)
    return gates


def _parse_evaluation(raw: dict, source: str) -> EvaluationConfig:
    path = f"{source}.evaluation"
    _check_keys(raw, ("max_new_tokens", "do_sample", "temperature", "top_p",
                      "batch_size", "max_examples", "duplicate_ngram_size",
                      "duplicate_overlap_threshold", "memorization_sample",
                      "compile", "render_similarity"), path)
    max_new_tokens = _expect_int(raw.get("max_new_tokens", 1024),
                                 f"{path}.max_new_tokens", minimum=1)
    do_sample = _expect_bool(raw.get("do_sample", False), f"{path}.do_sample")
    temperature = _expect_float(raw.get("temperature", 1.0), f"{path}.temperature",
                                exclusive_minimum=0.0)
    top_p = _expect_float(raw.get("top_p", 0.95), f"{path}.top_p",
                          exclusive_minimum=0.0, maximum=1.0)
    batch_size = _expect_int(raw.get("batch_size", 8), f"{path}.batch_size",
                             minimum=1)
    max_examples = raw.get("max_examples", 200)
    if max_examples is not None:
        max_examples = _expect_int(max_examples, f"{path}.max_examples", minimum=1)
    ngram = _expect_int(raw.get("duplicate_ngram_size", 8),
                        f"{path}.duplicate_ngram_size", minimum=2)
    threshold = _expect_float(raw.get("duplicate_overlap_threshold", 0.8),
                              f"{path}.duplicate_overlap_threshold",
                              exclusive_minimum=0.0, maximum=1.0)
    sample = _expect_int(raw.get("memorization_sample", 5000),
                         f"{path}.memorization_sample", minimum=0)

    compile_raw = _expect_mapping(raw.get("compile", {}), f"{path}.compile")
    _check_keys(compile_raw, ("engine", "timeout_seconds", "render", "keep_artifacts"),
                f"{path}.compile")
    engine = _expect_str(compile_raw.get("engine", "pdflatex"), f"{path}.compile.engine")
    if engine not in COMPILE_ENGINES:
        raise ConfigError(
            f"{path}.compile.engine: {engine!r} is not supported; choose one of "
            f"{list(COMPILE_ENGINES)}")
    timeout = _expect_int(compile_raw.get("timeout_seconds", 20),
                          f"{path}.compile.timeout_seconds", minimum=1)
    render = _expect_bool(compile_raw.get("render", True), f"{path}.compile.render")
    keep = _expect_bool(compile_raw.get("keep_artifacts", False),
                        f"{path}.compile.keep_artifacts")

    similarity_raw = _expect_mapping(raw.get("render_similarity", {}),
                                     f"{path}.render_similarity")
    _check_keys(similarity_raw, ("enabled", "size"), f"{path}.render_similarity")
    similarity_enabled = _expect_bool(similarity_raw.get("enabled", True),
                                      f"{path}.render_similarity.enabled")
    similarity_size = _expect_int(similarity_raw.get("size", 64),
                                  f"{path}.render_similarity.size", minimum=8)
    return EvaluationConfig(
        max_new_tokens=max_new_tokens, do_sample=do_sample,
        temperature=temperature, top_p=top_p, batch_size=batch_size,
        max_examples=max_examples, duplicate_ngram_size=ngram,
        duplicate_overlap_threshold=threshold, memorization_sample=sample,
        compile=CompileConfig(engine=engine, timeout_seconds=timeout,
                              render=render, keep_artifacts=keep),
        render_similarity=RenderSimilarityConfig(enabled=similarity_enabled,
                                                 size=similarity_size))


def _parse_hardware(raw: dict, source: str) -> HardwareConfig:
    path = f"{source}.hardware"
    _check_keys(raw, ("profile", "expected_gpu_name_regex", "min_vram_gib",
                      "min_free_disk_gib", "device"), path)
    profile = _expect_str(raw.get("profile", "rtxpro6000-full"),
                          f"{path}.profile")
    if profile not in HARDWARE_PROFILES:
        raise ConfigError(
            f"{path}.profile: {profile!r} is not a known hardware profile; "
            f"choose one of {sorted(HARDWARE_PROFILES)}")
    defaults = HARDWARE_PROFILES[profile]
    pattern = _expect_str(
        raw.get("expected_gpu_name_regex", defaults["expected_gpu_name_regex"]),
        f"{path}.expected_gpu_name_regex")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ConfigError(f"{path}.expected_gpu_name_regex: invalid regex: {exc}")
    min_vram = _expect_float(raw.get("min_vram_gib", defaults["min_vram_gib"]),
                             f"{path}.min_vram_gib", exclusive_minimum=0.0)
    min_disk = _expect_float(
        raw.get("min_free_disk_gib", defaults["min_free_disk_gib"]),
        f"{path}.min_free_disk_gib", exclusive_minimum=0.0)
    device = _expect_str(raw.get("device", "cuda:0"), f"{path}.device")
    return HardwareConfig(expected_gpu_name_regex=pattern, min_vram_gib=min_vram,
                          min_free_disk_gib=min_disk, device=device,
                          profile=profile)


def _parse_environment(raw: dict, source: str) -> EnvironmentConfig:
    path = f"{source}.environment"
    _check_keys(raw, ("lock_file", "python_version"), path)
    lock_file = _expect_str(raw.get("lock_file"), f"{path}.lock_file")
    python_version = _expect_str(raw.get("python_version", "3.12"),
                                 f"{path}.python_version")
    if not re.fullmatch(r"3\.\d+", python_version):
        raise ConfigError(
            f"{path}.python_version: expected '3.x', got {python_version!r}")
    return EnvironmentConfig(lock_file=lock_file, python_version=python_version)


# ---------------------------------------------------------------------------
# Loading with inheritance
# ---------------------------------------------------------------------------


def _deep_merge(parent, child):
    if isinstance(parent, dict) and isinstance(child, dict):
        merged = dict(parent)
        for key, value in child.items():
            merged[key] = _deep_merge(parent.get(key), value)
        return merged
    return child


def _load_raw(path: Path, seen) -> dict:
    path = Path(path).resolve()
    if path in seen:
        chain = " -> ".join(str(item) for item in seen + (path,))
        raise ConfigError(f"Configuration inheritance cycle: {chain}")
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ConfigError(
            "PyYAML is required to load YAML configuration files; "
            "install it with 'pip install PyYAML'") from exc
    data = yaml.safe_load(path.read_text())
    if data is None:
        data = {}
    _expect_mapping(data, str(path))
    parent_name = data.pop("inherit", None)
    if parent_name is None:
        return data
    if not isinstance(parent_name, str):
        raise ConfigError(f"{path}: 'inherit' must be a file name")
    parent_path = path.parent / parent_name
    merged_parent = _load_raw(parent_path, seen + (path,))
    return _deep_merge(merged_parent, data)


def resolve_aux_file(config_path: Path, value: str) -> Path:
    """Resolve a relative auxiliary file (e.g. a lock) against the config.

    Candidates, in order: ``<config dir>/<value>`` and
    ``<config dir>/../<value>``. Exactly one candidate must exist so that
    shipped configs (``configs/*.yaml`` + ``locks/*.lock``) and copied configs
    both resolve without guessing.
    """
    config_path = Path(config_path).resolve()
    candidate = Path(value)
    if candidate.is_absolute():
        if not candidate.is_file():
            raise ConfigError(f"File not found: {candidate}")
        return candidate
    candidates = [config_path.parent / value, config_path.parent.parent / value]
    existing = [item for item in candidates if item.is_file()]
    if len(existing) == 1:
        return existing[0].resolve()
    if not existing:
        tried = ", ".join(str(item) for item in candidates)
        raise ConfigError(f"Cannot resolve {value!r} from {config_path}; tried: {tried}")
    raise ConfigError(
        f"{value!r} is ambiguous from {config_path}: matches {[str(i) for i in existing]}")


def load_config(path) -> Stage1Config:
    path = Path(path).resolve()
    raw = _load_raw(path, ())
    config = parse_config(raw, source=str(path))
    config.config_path = path
    config.source_sha256 = canonical_digest(raw)
    if config.tokenizer.chat_template_file:
        config.tokenizer.chat_template_file = resolve_aux_file(
            path, str(config.tokenizer.chat_template_file))
    if config.environment.lock_file:
        lock_path = resolve_aux_file(path, config.environment.lock_file)
        config.environment.lock_path = lock_path
        from .util import lock_sha256
        config.environment.lock_sha256 = lock_sha256(lock_path)
    return config


# ---------------------------------------------------------------------------
# Gates and TrainingArguments
# ---------------------------------------------------------------------------


def resolve_gate(config: Stage1Config, name: str) -> GateConfig:
    if name not in TRAINING_GATES:
        raise ConfigError(
            f"Unknown gate {name!r}; training gates are {list(TRAINING_GATES)}. "
            "The data and preflight gates are separate commands.")
    overrides = config.gates.get(name, {})
    training = config.training

    def value(key, default):
        return overrides.get(key, default)

    max_rows = value("max_rows", None)
    if max_rows is not None:
        max_rows = _expect_int(max_rows, f"gates.{name}.max_rows", minimum=1)
    epochs = _expect_int(value("epochs", training.epochs), f"gates.{name}.epochs",
                         minimum=1)
    save_steps = _expect_int(value("save_steps", training.save_steps),
                             f"gates.{name}.save_steps", minimum=1)
    eval_steps = _expect_int(value("eval_steps", training.eval_steps),
                             f"gates.{name}.eval_steps", minimum=1)
    logging_steps = _expect_int(value("logging_steps", training.logging_steps),
                                f"gates.{name}.logging_steps", minimum=1)
    max_steps = _expect_int(value("max_steps", training.max_steps),
                            f"gates.{name}.max_steps", minimum=-1)
    success = value("success_eval_loss", None)
    if success is not None:
        success = _expect_float(success, f"gates.{name}.success_eval_loss",
                                exclusive_minimum=0.0)
    reduction = value("min_loss_reduction", None)
    if reduction is not None:
        reduction = _expect_float(reduction, f"gates.{name}.min_loss_reduction",
                                  minimum=0.0, maximum=1.0)
    generation_samples = _expect_int(value("generation_samples", 2),
                                     f"gates.{name}.generation_samples", minimum=1)
    check_gradients = _expect_bool(value("check_gradients", True),
                                   f"gates.{name}.check_gradients")
    return GateConfig(name=name, max_rows=max_rows, epochs=epochs,
                      save_steps=save_steps, eval_steps=eval_steps,
                      logging_steps=logging_steps, max_steps=max_steps,
                      success_eval_loss=success, min_loss_reduction=reduction,
                      generation_samples=generation_samples,
                      check_gradients=check_gradients)


def training_arguments_kwargs(config: Stage1Config, gate: GateConfig, *,
                              run_dir: Path, has_eval: bool,
                              run_name: str | None = None) -> dict:
    """Pure mapping from resolved config to transformers TrainingArguments.

    Kept free of torch/transformers imports so the mapping is unit-testable on
    a CPU-only machine. ``train.py`` calls ``TrainingArguments(**kwargs)``.
    """
    training = config.training
    checkpoint_dir = Path(run_dir) / "checkpoints" / gate.name
    kwargs = {
        "output_dir": str(checkpoint_dir),
        "overwrite_output_dir": False,
        "do_train": True,
        "do_eval": has_eval,
        "eval_strategy": "steps" if has_eval else "no",
        "eval_steps": gate.eval_steps,
        "save_strategy": "steps",
        "save_steps": gate.save_steps,
        "save_total_limit": training.save_total_limit,
        "logging_steps": gate.logging_steps,
        "logging_first_step": True,
        "learning_rate": training.learning_rate,
        "lr_scheduler_type": training.lr_scheduler_type,
        "warmup_ratio": training.warmup_ratio,
        "max_grad_norm": training.max_grad_norm,
        "num_train_epochs": gate.epochs,
        "max_steps": gate.max_steps,
        "per_device_train_batch_size": training.per_device_train_batch_size,
        "per_device_eval_batch_size": training.per_device_eval_batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "optim": training.optim,
        "bf16": True,
        "bf16_full_eval": True,
        "gradient_checkpointing": training.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {
            "use_reentrant": training.gradient_checkpointing_use_reentrant},
        "seed": config.seed,
        "data_seed": config.seed,
        "report_to": list(training.report_to),
        "run_name": run_name or f"stage1-{config.model.adapter}-{gate.name}",
        "include_num_input_tokens_seen": True,
        "dataloader_num_workers": training.dataloader_num_workers,
        "remove_unused_columns": False,
        "label_names": ["labels"],
    }
    return kwargs


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@dataclass
class RunPaths:
    export_dir: Path
    prepared_dir: Path
    run_dir: Path

    def validate_inputs(self) -> None:
        for label, path in (("export_dir", self.export_dir),
                            ("prepared_dir", self.prepared_dir)):
            if not path.is_dir():
                raise ConfigError(f"{label} does not exist: {path}")

    def ensure_run_dir(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)


def resolve_run_paths(config: Stage1Config, *, export=None, prepared=None,
                      run_dir=None) -> RunPaths:
    export_dir = require_absolute(
        export if export is not None else config.data.export_dir,
        "--export (or data.export_dir)")
    prepared_dir = require_absolute(
        prepared if prepared is not None else config.data.prepared_dir,
        "--prepared (or data.prepared_dir)")
    resolved_run = require_absolute(run_dir, "--run-dir")
    return RunPaths(export_dir=export_dir, prepared_dir=prepared_dir,
                    run_dir=resolved_run)
