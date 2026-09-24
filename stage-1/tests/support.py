"""Shared synthetic fixtures for Stage 1 tests.

Nothing here downloads anything or needs a GPU. Optional dependencies
(pyarrow, PyYAML, Pillow, pdflatex, torch) are detected and tests skip
cleanly when they are missing, matching the cleaning pipeline's convention.
"""
import importlib.util
import json
import shutil
import struct
import sys
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage1 import config as config_module  # noqa: E402
from stage1 import formatting  # noqa: E402
from stage1.util import sha256_bytes, sha256_text  # noqa: E402

HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None
HAS_YAML = importlib.util.find_spec("yaml") is not None
HAS_PILLOW = importlib.util.find_spec("PIL") is not None
HAS_PDFLATEX = shutil.which("pdflatex") is not None
HAS_PDFTOPPM = shutil.which("pdftoppm") is not None
HAS_BASH = shutil.which("bash") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
try:
    import torch as _torch  # noqa: F401
    HAS_TORCH = True
    HAS_CUDA = _torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

requires_pyarrow = unittest.skipUnless(HAS_PYARROW, "pyarrow is not installed")
requires_yaml = unittest.skipUnless(HAS_YAML, "PyYAML is not installed")
requires_pillow = unittest.skipUnless(HAS_PILLOW, "Pillow is not installed")
requires_pdflatex = unittest.skipUnless(HAS_PDFLATEX, "pdflatex is not installed")
requires_torch = unittest.skipUnless(HAS_TORCH, "torch is not installed")
requires_transformers = unittest.skipUnless(HAS_TRANSFORMERS,
                                            "transformers is not installed")
requires_cuda = unittest.skipUnless(HAS_CUDA, "no CUDA device available")


# ---------------------------------------------------------------------------
# Minimal PNG encoder (stdlib only, readable by Pillow)
# ---------------------------------------------------------------------------


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))


def make_png_bytes(seed: int = 0, width: int = 8, height: int = 8) -> bytes:
    rows = []
    for y in range(height):
        rows.append(b"\x00" + bytes(((x * 7 + y * 3 + seed * 11) % 256)
                                    for x in range(width)))
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header)
            + _png_chunk(b"IDAT", zlib.compress(b"".join(rows)))
            + _png_chunk(b"IEND", b""))


# ---------------------------------------------------------------------------
# Fake tokenizer
# ---------------------------------------------------------------------------

FAKE_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{%- if message.role == 'user' %}<|im_start|>user\n{{ message.content }}<|im_end|>\n"
    "{%- elif message.role == 'assistant' %}<|im_start|>assistant\n<think>\n\n</think>\n\n{{ message.content }}<|im_end|>\n"
    "{%- elif message.role == 'system' %}<|im_start|>system\n{{ message.content }}<|im_end|>\n"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n<think>\n\n</think>\n\n{%- endif %}"
)


class FakeTokenizer:
    """Character-level tokenizer with a ChatML-like template and offsets."""

    def __init__(self, *, template: str | None = None, merge=None,
                 raise_on_offsets: bool = False, prefix_breaker: bool = False):
        self.chat_template = FAKE_TEMPLATE if template is None else template
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.padding_side = "right"
        self.name_or_path = "fake-tokenizer"
        self.merge = merge
        self.raise_on_offsets = raise_on_offsets
        self.prefix_breaker = prefix_breaker
        self.template_calls = []
        self._decoded = {}

    def __len__(self):
        return 512

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, chat_template=None,
                            **kwargs):
        self.template_calls.append({
            "messages": list(messages),
            "add_generation_prompt": add_generation_prompt,
            "kwargs": dict(kwargs),
            "chat_template": chat_template,
        })
        parts = ["<s>"]
        for message in messages:
            role, content = message["role"], message["content"]
            if role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif role == "assistant":
                parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n"
                             f"{content}<|im_end|>\n")
            elif role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        text = "".join(parts)
        if self.prefix_breaker and messages and messages[-1]["role"] == "assistant":
            text = text.replace("</think>", "</think >", 1)
        return text

    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=False, **kwargs):
        if self.raise_on_offsets and return_offsets_mapping:
            raise TypeError("return_offsets_mapping is not supported")
        input_ids, offsets = [], []
        index = 0
        while index < len(text):
            if self.merge is not None and index == self.merge[0]:
                token_id = 10_000_000 + index
                self._decoded[token_id] = text[index:self.merge[1]]
                input_ids.append(token_id)
                offsets.append((index, self.merge[1]))
                index = self.merge[1]
            else:
                token_id = ord(text[index])
                self._decoded[token_id] = text[index]
                input_ids.append(token_id)
                offsets.append((index, index + 1))
                index += 1
        if not return_offsets_mapping:
            return {"input_ids": input_ids}
        return {"input_ids": input_ids, "offset_mapping": offsets}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._decoded.get(int(token_id), "?") for token_id in ids)


def fake_tokenizer_factory(**kwargs):
    tokenizer = FakeTokenizer(**kwargs)

    def factory(config, *, local_files_only=False, cache_dir=None):
        return tokenizer

    return factory


def format_fake(row_id, instruction, tikz, tokenizer=None):
    tokenizer = tokenizer or FakeTokenizer()
    template = formatting.resolve_chat_template(tokenizer, config_module.TokenizerConfig())
    return formatting.format_example(
        tokenizer, row_id=row_id, instruction=instruction, tikz=tikz,
        template=template, kwargs={})


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


def deep_update(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def config_dict(**overrides) -> dict:
    base = {
        "tool_version": "stage1-sft-v1",
        "stage": 1,
        "seed": 20260923,
        "model": {
            "id": "test/Model-A",
            "revision": "a" * 40,
            "adapter": "minicpm5",
            "loader": "unsloth-language-model",
            "local_path": None,
            "trust_remote_code": False,
            "attn_implementation": None,
        },
        "tokenizer": {
            "chat_template_file": None,
            "chat_template_kwargs": {"enable_thinking": False},
            "pad_token": None,
        },
        "data": {
            "export_dir": None,
            "prepared_dir": None,
            "max_seq_len": 4096,
            "require_complete_export": True,
            "max_quarantined_fraction": 0.02,
            "splits": {"validation_fraction": 0.01, "test_fraction": 0.01},
        },
        "training": {
            "epochs": 1,
            "learning_rate": 1e-5,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "max_grad_norm": 1.0,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "optim": "adamw_torch_fused",
            "bf16": True,
            "gradient_checkpointing": True,
            "gradient_checkpointing_use_reentrant": False,
            "packing": False,
            "save_steps": 200,
            "eval_steps": 200,
            "save_total_limit": 3,
            "logging_steps": 10,
            "dataloader_num_workers": 2,
            "max_steps": -1,
            "validate_batches": True,
            "eval_max_rows": 500,
            "report_to": [],
        },
        "gates": {
            "overfit-100": {
                "max_rows": 100, "epochs": 40, "save_steps": 50,
                "eval_steps": 50, "logging_steps": 5, "success_eval_loss": 0.05,
                "min_loss_reduction": 0.9, "generation_samples": 1,
                "check_gradients": True,
            },
            "smoke-1000": {
                "max_rows": 1000, "epochs": 1, "save_steps": 25,
                "eval_steps": 25, "logging_steps": 10,
                "generation_samples": 2, "check_gradients": True,
            },
            "full": {
                "max_rows": None, "epochs": 1, "save_steps": 200,
                "eval_steps": 200, "logging_steps": 10,
                "generation_samples": 4, "check_gradients": False,
            },
        },
        "evaluation": {
            "max_new_tokens": 1024,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 0.95,
            "batch_size": 8,
            "max_examples": 200,
            "duplicate_ngram_size": 8,
            "duplicate_overlap_threshold": 0.8,
            "memorization_sample": 5000,
            "compile": {
                "engine": "pdflatex", "timeout_seconds": 20, "render": True,
                "keep_artifacts": False,
            },
            "render_similarity": {"enabled": True, "size": 64},
        },
        "hardware": {
            # Mirrors configs/common.yaml: the profile supplies the values.
            "profile": "rtxpro6000-full",
            "device": "cuda:0",
        },
        "environment": {"lock_file": "locks/test.lock", "python_version": "3.12"},
    }
    return deep_update(base, overrides)


def make_config(**overrides):
    return config_module.parse_config(config_dict(**overrides), source="<test>")


# ---------------------------------------------------------------------------
# Export fixtures
# ---------------------------------------------------------------------------


def stable_row_id(index: int, tikz: str, image: bytes) -> str:
    return sha256_text(
        f"rid-v1\x1f{index}\x1f{tikz}\x1f{sha256_bytes(image)}")[:32]


def _logical_row(row: dict) -> str:
    """Independent copy of the cleaning pipeline's logical-row encoding."""
    return json.dumps(dict(id=row["id"], source_row_index=row["source_row_index"],
                           image_sha256=row["image_sha256"],
                           tikz_sha256=row["tikz_sha256"],
                           instruction=row["instruction"],
                           prompt_version=row["prompt_version"]),
                      sort_keys=True, ensure_ascii=False)


def default_rows(count: int = 6, *, pad_tikz: int = 0):
    rows = []
    for index in range(count):
        tikz = f"\\begin{{tikzpicture}}\\draw (0,0) -- ({index},1);\\end{{tikzpicture}}"
        if pad_tikz:
            tikz += "\\draw (0,0) -- (1,1);" * pad_tikz
        image = make_png_bytes(seed=index)
        rows.append({
            "instruction": f"Draw a simple line number {index}.",
            "tikz_code": tikz,
            "png_image": image,
            "file_id": f"file-{index}.png",
        })
    return rows


def make_export(root, rows=None, *, shard_size: int = 2,
                prompt_version: str = "caption-v1",
                source_dataset: str = "nllg/DaTikZ-V4",
                source_revision: str = "b" * 40,
                caption_model: str = "nvidia/Qwen3.8-27B-NVFP4",
                caption_model_revision: str = "c" * 40,
                run_id: str = "run-fixture",
                identity_sha256: str = "d" * 64,
                manifest_sha256: str = "e" * 64,
                rejected_rows: int = 0,
                complete_rows: int | None = None):
    """Write a cleaning-shaped export directory for tests."""
    import pyarrow as pa
    import pyarrow.parquet as parquet

    root = Path(root)
    shard_root = root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    rows = default_rows() if rows is None else rows

    records = []
    for index, row in enumerate(rows):
        image = row["png_image"]
        tikz = row["tikz_code"]
        records.append({
            "id": row.get("id") or stable_row_id(index, tikz, image),
            "source_row_index": index,
            "file_id": row.get("file_id"),
            "png_image": image,
            "tikz_code": tikz,
            "instruction": row["instruction"],
            "source_dataset": source_dataset,
            "source_revision": source_revision,
            "caption_model": caption_model,
            "caption_model_revision": caption_model_revision,
            "prompt_version": prompt_version,
            "image_sha256": sha256_bytes(image),
            "tikz_sha256": sha256_text(tikz),
        })

    schema = pa.schema([
        pa.field("id", pa.string(), nullable=False),
        pa.field("source_row_index", pa.int32(), nullable=False),
        pa.field("file_id", pa.string()),
        pa.field("png_image", pa.binary(), nullable=False),
        pa.field("tikz_code", pa.string(), nullable=False),
        pa.field("instruction", pa.string(), nullable=False),
        pa.field("source_dataset", pa.string(), nullable=False),
        pa.field("source_revision", pa.string(), nullable=False),
        pa.field("caption_model", pa.string(), nullable=False),
        pa.field("caption_model_revision", pa.string(), nullable=False),
        pa.field("prompt_version", pa.string(), nullable=False),
        pa.field("image_sha256", pa.string(), nullable=False),
        pa.field("tikz_sha256", pa.string(), nullable=False),
    ])
    shard_entries = []
    for shard_index, start in enumerate(range(0, len(records), shard_size)):
        chunk = records[start:start + shard_size]
        name = f"shard-{shard_index:05d}.parquet"
        path = shard_root / name
        parquet.write_table(pa.Table.from_pylist(chunk, schema=schema), path)
        logical = sha256_text("".join(_logical_row(row) + "\n" for row in chunk))
        shard_entries.append({
            "name": name,
            "rows": len(chunk),
            "logical_sha256": logical,
            "file_sha256": sha256_bytes(path.read_bytes()),
        })
    dataset_logical = sha256_text(json.dumps(
        [entry["logical_sha256"] for entry in shard_entries], sort_keys=True))

    export_meta = {
        "schema_version": "export-v1",
        "created_at": "2026-09-23T00:00:00Z",
        "tool_version": "build-dataset-v1",
        "run_id": run_id,
        "identity_sha256": identity_sha256,
        "manifest_sha256": manifest_sha256,
        "rows": len(records),
        "rejected_rows": rejected_rows,
        "shards": len(shard_entries),
        "shard_size": shard_size,
        "dataset_logical_sha256": dataset_logical,
        "complete_rows": len(records) if complete_rows is None else complete_rows,
    }
    run_record = {
        "schema_version": "run-v1",
        "run_id": run_id,
        "identity_sha256": identity_sha256,
        "manifest_sha256": manifest_sha256,
        "identity": {
            "dataset": {"dataset_id": source_dataset, "revision": source_revision,
                        "split": "train", "row_start": 0, "row_limit": 100000},
            "model": {"model_id": caption_model,
                      "revision": caption_model_revision},
            "prompt": {"version": prompt_version, "sha256": "f" * 64},
        },
    }
    provenance = {
        "run": run_record,
        "manifest": {"manifest_sha256": manifest_sha256},
        "export": export_meta,
        "counts": {"states": {"complete": len(records),
                              "rejected": rejected_rows}},
        "shards": shard_entries,
        "rejected": {},
        "attempts": {},
    }
    (root / "export.meta.json").write_text(json.dumps(export_meta))
    (root / "run-metadata.json").write_text(json.dumps(provenance))
    (root / "stats.json").write_text(json.dumps({"states": {"complete": len(records)}}))
    rejected_schema = pa.schema([
        pa.field("id", pa.string(), nullable=False),
        pa.field("source_row_index", pa.int32(), nullable=False),
        pa.field("rejection_reason", pa.string()),
        pa.field("error_category", pa.string()),
        pa.field("error_detail", pa.string()),
        pa.field("attempt_count", pa.int32(), nullable=False),
        pa.field("updated_at", pa.float64()),
        pa.field("image_sha256", pa.string(), nullable=False),
        pa.field("tikz_sha256", pa.string(), nullable=False),
    ])
    parquet.write_table(pa.Table.from_pylist([], schema=rejected_schema),
                        root / "rejected.parquet")
    (root / "checksums.json").write_text(json.dumps({
        "shards": shard_entries,
        "rejected": {"rows": rejected_rows,
                     "file_sha256": sha256_bytes((root / "rejected.parquet").read_bytes())},
        "attempts": {"rows": 0, "file_sha256": "0" * 64},
        "dataset_logical_sha256": dataset_logical,
    }))
    return {
        "root": root,
        "records": records,
        "shards": shard_entries,
        "dataset_logical_sha256": dataset_logical,
        "export_meta": export_meta,
    }


def prepared_export_and_dir(tmpdir, *, rows=None, shard_size=2, max_seq_len=4096,
                            max_quarantined_fraction=0.02,
                            validation_fraction=0.2, test_fraction=0.2,
                            tokenizer_factory=None, **export_kwargs):
    """Run prepare_dataset on a synthetic export; returns (export, prepared, report).

    Small synthetic datasets cannot honor the production 1%/1% fractions
    (they round to zero rows), so fixtures use 20%/20% by default.
    """
    from stage1 import data as data_module
    export = make_export(Path(tmpdir) / "export", rows=rows, shard_size=shard_size,
                         **export_kwargs)
    config = make_config(data={
        "max_seq_len": max_seq_len,
        "max_quarantined_fraction": max_quarantined_fraction,
        "splits": {"validation_fraction": validation_fraction,
                   "test_fraction": test_fraction}})
    factory = tokenizer_factory or fake_tokenizer_factory()
    prepared_dir = Path(tmpdir) / "prepared"
    report = data_module.prepare_dataset(
        [config], export_dir=export["root"], prepared_dir=prepared_dir,
        tokenizer_factory=factory)
    return export, prepared_dir, report


def quarantined_export_and_dir(tmpdir, *, count=6, pad_tikz=40, shard_size=2):
    """Prepare a fixture where every row exceeds max_seq_len and is quarantined.

    Returns ``(export, prepared_dir, report, eligibility)``.
    """
    from stage1 import data as data_module
    export, prepared_dir, report = prepared_export_and_dir(
        tmpdir, rows=default_rows(count, pad_tikz=pad_tikz), shard_size=shard_size,
        max_seq_len=128, max_quarantined_fraction=1.0)
    eligibility = data_module.load_eligibility(
        prepared_dir, next(iter(report["models"])))
    return export, prepared_dir, report, eligibility


def mixed_quarantine_export_and_dir(tmpdir, *, count=8,
                                    quarantined_indices=(0, 1, 2),
                                    pad_tikz=40, max_seq_len=300):
    """Prepare a fixture where only some rows are quarantined.

    Returns ``(export, prepared_dir, report, eligibility)``.
    """
    from stage1 import data as data_module
    rows = default_rows(count)
    for index in quarantined_indices:
        rows[index]["tikz_code"] += "\\draw (0,0) -- (1,1);" * pad_tikz
    export = make_export(Path(tmpdir) / "export", rows=rows)
    config = make_config(data={
        "max_seq_len": max_seq_len, "max_quarantined_fraction": 1.0,
        "splits": {"validation_fraction": 0.25, "test_fraction": 0.25}})
    prepared_dir = Path(tmpdir) / "prepared"
    report = data_module.prepare_dataset(
        [config], export_dir=export["root"], prepared_dir=prepared_dir,
        tokenizer_factory=fake_tokenizer_factory())
    eligibility = data_module.load_eligibility(prepared_dir, "minicpm5")
    return export, prepared_dir, report, eligibility


def write_artifact(directory, *, kind="checkpoint", identity_sha256="a" * 64,
                   gate="smoke-1000", model_id="test/Model-A",
                   model_revision="a" * 40, adapter="minicpm5",
                   global_step=10, supervised_tokens_seen=100,
                   epochs_completed=0.5, complete=True, meta=True,
                   training_method="full", base_model_id=None,
                   base_model_revision=None, lora=None):
    """Write a minimal checkpoint/final artifact directory for tests.

    ``training_method="lora"`` writes native-PEFT-shaped files (adapter config
    and adapter weights) and records base-model provenance, so full and LoRA
    artifacts have genuinely different layouts.
    """
    from stage1 import checkpointing
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
        (directory / name).write_text("{}")
    if training_method == "lora":
        if complete:
            (directory / "adapter_config.json").write_text(json.dumps(
                {"r": 64, "lora_alpha": 64, "lora_dropout": 0.0,
                 "bias": "none",
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"]}))
            (directory / "adapter_model.safetensors").write_bytes(b"adapter")
        base_model_id = base_model_id or model_id
        base_model_revision = base_model_revision or model_revision
        if lora is None:
            lora = {"rank": 64, "alpha": 64, "dropout": 0.0, "bias": "none",
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"]}
    else:
        (directory / "config.json").write_text("{}")
        if complete:
            (directory / "model.safetensors").write_bytes(b"weights")
    if kind == "final":
        (directory / "tokenizer_config.json").write_text("{}")
    if meta:
        checkpointing.write_artifact_meta(
            directory, kind=kind, identity_sha256=identity_sha256,
            model_id=model_id, model_revision=model_revision, adapter=adapter,
            gate=gate, global_step=global_step,
            supervised_tokens_seen=supervised_tokens_seen,
            epochs_completed=epochs_completed, run_id="run1",
            training_method=training_method, base_model_id=base_model_id,
            base_model_revision=base_model_revision, lora=lora)
    return directory


def load_json(path):
    return json.loads(Path(path).read_text())


def synthetic_prepared_dir(tmpdir, *, total_rows=6, quarantined=("row-1",),
                           adapter="minicpm5", model_id="test/Model-A"):
    """Write a prepared directory without pyarrow, for dependency-free tests.

    Mirrors the production layout and hash chain: the split manifest, the
    dataset report, the model fingerprint, the token report and the quarantine
    artifact, with every recorded hash computed the same way production does.
    """
    import json

    from stage1.util import canonical_digest, sha256_file, sha256_text

    root = Path(tmpdir) / "prepared"
    model_dir = root / "models" / adapter
    model_dir.mkdir(parents=True)
    split_pattern = ("train", "train", "validation", "validation", "test", "test")
    rows = []
    split_of = {}
    for index in range(total_rows):
        row_id = f"row-{index}"
        split = split_pattern[index % len(split_pattern)]
        split_of[row_id] = split
        rows.append({"id": row_id, "source_row_index": index,
                     "group_id": f"{index:032x}", "split": split})
    counts = {split: sum(1 for row in rows if row["split"] == split)
              for split in ("train", "validation", "test")}
    payload = {
        "schema_version": "stage1-prepared-v1",
        "tool_version": "stage1-sft-v1",
        "seed": 20260923,
        "splits": {"validation_fraction": 0.2, "test_fraction": 0.2},
        "counts": counts,
        "group_count": total_rows,
        "rows": rows,
    }
    manifest_sha = canonical_digest(payload)
    data_identity_payload = {
        "schema_version": "stage1-prepared-v1",
        "dataset_logical_sha256": "1" * 64,
        "split_manifest_sha256": manifest_sha,
        "seed": 20260923,
        "splits": payload["splits"],
        "row_count": total_rows,
        "complete_rows": total_rows,
        "source_dataset": "nllg/DaTikZ-V4",
        "source_revision": "b" * 40,
    }
    data_identity = dict(data_identity_payload)
    data_identity["sha256"] = canonical_digest(data_identity_payload)
    manifest = dict(payload)
    manifest["created_at"] = "2026-09-24T00:00:00Z"
    manifest["split_manifest_sha256"] = manifest_sha
    manifest["data_identity"] = data_identity
    (root / "split-manifest.json").write_text(json.dumps(manifest))

    quarantine_entries = [{
        "id": row_id, "source_row_index": int(row_id.split("-")[1]),
        "split": split_of[row_id], "reason": "over_length",
        "max_seq_len": 100, "total_tokens": 200, "prompt_tokens": 10,
        "supervised_tokens": 190,
    } for row_id in quarantined]
    quarantine_text = "".join(
        json.dumps(entry, sort_keys=True) + "\n" for entry in quarantine_entries)
    quarantine_sha = sha256_text(quarantine_text)
    if quarantine_text:
        (model_dir / "quarantine.jsonl").write_text(quarantine_text)
    quarantined_by_split = {split: sum(1 for entry in quarantine_entries
                                       if entry["split"] == split)
                            for split in ("train", "validation", "test")}
    token_report = {
        "schema_version": "stage1-prepared-v1",
        "model_id": model_id,
        "model_revision": "a" * 40,
        "adapter": adapter,
        "max_seq_len": 100,
        "chat_template_sha256": "c" * 64,
        "quarantine": {"total": len(quarantine_entries),
                       "over_length": len(quarantine_entries),
                       "sha256": quarantine_sha},
        "splits": {
            split: {
                "examples": counts[split] - quarantined_by_split[split],
                "over_max_seq_len": quarantined_by_split[split],
            } for split in ("train", "validation", "test")
        },
    }
    (model_dir / "token-report.json").write_text(json.dumps(token_report))
    fingerprint = {
        "model_id": model_id, "revision": "a" * 40, "adapter": adapter,
        "tokenizer_class": "FakeTokenizer", "vocab_size": 512,
        "pad_token_id": 0, "chat_template_sha256": "c" * 64,
        "chat_template_source": "tokenizer", "files": {},
    }
    (model_dir / "model.json").write_text(json.dumps(fingerprint))
    dataset_report = {
        "schema_version": "stage1-prepared-v1",
        "status": "pass",
        "created_at": "2026-09-24T00:00:00Z",
        "tool_version": "stage1-sft-v1",
        "export": {
            "root": "/synthetic/export", "dataset_logical_sha256": "1" * 64,
            "checksums_sha256": "2" * 64, "rows": total_rows,
            "complete_rows": total_rows, "rejected_rows": 0, "shards": 1,
            "source_dataset": "nllg/DaTikZ-V4", "source_revision": "b" * 40,
            "prompt_version": "caption-v1", "run_id": "run-synthetic",
        },
        "data_identity": data_identity,
        "models": {adapter: {
            "token_report_sha256": sha256_file(model_dir / "token-report.json"),
            "model_fingerprint_sha256": sha256_file(model_dir / "model.json"),
            "quarantined": len(quarantine_entries),
            "quarantine_fraction": round(len(quarantine_entries) / total_rows, 6),
            "quarantine_sha256": quarantine_sha,
            "max_seq_len": 100,
            "by_split": {
                split: {"eligible": counts[split] - quarantined_by_split[split],
                        "quarantined": quarantined_by_split[split]}
                for split in ("train", "validation", "test")
            },
        }},
    }
    (root / "dataset-report.json").write_text(json.dumps(dataset_report))
    return root


def write_config_file(path, data):
    import yaml
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))
