"""LoRA adapter merge/export.

This module is only ever called from ``scripts/merge_adapter.py``: training
never merges automatically. The merge path loads the pinned base model
**base-only** (no fresh adapter is created), attaches the saved adapter exactly
once, verifies that merged logits match base-plus-adapter logits on a fixed
input, and writes the merged model with a metadata record.
"""
from __future__ import annotations

from pathlib import Path

from .errors import DataError
from .util import write_json_atomic

MERGE_SCHEMA_VERSION = "stage1-merge-v1"


def prepare_base_and_adapter(config, adapter_dir, *, base_loader=None,
                             adapter_attacher=None, local_files_only: bool = False,
                             cache_dir=None):
    """Load the pinned base checkpoint and attach the saved adapter once.

    ``base_loader`` and ``adapter_attacher`` are injection seams for tests;
    production uses ``adapters.load_base_model_and_tokenizer`` and
    ``adapters.attach_lora_adapter``.
    """
    from . import adapters

    loader = base_loader or adapters.load_base_model_and_tokenizer
    attacher = adapter_attacher or adapters.attach_lora_adapter
    model, tokenizer, report = loader(
        config, local_files_only=local_files_only, cache_dir=cache_dir)
    report["adapter_attached"] = "once"
    model = attacher(model, adapter_dir)
    return model, tokenizer, report


def merge_with_verification(model, batch, *, tolerance: float):
    """Merge the adapter and verify merged logits against base+adapter.

    LoRA dropout must be disabled for the comparison; both forwards run in
    eval mode, which is what the loading paths already do.
    """
    import torch

    model.eval()
    with torch.no_grad():
        before = model(**batch).logits.detach().float()
    merged = model.merge_and_unload()
    merged.eval()
    with torch.no_grad():
        after = merged(**batch).logits.detach().float()
    max_abs_diff = float((before - after).abs().max())
    verification = {
        "checked": True,
        "max_abs_diff": max_abs_diff,
        "tolerance": tolerance,
        "passed": max_abs_diff <= tolerance,
    }
    if max_abs_diff > tolerance:
        raise DataError(
            f"Merged logits differ from base+adapter by {max_abs_diff} "
            f"(tolerance {tolerance}); refusing to export an unverified merged "
            "model")
    return merged, verification


def export_merged(merged, tokenizer, out_dir, *, metadata: dict) -> dict:
    """Save the merged model, tokenizer and merge metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    metadata_path = out_dir / "merge-metadata.json"
    write_json_atomic(metadata_path, metadata)
    return {"out_dir": str(out_dir), "metadata_path": str(metadata_path)}


def merge_adapter(config, adapter_dir, out_dir, *, batch, base_loader=None,
                  adapter_attacher=None, tolerance: float = 1e-3,
                  verify: bool = True, metadata: dict | None = None,
                  local_files_only: bool = False, cache_dir=None) -> dict:
    """Full merge flow: base-only load, one attach, verify, export."""
    model, tokenizer, report = prepare_base_and_adapter(
        config, adapter_dir, base_loader=base_loader,
        adapter_attacher=adapter_attacher, local_files_only=local_files_only,
        cache_dir=cache_dir)
    if verify:
        merged, verification = merge_with_verification(
            model, batch, tolerance=tolerance)
    else:
        merged = model.merge_and_unload()
        verification = {"checked": False,
                        "reason": "verification explicitly skipped"}
    merged_metadata = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "adapter": str(adapter_dir),
        "base_model_id": config.model.id,
        "base_model_revision": config.model.revision,
        "merged_from_adapter": True,
        "config_source_sha256": config.source_sha256,
        "load_report": report,
        "verification": verification,
    }
    if metadata:
        merged_metadata.update(metadata)
    written = export_merged(merged, tokenizer, out_dir, metadata=merged_metadata)
    return {
        "out_dir": written["out_dir"],
        "metadata_path": written["metadata_path"],
        "verification": verification,
        "load_report": report,
    }
