"""Adapter from the audited Stage 1 export/preparation to native TRL data."""
from __future__ import annotations

from dataclasses import dataclass

from stage1 import data, train


@dataclass(frozen=True)
class NativeDatasets:
    train: object
    validation: object
    train_stats: dict
    validation_stats: dict
    prepared: dict
    export: object


def load_native_datasets(config, paths, gate, tokenizer, template, *, progress=None):
    """Load exact prepared splits with quarantine and assistant-only labels."""
    prepared = data.verify_prepared(
        paths.prepared_dir, adapter_slug=config.model.adapter)
    export = data.verify_export(
        paths.export_dir,
        require_complete=config.data.require_complete_export,
        quick=False,
    )
    eligibility = data.load_eligibility(paths.prepared_dir, config.model.adapter)
    train_dataset, train_stats = train.build_tokenized_dataset(
        config, tokenizer, template, export, prepared["manifest"], "train",
        eligibility, limit=gate.max_rows, seed=config.seed, progress=progress)
    if gate.name == "overfit-100":
        validation_dataset = train_dataset
        validation_stats = dict(train_stats)
    else:
        validation_dataset, validation_stats = train.build_tokenized_dataset(
            config, tokenizer, template, export, prepared["manifest"],
            "validation", eligibility, limit=config.training.eval_max_rows,
            seed=config.seed, progress=progress)
    return NativeDatasets(
        train=train_dataset,
        validation=validation_dataset,
        train_stats=train_stats,
        validation_stats=validation_stats,
        prepared=prepared,
        export=export,
    )

