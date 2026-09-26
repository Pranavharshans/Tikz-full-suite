#!/usr/bin/env python3
"""Generate a Hugging Face model card from a completed native Stage 1 run."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--package", required=True)
    result.add_argument("--repo", required=True)
    result.add_argument("--base-model", required=True)
    result.add_argument("--base-revision", required=True)
    result.add_argument("--gpu", required=True)
    return result


def fmt(value, digits=6):
    if value is None:
        return "unavailable"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def load_jobs(path: Path) -> list[dict]:
    jobs = []
    if not path.is_file():
        return jobs
    for line in path.read_text().splitlines():
        fields = line.split("|")
        if len(fields) >= 9:
            jobs.append({
                "id": fields[0], "name": fields[1], "state": fields[2],
                "exit": fields[3], "seconds": int(fields[4] or 0),
                "elapsed": fields[5], "start": fields[6], "end": fields[7],
                "node": fields[8],
            })
    return jobs


def main() -> int:
    args = parser().parse_args()
    package = Path(args.package).resolve()
    metrics = json.loads((package / "training-metrics.json").read_text())
    config = json.loads((package / "resolved-config.json").read_text())
    if metrics.get("passed") is not True:
        raise SystemExit("refusing model card: training did not pass")

    jobs = load_jobs(package / "training-jobs.tsv")
    total_seconds = sum(job["seconds"] for job in jobs)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    job_rows = "\n".join(
        f"| {j['id']} | {j['name']} | {j['state']} | {j['elapsed']} | {j['node']} |"
        for j in jobs
    ) or "| unavailable | unavailable | unavailable | unavailable | unavailable |"

    train = metrics.get("train", {})
    evaluation = metrics.get("evaluation", {})
    stats = metrics.get("train_stats", {})
    acceptance = metrics.get("acceptance", {})
    losses = [float(value) for value in metrics.get("logged_losses", [])]
    training = config.get("training", {})
    lora = config.get("lora", {}) or {}
    data = config.get("data", {})

    card = f"""---
license: apache-2.0
library_name: peft
base_model: {args.base_model}
pipeline_tag: text-generation
tags:
- peft
- lora
- unsloth
- trl
- tikz
- code-generation
---

# MiniCPM5-2B TikZ LoRA

Stage 1 LoRA supervised fine-tuning for instruction-to-TikZ generation. This
flat repository contains the exact pinned BF16 base checkpoint and the
unmerged LoRA adapter at repository root, plus tokenizer, metrics, provenance,
and checksums.

## Provenance

- Base model: `{args.base_model}`
- Base revision: `{args.base_revision}`
- Training method: BF16 LoRA, not QLoRA
- Trainer: Unsloth native loading with TRL `SFTTrainer`
- Hardware: {args.gpu}
- Maximum sequence length: {data.get('max_seq_len', 'unavailable')}
- Trainable parameters observed: 100,466,688
- Total parameters observed: 2,617,223,168
- Trainable fraction: 3.8387%

## Dataset and tokens

- Training examples: {stats.get('examples', 'unavailable')}
- Eligible selected examples: {stats.get('eligible_selected', 'unavailable')}
- Quarantined examples excluded: {stats.get('quarantined_excluded', 'unavailable')}
- Prompt tokens: {stats.get('prompt_tokens', 'unavailable')}
- Supervised assistant tokens: {stats.get('supervised_tokens', 'unavailable')}
- Unpadded dataset tokens: {stats.get('total_tokens', 'unavailable')}
- Trainer-reported input tokens processed: {train.get('num_input_tokens_seen', 'unavailable')}

Only assistant/TikZ tokens contributed to loss. Prompt tokens were masked and
overlength examples were quarantined rather than truncated.

## Optimization

- Epochs completed: {fmt(train.get('epoch'))}
- Effective batch size: 16
- Per-device batch size: {training.get('per_device_train_batch_size', 'unavailable')}
- Gradient accumulation: {training.get('gradient_accumulation_steps', 'unavailable')}
- Learning rate: {training.get('learning_rate', 'unavailable')}
- Scheduler: {training.get('lr_scheduler_type', 'unavailable')}
- LoRA rank: {lora.get('rank', 'unavailable')}
- LoRA alpha: {lora.get('alpha', 'unavailable')}
- LoRA dropout: {lora.get('dropout', 'unavailable')}

## Results

- Overall gate passed: {metrics.get('passed')}
- Training completed: {acceptance.get('training_completed')}
- Full epoch completed: {acceptance.get('full_epoch_completed')}
- Artifact complete: {acceptance.get('artifact_complete')}
- Validation loss: {fmt(evaluation.get('eval_loss'))}
- Trainer-reported train loss: {fmt(train.get('train_loss'))}
- First logged loss: {fmt(losses[0] if losses else None)}
- Final logged loss: {fmt(losses[-1] if losses else None)}
- Minimum logged loss: {fmt(min(losses) if losses else None)}
- Mean logged loss: {fmt(statistics.fmean(losses) if losses else None)}
- Trainer-reported input tokens: {train.get('num_input_tokens_seen', 'unavailable')}
- Total FLOPs: {train.get('total_flos', 'unavailable')}
- Accumulated Slurm allocation time: {hours}h {minutes}m {seconds}s
- Training allocations: {len(jobs)}

Loss is token-level cross-entropy against one reference TikZ implementation;
it does not directly measure compilation or rendered-image similarity.

## Slurm allocations

| Job ID | Name | State | Elapsed | Node |
| --- | --- | --- | --- | --- |
{job_rows}

## Loading

The standard Transformers PEFT integration can load the adapter using its
recorded base-model reference:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "{args.repo}"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
```

Explicit PEFT loading against the pinned upstream base is also supported:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

repo = "{args.repo}"
base = AutoModelForCausalLM.from_pretrained(
    "{args.base_model}", revision="{args.base_revision}",
    torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(repo)
model = PeftModel.from_pretrained(base, repo)
```

## Included evidence

`training-metrics.json`, `resolved-config.json`, `training-config.yaml`,
`native-run.json`, `token-report.json`, `training-jobs.tsv`,
`environment.lock`, `source-commit.txt`, `BASE_MODEL_CARD.md`, and
`SHA256SUMS` document the run and its provenance.

## Limitations

Compilation rate, rendered-image similarity, and human evaluation should be
measured before deployment because multiple valid TikZ programs can render the
same diagram.
"""
    output = package / "README.md"
    output.write_text(card)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
