# Ling 3.0 tiny Stage 1 integration

This backend trains `inclusionAI/Ling-3.0-tiny` with plain Transformers, PEFT
and TRL. It deliberately does not import or patch the model with Unsloth:
Ling's custom `BailingMoeV3ForCausalLM` KDA/MLA sparse-MoE architecture has no
validated Unsloth path in this repository.

The shared audited native machinery remains authoritative for export identity,
splits, tokenizer fingerprints, quarantine, assistant-only labels, run
identity, checkpoint resume and acceptance metrics. Only model construction and
LoRA attachment differ.

## Safety properties

- exact Hub revision and exact environment lock;
- checkpoint-native chat template with `enable_thinking: false`;
- BF16 base weights, never QLoRA or implicit quantization;
- every configured LoRA suffix must match a real `Linear` module;
- attention-only adapter targets, avoiding all routed-expert MLPs;
- fresh prepared data and tokenizer report required for adapter `ling3`;
- gates never advance automatically.

## Required order

1. Install `locks/ling3-tiny.lock` in a separate Python 3.12 environment.
2. Download the exact pinned model snapshot.
3. Run `prepare_dataset.py` with only `configs/ling3-tiny-lora.yaml` into a new
   prepared directory. Do not reuse Qwen or MiniCPM prepared artifacts.
4. Run `overfit-100` on one RTX PRO 6000.
5. Run `smoke-1000`, including checkpoint resume.
6. Start `full` only after all earlier gates pass.

Example training command (run inside an allocated GPU job):

```bash
python stage-1/ling_native/train_ling.py \
  --method lora \
  --gate overfit-100 \
  --export /absolute/export \
  --prepared /absolute/prepared-ling3 \
  --run-dir /absolute/runs-native/ling3-tiny-overfit-100-v1 \
  --cache-dir /absolute/huggingface-cache \
  --local-files-only \
  --resume none
```

## Verified without GPU

- the pinned environment resolves without dependency conflicts;
- the pinned tokenizer loads as `PreTrainedTokenizerFast` with vocabulary
  157,153 and pad-token ID 156,892;
- its native template fingerprint is
  `eb6226c94ae38058f875d159f86a206b3a165828c0e7d6bda664ae14667f798a`;
- `enable_thinking: false` emits `detailed thinking off` and an empty thinking
  block, the rendered prompt is an exact prefix of the full conversation, and
  assistant-only boundary masking succeeds;
- the complete Stage 1 unit suite passes with the Ling YAML enabled.

Status: code-complete and CPU/tokenizer-tested; the model load, memory use,
forward/backward, adapter serialization and resume remain GPU-unverified.
