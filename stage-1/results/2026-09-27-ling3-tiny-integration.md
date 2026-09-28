# Ling-3.0-tiny TikZ LoRA integration — 2026-09-27

Stage 1 integration was completed for `inclusionAI/Ling-3.0-tiny`. This is an
implementation and CPU/tokenizer validation record, not a completed training
result. No Ling GPU gate had passed when this record was written.

- Integration commit: `f0aa5d0`
- Base model: `inclusionAI/Ling-3.0-tiny`
- Pinned base revision: `9a98e35fe1c9ee255f78dd64771c7ae15a799481`
- Architecture: custom `BailingMoeV3ForCausalLM` hybrid KDA/MLA sparse MoE
- Backend: Transformers + PEFT + TRL, deliberately isolated from Unsloth
- Method: unquantized BF16 LoRA
- Maximum sequence length: 8,192
- Intended first GPU: 1× NVIDIA RTX PRO 6000 Blackwell 96 GB
- Thinking mode: disabled for direct TikZ targets

## Adapter design

LoRA rank and alpha are both 64 with dropout 0.0. The adapter targets the KDA
and MLA attention projections:

`q_proj`, `k_proj`, `v_proj`, `f_proj`, `b_proj`, `g_proj`, `o_proj`,
`q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, and `dense`.

The routed-expert `gate_proj`, `up_proj`, and `down_proj` modules are
deliberately excluded. Those suffixes occur in every routed expert across
nearly all layers and would create a much larger adapter than intended. Before
PEFT mutates the model, Stage 1 requires every configured target to match a
real linear module. After attachment it verifies that every target produced
trainable LoRA parameters and that all base parameters remain frozen.

## Dataset and tokenizer contract

Ling uses its checkpoint-native Bailing V3 chat template rather than the
Qwen/MiniCPM verbatim ChatML template. Dataset preparation must run into a new
Ling-specific prepared directory; Qwen and MiniCPM token reports cannot be
reused.

Verified tokenizer facts:

- tokenizer class: `PreTrainedTokenizerFast`
- vocabulary size: 157,153
- pad-token ID: 156,892
- native template SHA256:
  `eb6226c94ae38058f875d159f86a206b3a165828c0e7d6bda664ae14667f798a`
- `enable_thinking: false` emits `detailed thinking off`
- the rendered generation prompt is an exact prefix of the full conversation
- assistant-only token-boundary masking passes on a TikZ probe

As with the completed models, prompt tokens are masked from loss and rows over
8,192 tokens are quarantined rather than truncated. Ling-specific eligible and
quarantined counts are not yet known; they are produced by the first dataset
preparation job.

## Validation completed

- pinned dependency set resolves without conflicts;
- Ling YAML passes strict semantic configuration parsing;
- standalone launcher works without eagerly importing GPU libraries;
- complete Stage 1 suite: **518 tests passed**, 92 dependency/GPU tests skipped;
- `git diff --check` and Python byte-compilation passed;
- no Slurm job, model-weight download, forward/backward step, or training run
  was performed as part of the integration commit.

## Required GPU gates

The model is not approved for full training yet. The required sequence is:

1. prepare and audit the Ling tokenizer dataset;
2. run `overfit-100` and prove model load, memory fit, forward/backward,
   learning, evaluation, adapter serialization and final artifact creation;
3. run `smoke-1000` and a native checkpoint-resume drill;
4. start the full one-epoch gate only after all previous evidence passes.

The standalone entry point is `stage-1/ling_native/train_ling.py`. The full
run must use a fresh run directory and must never initialize from an overfit or
smoke adapter.

