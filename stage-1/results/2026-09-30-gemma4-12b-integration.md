# Gemma 4 12B standalone integration — 2026-09-30

Model: `unsloth/gemma-4-12b-it`, revision
`55cdba0740a9765956f49501f689a66b098feda3` (public Hub API).
Architecture: `Gemma4UnifiedForConditionalGeneration`, 48 text layers,
262,144 context. Instruction-tuned checkpoint; text-only BF16 LoRA via
Unsloth `FastVisionModel`, with vision/audio adaptation disabled.

The dedicated launcher reuses the existing native trainer: independent
prepared/run identity, assistant-only masks, overfit-100, smoke-1000, full
dataset gate, native Trainer checkpoints and `--resume auto`.

The upstream template strips channel-looking assistant content. The pinned
Stage 1 template preserves target bytes, uses native turn markers and the
thinking-disabled empty thought prefix. The global attention layers share
K/V; v_proj targets apply to sliding attention projections.

## Verified locally

- Python 3.12, Transformers 5.17.0 recognizes the public Unified config.
- Actual pinned Gemma tokenizer: vocabulary 262,144, padding ID 0.
- Three target probes (ordinary TikZ, literal channel tokens with whitespace,
  literal think tags): prompt labels all -100, every target label equals its
  input token, exact target bytes retained, end-of-turn supervised.
- Native loader regression verifies pinned revision, unquantized loading and
  disabled audio/vision adaptation flags.
- Stage 1 suite: 527 tests run, 92 skipped, no failures (435 passed).
- Launcher --help, both shell syntax checks, git diff --check passed.

## RTX PRO 6000 validation

The pinned environment loaded and patched the full 12B checkpoint on one
NVIDIA RTX PRO 6000 Blackwell Server Edition. Transformers 5.17 removed
`warmup_ratio`; the native runner now preserves the configured ratio through
the replacement `warmup_steps` float interface. The compatibility regression
passes without weakening the runner's refusal of unknown training arguments.

- `overfit-100`: **PASS**, 40 epochs and 280 optimizer steps; evaluation loss
  0.01183, trainer runtime 1,832 seconds, and peak observed GPU memory 47.5 GB.
- `smoke-1000`: **PASS**, one epoch and 63 optimizer steps; evaluation loss
  0.556813, trainer loss 0.594786, 1,119,676 reported input tokens, and peak
  observed GPU memory 64.8 GB.
- Resume drill: **PASS** from smoke `checkpoint-50`; it completed step 63 with
  evaluation loss 0.5568 and saved a complete final adapter.
- Full-data gate: in progress in resumable four-hour allocations. Early
  validation loss improved from 0.4786 at step 600 to 0.4604 at step 800 and
  approximately 0.4321 later in the first allocation. Full completion is not
  claimed here.

Torch Inductor occasionally waits five minutes for compile-worker shutdown
after final artifacts and metrics have already been written. Completed jobs
still exit `0:0`; this cleanup warning has not invalidated a gate.

## Remaining validation

The full-data gate must reach all 5,918 optimizer steps, pass its acceptance
record, and save its final adapter before the Gemma training result can be
called complete. Compilation rate, rendered-image similarity, and qualitative
evaluation remain separate post-training work.

Sources: [checkpoint](https://huggingface.co/unsloth/gemma-4-12b-it),
[Gemma guide](https://unsloth.ai/docs/models/gemma-4),
[Unsloth package metadata](https://pypi.org/pypi/unsloth/2026.9.12/json),
[Unsloth Zoo metadata](https://pypi.org/pypi/unsloth_zoo/2026.9.8/json).
