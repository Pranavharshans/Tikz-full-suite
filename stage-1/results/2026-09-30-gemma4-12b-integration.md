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

## Remaining validation

No weights loaded, GPU run, training gate, memory-fit test, or Slurm submission.
The GPU environment is a candidate, not a validated lock. Unsloth 2026.9.12
metadata caps Transformers at 5.5.0 (before Unified support). The explicit
installer first obtains its dependencies, then applies Transformers 5.17.0
(within Unsloth Zoo 2026.9.8's Linux ceiling) and retains the pinned Unsloth
wheel without dependency resolution. `pip check` will report that intentional
metadata conflict. Require GPU import/load, forward/backward, adapter-family
checks and bounded overfit before advancing. Token quarantine must also pass
for the Gemma-specific prepared dataset; it was not tested on the full export.

Sources: [checkpoint](https://huggingface.co/unsloth/gemma-4-12b-it),
[Gemma guide](https://unsloth.ai/docs/models/gemma-4),
[Unsloth package metadata](https://pypi.org/pypi/unsloth/2026.9.12/json),
[Unsloth Zoo metadata](https://pypi.org/pypi/unsloth_zoo/2026.9.8/json).
