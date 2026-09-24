# Stage 1 dependency locks

Each model config points at exactly one lock file through
`environment.lock_file`. A lock file is the authoritative list of pinned
versions for the GPU environment that trains and evaluates that model.

Rules:

- Install production environments from the lock, not from `pyproject.toml`
  extras: create the virtual environment with the interpreter from
  `stage-1/.python-version` (3.12), then
  `pip install -r locks/<model>.lock`. The lock files are plain pip
  requirements files: every non-comment line is an installable
  `name==version` pin.
- The interpreter pin lives in `stage-1/.python-version` (readable by pyenv and
  by the preflight). It must not be repeated in a requirements file; the lock
  parser rejects `python==` lines with that message.
- `# comments` are allowed. Do not edit a lock file casually: a lock change is
  an environment change and invalidates run identities, so it must be a
  reviewed commit.
- The two model locks are intentionally separate. MiniCPM uses Transformers
  4.57.3 while Qwen uses 5.5.0; never collapse the environments.

Why these versions:

- `unsloth==2026.9.11` requires `transformers>=4.51.3,<=5.5.0`, `trl<=0.24.0`,
  `torch<2.13.0`, `datasets>=3.4.1,<4.4.0`, `peft>=0.18.0` and
  `unsloth_zoo>=2026.9.7`. The lock is the newest coherent point inside those
  caps: `transformers==5.5.0`, `trl==0.24.0`, `datasets==4.3.0`,
  `torch==2.12.1`.
- `Qwen/Qwen3.5-4B` is a hybrid linear/full-attention multimodal checkpoint
  (`Qwen3_5ForConditionalGeneration`); Unsloth ships a Qwen3.5 (4B) notebook,
  so the pinned Unsloth is expected to cover it. The preflight is the
  authority: it must load the exact checkpoint before any gate can run.
- `openbmb/MiniCPM5-2B` is `LlamaForCausalLM`. Upstream documents
  `transformers>=5.6,<6` as the primary path and `transformers==4.57.3` as the
  fallback. The MiniCPM lock uses that fallback because it is inside
  Unsloth's supported range and avoids Transformers-v5 weight conversion. It
  pins `huggingface_hub==0.36.0` to satisfy Transformers 4.57.3's `<1.0`
  requirement.
- Stage 1 does not support sequence packing, so no varlen attention backend is
  required and `flash-attn` is not pinned. The preflight still records which
  attention backends are importable.
