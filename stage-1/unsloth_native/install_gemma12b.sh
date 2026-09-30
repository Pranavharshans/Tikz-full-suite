#!/usr/bin/env bash
# Run inside a dedicated Python 3.12 Linux/NVIDIA virtual environment.
set -euo pipefail
STAGE1_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# Install Unsloth and its transitive dependencies using its published metadata.
python -m pip install unsloth==2026.9.12
# Then override the obsolete Transformers ceiling with the candidate base lock.
python -m pip install -r "$STAGE1_ROOT/locks/gemma4-12b.lock"
# Explicit override: published Unsloth metadata caps Transformers at 5.5.0,
# before Gemma4Unified support. Do not let the resolver downgrade the base.
python -m pip install --no-deps unsloth==2026.9.12
python -c 'import unsloth; from transformers import Gemma4UnifiedForConditionalGeneration; print("Gemma Unified imports OK; GPU loading/training still requires the staged gates")'
