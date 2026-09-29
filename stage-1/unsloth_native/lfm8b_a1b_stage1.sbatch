#!/bin/bash -l
#SBATCH --job-name=lfm25-8b-a1b
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-lfm25-8b-a1b-%j.out

# One Stage 1 gate per job. Arguments:
#   python export prepared run_dir gate resume [cache_dir]
set -euo pipefail
unset SLURM_EXPORT_ENV

if (( $# < 6 || $# > 7 )); then
  echo "usage: sbatch [slurm options] lfm8b_a1b_stage1.sbatch PYTHON EXPORT PREPARED RUN_DIR GATE RESUME [CACHE_DIR]" >&2
  exit 2
fi

PYTHON=$1
EXPORT=$2
PREPARED=$3
RUN_DIR=$4
GATE=$5
RESUME=$6
CACHE_DIR=${7:-}

for value_name in PYTHON EXPORT PREPARED RUN_DIR; do
  value=${!value_name}
  case "$value" in
    /*) ;;
    *) echo "$value_name must be an absolute path" >&2; exit 2 ;;
  esac
done
if [[ -n "$CACHE_DIR" ]]; then
  case "$CACHE_DIR" in /*) ;; *) echo "CACHE_DIR must be an absolute path" >&2; exit 2 ;; esac
  export HF_HOME="$CACHE_DIR"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_DISABLE_XET=1
fi
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable is missing or not executable: $PYTHON" >&2
  exit 2
fi

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
echo "== environment =="
"$PYTHON" --version
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv || true

ARGS=(
  --gate "$GATE"
  --export "$EXPORT"
  --prepared "$PREPARED"
  --run-dir "$RUN_DIR"
  --resume "$RESUME"
)
if [[ -n "$CACHE_DIR" ]]; then
  ARGS+=(--cache-dir "$CACHE_DIR" --local-files-only)
fi

echo "== LFM2.5-8B-A1B Stage 1 gate: $GATE; resume: $RESUME =="
"$PYTHON" stage-1/unsloth_native/train_lfm8b_a1b.py "${ARGS[@]}"
echo "== gate finished; inspect $RUN_DIR/metrics.json =="
