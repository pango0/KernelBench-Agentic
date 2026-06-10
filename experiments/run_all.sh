#!/bin/bash
# Run the ENTIRE experiment matrix (all model x method configurations) in ONE job,
# then build the report. Sequential: cells are independent and write to separate
# run dirs, so a failure in one cell does not block the others.
#
# Configurations (defined in experiments/config.py):
#   models : qwen   = Qwen2.5-Coder-7B-Instruct        (local, multi-GPU)
#   methods: zeroshot, guided, iterative, agentic
#
# Usage:
#   sbatch experiments/run_all.sh                    # every cell (skips completed)
#   METHODS="agentic"   sbatch experiments/run_all.sh
#   FORCE=1 sbatch experiments/run_all.sh            # re-run completed cells too
#
#SBATCH --job-name=kb_all
#SBATCH --partition=gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --account=ACD115083
#SBATCH --output=experiments/logs/%x_%j.log
#SBATCH --error=experiments/logs/%x_%j.err

set -uo pipefail   # not -e: one failing cell must not abort the sweep

REPO=/work/b11902044/final
cd "${SLURM_SUBMIT_DIR:-$REPO}"
mkdir -p experiments/logs experiments/runs

module load miniconda3 2>/dev/null || true
if [[ -f "${HOME}/.conda/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/.conda/etc/profile.d/conda.sh"
fi
conda activate final

# CUDA toolkit + GCC>=9 so PyTorch can actually build the generated kernels.
source experiments/toolchain.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS="${MODELS:-qwen}"
METHODS="${METHODS:-zeroshot guided iterative agentic}"
LEVELS="${LEVELS:-1,2,3}"

echo "############################################################"
echo "# Step 0: prepare data (prompts / profiles / baseline)"
echo "############################################################"
python experiments/prepare_data.py

# prepare_data.py drops this flag when the problem count changed; if so, every
# existing cell is stale (old kernels only cover the old count) -> force re-run.
if [[ -f experiments/runs/.force_rerun ]]; then
  echo ">>> problem count changed: forcing re-run of all cells"
  export FORCE=1
  rm -f experiments/runs/.force_rerun
fi

echo "############################################################"
echo "# Matrix sweep: models=[${MODELS}] methods=[${METHODS}]"
echo "# started $(date)"
echo "############################################################"

for MODEL in $MODELS; do
  for METHOD in $METHODS; do
    case "${MODEL}_${METHOD}" in
      qwen_zeroshot)  RUN_DIR=zero_shot ;;
      qwen_guided)    RUN_DIR=guided ;;
      qwen_iterative) RUN_DIR=iterative ;;
      *)              RUN_DIR="${METHOD}_${MODEL}" ;;
    esac
    [[ -d "KernelBench/runs/${METHOD}_${MODEL}" ]] && RUN_DIR="${METHOD}_${MODEL}"

    if [[ "${FORCE:-0}" != "1" && -f "KernelBench/runs/${RUN_DIR}/analysis_level3.json" ]]; then
      echo ">>> SKIP ${MODEL}/${METHOD} (results exist; FORCE=1 to override)"
      continue
    fi

    echo
    echo ">>> ============================================================"
    echo ">>> CELL ${MODEL} / ${METHOD}   $(date)"
    echo ">>> ============================================================"
    python experiments/run_experiment.py \
      --model "${MODEL}" --method "${METHOD}" --levels "${LEVELS}" --stage all \
      || echo ">>> WARNING: cell ${MODEL}/${METHOD} returned non-zero; continuing"
  done
done

echo
echo "############################################################"
echo "# Building report"
echo "############################################################"
python experiments/make_report.py || true

echo "# matrix sweep finished $(date)"
