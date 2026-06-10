#!/bin/bash
# Run ONE (model, method) cell end-to-end on SLURM: generate -> eval -> analyze -> summarize.
#
# Usage:
#   sbatch --export=ALL,MODEL=qwen,METHOD=zeroshot experiments/run_cell.sh
#   MODEL=qwen METHOD=agentic sbatch experiments/run_cell.sh          # also works
#
# MODEL  in {qwen}
# METHOD in {zeroshot, guided, iterative, agentic}
#
#SBATCH --job-name=kbcell
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

set -euo pipefail

REPO=/work/b11902044/final
cd "${SLURM_SUBMIT_DIR:-$REPO}"
mkdir -p experiments/logs experiments/runs

MODEL="${MODEL:?set MODEL=qwen}"
METHOD="${METHOD:?set METHOD=zeroshot|guided|iterative|agentic}"
LEVELS="${LEVELS:-1,2,3}"
STAGE="${STAGE:-all}"

module load miniconda3 2>/dev/null || true
if [[ -f "${HOME}/.conda/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/.conda/etc/profile.d/conda.sh"
fi
conda activate final

# CUDA toolkit + GCC>=9 so PyTorch can actually build the generated kernels.
source experiments/toolchain.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== ${MODEL} x ${METHOD} (stage=${STAGE}, levels=${LEVELS}) ==="
python experiments/run_experiment.py \
  --model "${MODEL}" \
  --method "${METHOD}" \
  --levels "${LEVELS}" \
  --stage "${STAGE}"
