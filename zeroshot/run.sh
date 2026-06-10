#!/bin/bash
#SBATCH --job-name=zeroshot
#SBATCH --partition=gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --account=ACD115083
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/work/b11902044/final/zeroshot}"
mkdir -p logs

module load miniconda3 2>/dev/null || true
if [[ -f "${HOME}/.conda/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/.conda/etc/profile.d/conda.sh"
fi
conda activate final

# CUDA toolkit + GCC>=9 so PyTorch can build the generated kernels.
source /work/b11902044/final/experiments/toolchain.sh

python main.py \
  --input ../data.json \
  --output results.json \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --max-new-tokens 4096 \
  --dtype float16 \
  --batch-size 4 \
  --kernels-dir runs/zero_shot
  # --resume

# Copy kernels to KernelBench for batch eval
KB_RUN=/work/b11902044/final/KernelBench/runs/zero_shot
mkdir -p "${KB_RUN}"
cp runs/zero_shot/level_*.py "${KB_RUN}/" 2>/dev/null || true
echo "Kernels copied to ${KB_RUN}"
