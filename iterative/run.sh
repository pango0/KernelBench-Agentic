#!/bin/bash
#SBATCH --job-name=iterative
#SBATCH --partition=gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --account=ACD115083
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/work/b11902044/final/iterative}"
mkdir -p logs

module load miniconda3 2>/dev/null || true
if [[ -f "${HOME}/.conda/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/.conda/etc/profile.d/conda.sh"
fi
conda activate final

# CUDA toolkit + GCC>=9 so PyTorch can build the generated kernels.
source /work/b11902044/final/experiments/toolchain.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 8 GPUs -> 4 workers; each worker: cuda:0 = LLM, cuda:1 = KernelBench eval (no OOM overlap).
python main.py \
  --input ../data.json \
  --output results.json \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --max-new-tokens 2048 \
  --max-turns 3 \
  --dtype float16 \
  --num-gpus 8 \
  --gpus-per-worker 2 \
  --num-perf-trials 10 \
  --kernels-dir runs/iterative \
  --prompts-dir prompts
  # --resume (use export_artifacts.py to re-export kernels from existing results.json)

# Copy kernels to KernelBench for batch eval
KB_RUN=/work/b11902044/final/KernelBench/runs/iterative
mkdir -p "${KB_RUN}"
cp runs/iterative/level_*.py "${KB_RUN}/" 2>/dev/null || true
echo "Kernels copied to ${KB_RUN}"
