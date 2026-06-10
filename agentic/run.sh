#!/bin/bash
#SBATCH --job-name=agentic
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
cd "${SLURM_SUBMIT_DIR:-/work/b11902044/final/agentic}"
mkdir -p logs

module load miniconda3 2>/dev/null || true
if [[ -f "${HOME}/.conda/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/.conda/etc/profile.d/conda.sh"
fi
conda activate final

# CUDA toolkit + GCC>=9 so PyTorch can build the generated kernels.
source /work/b11902044/final/experiments/toolchain.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 8 GPUs -> 4 workers; each worker: cuda:0 = LLM, cuda:1 = eval
python main.py \
  --input ../data.json \
  --output results.json \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --max-new-tokens 4096 \
  --max-turns 3 \
  --dtype float16 \
  --num-gpus 8 \
  --gpus-per-worker 2 \
  --num-correct-trials 5 \
  --num-perf-trials 10 \
  --kernels-dir agentic
  # --stop-on-correct
  # --target-speedup 1.5

# Copy kernels for KernelBench batch eval
# python export_kernels.py --results results.json
# cp -r agentic /work/b11902044/final/KernelBench/runs/agentic
