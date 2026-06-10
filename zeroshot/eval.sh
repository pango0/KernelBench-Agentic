#!/bin/bash
#SBATCH --job-name=zeroshot_eval
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

KB=/work/b11902044/final/KernelBench
RUN=zero_shot
RUNS_DIR="${KB}/runs/${RUN}"
EVAL_SCRIPT="${KB}/scripts/eval_from_generations.py"
ANALYSIS_SCRIPT="${KB}/scripts/benchmark_eval_analysis.py"
BASELINE_FILE="${KB}/results/timing/V100_SXM2_32GB/baseline_time_torch.json"

# Shared eval_results.json would collide across levels (keys are "1".."10" only).
# Snapshot per level, then remove before the next level's eval.
# Add 4 when level_4 kernels exist under runs/zero_shot/
for LEVEL in 1 2 3; do
  echo "========== Eval level ${LEVEL} =========="
  rm -f "${RUNS_DIR}/eval_results.json"

  python "${EVAL_SCRIPT}" \
    run_name="${RUN}" \
    dataset_src=local \
    level="${LEVEL}" \
    subset='(1,10)' \
    num_gpu_devices=4 \
    gpu_arch=Volta

  cp "${RUNS_DIR}/eval_results.json" "${RUNS_DIR}/eval_results_level${LEVEL}.json"
  echo "Saved ${RUNS_DIR}/eval_results_level${LEVEL}.json"
done

echo "========== Analysis =========="
cd "${KB}"
for LEVEL in 1 2 3; do
  echo "-------- Level ${LEVEL} --------"
  python "${ANALYSIS_SCRIPT}" \
    run_name="${RUN}" \
    level="${LEVEL}" \
    hardware=V100_SXM2_32GB \
    baseline=baseline_time_torch \
    baseline_file="${BASELINE_FILE}" \
    eval_results_file="${RUNS_DIR}/eval_results_level${LEVEL}.json" \
    output_file="${RUNS_DIR}/analysis_level${LEVEL}.json"
done
