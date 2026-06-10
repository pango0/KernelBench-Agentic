#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
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

python scripts/eval_from_generations.py \
  run_name=zero_shot \
  dataset_src=local \
  level=1 \
  subset='(1,10)' \
  num_gpu_devices=4 \
  gpu_arch=Volta

python scripts/eval_from_generations.py \
  run_name=zero_shot \
  dataset_src=local \
  level=2 \
  subset='(1,10)' \
  num_gpu_devices=4 \
  gpu_arch=Volta

python scripts/eval_from_generations.py \
  run_name=zero_shot \
  dataset_src=local \
  level=3 \
  subset='(1,10)' \
  num_gpu_devices=4 \
  gpu_arch=Volta

# python scripts/eval_from_generations.py \
#   run_name=zero_shot \
#   dataset_src=local \
#   level=4 \
#   subset='(1,10)' \
#   num_gpu_devices=4 \
#   gpu_arch=Volta
