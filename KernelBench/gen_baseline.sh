#!/bin/bash
#SBATCH --job-name=zeroshot_baseline
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
mkdir -p logs

module load miniconda3 2>/dev/null || true
if [[ -f "${HOME}/.conda/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/.conda/etc/profile.d/conda.sh"
fi
conda activate final

CUDA_VISIBLE_DEVICES=0 python scripts/generate_baseline_time_v100.py 1 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/generate_baseline_time_v100.py 2 0 &
CUDA_VISIBLE_DEVICES=2 python scripts/generate_baseline_time_v100.py 3 0 &
CUDA_VISIBLE_DEVICES=3 python scripts/generate_baseline_time_v100.py 4 0 &

wait