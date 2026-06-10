#!/bin/bash
#SBATCH --job-name=guided_profile
#SBATCH --partition=gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --account=ACD115083
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/work/b11902044/final/guided}"
mkdir -p logs

module load miniconda3 2>/dev/null || true
if [[ -f "${HOME}/.conda/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/.conda/etc/profile.d/conda.sh"
fi
conda activate final

# CUDA toolkit + GCC>=9 so PyTorch can build the generated kernels.
source /work/b11902044/final/experiments/toolchain.sh


CUDA_VISIBLE_DEVICES=0 python profile_reference.py 1 0 &
CUDA_VISIBLE_DEVICES=1 python profile_reference.py 2 0 &
CUDA_VISIBLE_DEVICES=2 python profile_reference.py 3 0 &

wait