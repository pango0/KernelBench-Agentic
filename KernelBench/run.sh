#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --partition=gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=ACD115083
#SBATCH --output=logs/gen/%x_%j.log
#SBATCH --error=logs/gen/%x_%j.err

set -euo pipefail
# sbatch copies the script to /var/spool/slurm/... — do NOT use dirname "$0"
cd "${SLURM_SUBMIT_DIR:-/work/b11902044/final/KernelBench}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

module load miniconda3
if [[ -f "${HOME}/.conda/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/.conda/etc/profile.d/conda.sh"
elif command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate final

if [[ -f .env ]]; then set -a; source .env; set +a; fi
# vLLM local OpenAI-compatible API (openai SDK requires a non-empty key)
export SGLANG_API_KEY="${SGLANG_API_KEY:-EMPTY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
  --port 10210 \
  --tensor-parallel-size 1 \
  --dtype half \
  --max-model-len 8192 &

until curl -sf http://localhost:10210/health; do sleep 5; done

python scripts/generate_samples.py \
  run_name=zero_shot_L1_hf \
  dataset_src=huggingface \
  level=1 \
  subset='(1,10)' \
  prompt_option=zero_shot \
  server_type=local \
  server_address=localhost \
  server_port=10210 \
  model_name=Qwen/Qwen2.5-Coder-7B-Instruct \
  temperature=0 \
  max_tokens=6000 \
  num_workers=1