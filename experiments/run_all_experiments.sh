#!/bin/bash
# Run EVERY experiment for the paper in one job: the core methods plus the full
# agentic ablation study, then build the report.
#
#   core methods : zeroshot, guided, iterative, agentic        (model: qwen)
#   component abl: agentic_no_rag, agentic_no_analyzer, agentic_no_feedback,
#                  agentic_single_turn, agentic_no_bestof   (best-of-4 backbone)
#   best-of-n    : agentic_bestof2, agentic_bestof8 (+ no_bestof=n1, agentic=n4)
#
# The agentic method is the multi-agent loop documented in docs/AGENTIC_METHOD.md
# (Code Analyzer -> RAG Researcher <-> Documentation -> Kernel Generator ->
#  Evaluator -> Feedback Analyzer -> loop, + post-training data collection).
#
# Usage:
#   sbatch experiments/run_all_experiments.sh                 # everything (skips done)
#   FORCE=1 sbatch experiments/run_all_experiments.sh         # redo completed cells
#   METHODS="agentic agentic_no_rag" sbatch experiments/run_all_experiments.sh
#
#SBATCH --job-name=kb_paper
#SBATCH --partition=gp2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --account=ACD115083
#SBATCH --output=experiments/logs/%x_%j.log
#SBATCH --error=experiments/logs/%x_%j.err

set -uo pipefail   # not -e: one failing cell must not abort the whole sweep

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

MODEL="${MODEL:-qwen}"
CORE_METHODS="zeroshot guided iterative agentic"
ABLATIONS="agentic_no_rag agentic_no_analyzer agentic_no_feedback agentic_single_turn agentic_no_bestof agentic_bestof2 agentic_bestof8"
METHODS="${METHODS:-${CORE_METHODS} ${ABLATIONS}}"
LEVELS="${LEVELS:-1,2,3}"

echo "############################################################"
echo "# Step 0: prepare data (prompts / profiles / baseline)"
echo "############################################################"
python experiments/prepare_data.py

# prepare_data.py drops this flag when the problem count changed -> force re-run.
if [[ -f experiments/runs/.force_rerun ]]; then
  echo ">>> problem count changed: forcing re-run of all cells"
  export FORCE=1
  rm -f experiments/runs/.force_rerun
fi

echo "############################################################"
echo "# Sweep: model=${MODEL} methods=[${METHODS}]   $(date)"
echo "############################################################"

# A self-check of the agentic orchestration before burning GPU hours.
echo ">>> agentic offline self-test"
python agentic/selftest.py || echo ">>> WARNING: agentic self-test failed; continuing anyway"

for METHOD in $METHODS; do
  # run-dir naming mirrors config.run_dir_name (legacy aliases for completed qwen runs)
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

echo
echo "############################################################"
echo "# Building report (core matrix + ablation study)"
echo "############################################################"
python experiments/make_report.py || true

echo "# all experiments finished $(date)"
