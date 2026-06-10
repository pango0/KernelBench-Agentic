#!/bin/bash
# Submit a batch of (model, method) cells to SLURM, one job each.
#
# Usage:
#   experiments/submit_all.sh                       # submit every pending cell
#   experiments/submit_all.sh qwen                  # all methods for one model
#   experiments/submit_all.sh qwen agentic          # a single cell
#   METHODS="iterative agentic" experiments/submit_all.sh
#   FORCE=1 experiments/submit_all.sh ...           # resubmit even if results exist
#
# Cells whose analysis JSON already exists are skipped unless FORCE=1.
# After the jobs finish, build the report with:
#   python experiments/make_report.py

set -euo pipefail
REPO=/work/b11902044/final
cd "$REPO"

ALL_MODELS=(qwen)
ALL_METHODS=(zeroshot guided iterative agentic)

# Positional args: [model] [method]; env MODELS/METHODS override.
MODELS="${MODELS:-${1:-${ALL_MODELS[*]}}}"
METHODS="${METHODS:-${2:-${ALL_METHODS[*]}}}"

for MODEL in $MODELS; do
  for METHOD in $METHODS; do
    # naming mirrors config.run_dir_name (legacy aliases for completed qwen runs)
    case "${MODEL}_${METHOD}" in
      qwen_zeroshot)  RUN_DIR=zero_shot ;;
      qwen_guided)    RUN_DIR=guided ;;
      qwen_iterative) RUN_DIR=iterative ;;
      *)              RUN_DIR="${METHOD}_${MODEL}" ;;
    esac
    # prefer canonical dir if it already exists
    [[ -d "KernelBench/runs/${METHOD}_${MODEL}" ]] && RUN_DIR="${METHOD}_${MODEL}"

    if [[ "${FORCE:-0}" != "1" && -f "KernelBench/runs/${RUN_DIR}/analysis_level3.json" ]]; then
      echo "skip ${MODEL}/${METHOD} (results exist in runs/${RUN_DIR}; FORCE=1 to override)"
      continue
    fi

    JOB="kb_${METHOD}_${MODEL}"
    echo "submit ${MODEL}/${METHOD}  -> job ${JOB}"
    sbatch --job-name="${JOB}" \
           --export=ALL,MODEL="${MODEL}",METHOD="${METHOD}" \
           experiments/run_cell.sh
  done
done

echo
echo "Track with:  squeue -u $USER"
echo "Then build:  python experiments/make_report.py"
