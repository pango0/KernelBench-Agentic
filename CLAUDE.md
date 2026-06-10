# KernelBench Project: LLM-Based Kernel Improvement Study

## 1. Overview

This project investigates the ability of large language models (LLMs) to improve or optimize code kernels using the **KernelBench** benchmark. The focus is on controlled comparisons across models and prompting strategies to evaluate performance in kernel-level code generation and optimization tasks.

The study isolates early difficulty levels and applies multiple reasoning strategies to understand how LLM behavior changes under different guidance regimes.

'''
module load miniconda3
conda activate /home/b11902044/.conda/envs/final
'''
---

## 2. Benchmark: KernelBench

- **Benchmark**: KernelBench
- **Scope restriction**: Only the first 3 difficulty levels are used
- **Data subset rule**: For each of the first 3 levels, only the first 10 problems are selected

### Dataset Configuration

| Level | Number of Tasks Used | Selection Rule |
|------|----------------------|----------------|
| Level 1 | 10 tasks | First 10 |
| Level 2 | 10 tasks | First 10 |
| Level 3 | 10 tasks | First 10 |

Total evaluation set size: **30 tasks**

---

## 3. Models

Only a local, open-weight model is evaluated (no API/proprietary models).

### 3.1 Local Model
- Qwen/Qwen2.5-Coder-7B-Instruct
- Runs locally on GPU via HuggingFace transformers (multi-GPU batched generation)
- Open-weight coding model used across all four methods

---

## 4. Research Methods

Each model is evaluated under four prompting / interaction paradigms:

### 4.1 Zero-Shot
- Direct problem input
- No examples
- No intermediate guidance
- Measures raw model capability

---

### 4.2 Guided
- Structured prompting with:
  - explicit decomposition steps
  - constraints
  - hints about kernel optimization patterns
- No iterative refinement loop

---

### 4.3 Iterative
- Multi-turn refinement process
- Pipeline:
  1. Initial solution generation
  2. Feedback on correctness/performance
  3. Revised solution generation
- Continues until convergence or fixed number of iterations

---

### 4.4 Agentic — multi-agent loop
A multi-agent system on the **same single local model** (roles differ only by system
prompt). Pipeline (see `docs/AGENTIC_METHOD.md` for the full spec + ablations + related
work):

```
PyTorch/HIP kernel → Code Analyzer → RAG Researcher ↔ Documentation
                          │               │
                          ▼               ▼
                   Kernel Generator ←──────┘
                          │
                      Evaluator → Feedback Analyzer → Data collector (post-training)
                          ▲                  │
                          └─────── loop ─────┘
```

- **Code Analyzer** — names ops/bottlenecks/fusion + emits retrieval queries.
- **RAG Researcher ↔ Documentation** — BM25 (offline, default) retrieval over
  `agentic/docs/` (optional dense backend); synthesizes a cited optimization brief.
- **Kernel Generator** — writes/rewrites `ModelNew`; supports best-of-n.
- **Evaluator** — KernelBench compile/correctness/timing (the only non-LLM agent).
- **Feedback Analyzer** — turns raw eval output into a structured diagnosis + directive
  that steers the next turn (the loop back to the generator).
- **Data collector** — logs SFT / preference / reflection JSONL for post-training.

Every box is independently toggleable, which generates the ablation study (§9.2).
Implementation: `agentic/{llm,rag,agents,evaluator,datacollector,pipeline,main}.py`;
offline orchestration self-test: `python agentic/selftest.py`.

---

## 5. Experimental Design

### 5.1 Full Factorial Setup

Each task is evaluated across:

- 1 model (local Qwen)
- 4 methods
- 30 tasks total

Total runs:

> 1 × 4 × 30 = **120 evaluations**

---

## 6. Evaluation Criteria

Performance is assessed using kernel-specific metrics:

### 6.1 Correctness
- Functional equivalence to reference kernel
- Pass/fail tests

### 6.2 Performance Improvement
- Runtime reduction
- Memory efficiency (if available)
- Computational complexity improvements

### 6.3 Robustness
- Stability across input variations
- Edge-case handling

### 6.4 Code Quality
- Readability
- Structural correctness
- Maintainability of generated kernel

---

## 7. Output Artifacts

The project will produce:

- Benchmark comparison tables
- Model × method performance heatmaps
- Per-level performance breakdowns
- Agentic ablation-study table (`report/data/ablation_metrics.csv`)
- Qualitative error taxonomy:
  - logical errors
  - memory errors
  - optimization failures
  - hallucinated APIs or constructs
- Post-training datasets harvested by the agentic loop (SFT / preference / reflection
  JSONL under `experiments/runs/agentic_qwen/post_training/`)

---

## 8. Expected Insights

The study aims to identify:

- How much structured prompting (guided / iterative / agentic) lifts a local open-weight model over plain zero-shot
- The effectiveness of iterative vs agentic reasoning in kernel optimization
- Sensitivity of the model to problem difficulty scaling within early KernelBench levels
- Trade-offs between reasoning overhead and performance gains

---

## 9. Implementation & Workflow

### 9.1 Directory layout

```
final/
├── data.json              # zero-shot prompts for all 30 tasks (levels 1-3 x 10)
├── data_guided.json       # same tasks, prompts enriched with reference GPU profiling
├── zeroshot/  guided/  iterative/  agentic/   # one method each, all share a CLI:
│     main.py              #   local Qwen (multi-GPU HuggingFace generation)
│     run.sh / eval.sh     #   legacy per-method SLURM scripts (still usable)
│   agentic/               # >>> multi-agent loop (see docs/AGENTIC_METHOD.md) <<<
│     llm.py rag.py agents.py evaluator.py datacollector.py pipeline.py
│     docs/*.md            #   RAG documentation corpus (BM25-indexed)
│     selftest.py          #   offline orchestration test (no GPU)
├── docs/AGENTIC_METHOD.md # formal method write-up (architecture, ablations, refs)
├── KernelBench/           # upstream benchmark + eval harness
│     runs/<run_dir>/      #   kernels + eval_results_level*.json + analysis_level*.json
│     scripts/eval_from_generations.py, benchmark_eval_analysis.py
│     results/timing/V100_SXM2_32GB/baseline_time_torch.json   # eager-PyTorch baseline
├── experiments/           # >>> unified orchestration layer (use this) <<<
│     config.py            #   registry: 1 model x 4 methods, paths, error taxonomy
│     run_experiment.py    #   run ONE cell end to end, save report-ready output
│     make_report.py       #   aggregate ALL cells -> CSVs + REPORT.md + heatmaps
│     run_cell.sh          #   SLURM wrapper for one cell (MODEL, METHOD env vars)
│     submit_all.sh        #   submit the whole matrix (skips already-done cells)
│     runs/<run_dir>/      #   generate.log, eval.log, gen_results.json, summary.{json,md}
└── report/
      REPORT.md            #   generated report scaffold (tables + figures wired in)
      data/*.csv           #   master_metrics, summary_by_cell, error_taxonomy
      figures/*.png        #   model x method heatmaps
```

### 9.2 Experiment matrix (single source of truth: `experiments/config.py`)

| tag | model | kind |
|---|---|---|
| `qwen` | Qwen2.5-Coder-7B-Instruct | local (HF, multi-GPU) |

Generation runs the local HuggingFace model on GPU for every method. To change the
model, edit `MODELS` in `experiments/config.py` — nothing else.

Core methods: `zeroshot`, `guided`, `iterative`, `agentic`. Run-dir name =
`<method>_<model>` (e.g. `agentic_qwen`), with legacy aliases `zero_shot` / `guided` /
`iterative` for the first completed Qwen runs so they appear in the report without
re-running.

**Agentic ablations** (`AGENTIC_ABLATIONS` in `config.py`; share `agentic/main.py`, not
part of the core matrix): `agentic_no_rag`, `agentic_no_analyzer`,
`agentic_no_feedback`, `agentic_single_turn`, `agentic_bestof2`. Each flips one
component of the full agentic system; `make_report.py` renders them in a dedicated
"Agentic ablation study" table. See `docs/AGENTIC_METHOD.md` §6.

### 9.3 Running experiments

The headline entry point. It runs **generate → eval → analyze → summarize** for one
(model, method) cell and writes everything a report needs:

```bash
# one cell, interactively (needs GPUs)
python experiments/run_experiment.py --model qwen --method agentic
python experiments/run_experiment.py --model qwen --method iterative

# re-summarize / re-analyze without re-running generation
python experiments/run_experiment.py --model qwen --method zeroshot --stage summarize

# on SLURM:
sbatch --export=ALL,MODEL=qwen,METHOD=agentic experiments/run_cell.sh   # one cell, one job
sbatch experiments/run_all.sh             # ALL configurations in a single job, then report
experiments/submit_all.sh                 # ALL configurations, one job PER cell (parallel)
experiments/submit_all.sh qwen            # all methods for Qwen
```

`run_all.sh` runs every cell sequentially inside one allocation and finishes by
building the report; `submit_all.sh` fans the cells out into parallel jobs. Both
skip cells whose results already exist (set `FORCE=1` to redo), and both honour
`MODELS=...` / `METHODS=...` to restrict the sweep.

Eval hardware is the V100 (Volta); the baseline is eager PyTorch.

### 9.4 Building the report

```bash
python experiments/make_report.py        # no GPU; reads existing eval results
```

Regenerate freely; it always reflects the latest completed cells and lists which are
still pending. Metrics: **compilation rate**, **correctness rate**, **fast_1**
(correct AND faster than eager PyTorch), and **geo-mean speedup** over correct samples.

### 9.5 Status (as of 2026-06-04)

Completed: **Qwen × {zeroshot, guided, iterative}** (3/4 cells).
Pending: Qwen × agentic. Launch the whole matrix with `sbatch experiments/run_all.sh`
(or `experiments/submit_all.sh` for parallel jobs), then rebuild the report.

### 9.6 One command to run everything

```bash
sbatch experiments/run_all.sh              # core matrix only (zeroshot/guided/iterative/agentic)
sbatch experiments/run_all_experiments.sh  # PAPER sweep: core matrix + ALL agentic ablations + report
```

`run_all_experiments.sh` is the headline script for the paper: `prepare_data.py` →
agentic self-test → every core method → every agentic ablation (`agentic_no_rag`,
`agentic_no_analyzer`, `agentic_no_feedback`, `agentic_single_turn`,
`agentic_bestof2`) → `make_report.py`. Both jobs are idempotent — completed cells are
skipped (set `FORCE=1` to redo) and honour `METHODS=...` to restrict the sweep. The
full agentic run also writes post-training data (SFT / preference / reflection JSONL)
to `experiments/runs/agentic_qwen/post_training/`.

**Changing the study size (e.g. 10 → 20 problems per level).** Edit one line —
`PROBLEMS_PER_LEVEL` in `experiments/config.py` — then run the command above.
`prepare_data.py` detects the new size and automatically re-exports `data.json`,
re-profiles references, rebuilds `data_guided.json`, and regenerates the V100 speed
baseline (required, or problems 11–20 get no speedup numbers); it then flags the run
so every cell is recomputed for the new set. The KernelBench source has 100/100/50
problems for levels 1/2/3, so sizes up to 50 are fine.

---