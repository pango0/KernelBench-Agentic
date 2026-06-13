# The Agentic Method: A Multi-Agent Loop for Kernel Optimization

This document specifies the *agentic* method used in the KernelBench local-model
study. It is written to be lifted directly into the methods section of a paper. All
agents run on a **single local open-weight model** (no API calls); they differ only
in role (system prompt) and the information routed to them.

---

## 1. Motivation

Single-pass code generation gives a model one chance and no grounding: it cannot
look up an optimization technique it half-remembers, cannot see whether its kernel
compiled, and cannot reason about *why* it failed. KernelBench tasks reward exactly
the opposite — correct, then *fast*, GPU kernels — which is an iterative, knowledge-
intensive engineering loop. We therefore cast kernel optimization as a **multi-agent
loop** in which specialized roles analyze, retrieve, generate, evaluate, and diagnose,
passing structured artifacts between each other and iterating on real GPU feedback.

The design composes three lines of work: (i) *iterative self-improvement* with
execution feedback (Self-Refine, Reflexion); (ii) *retrieval-augmented generation*
(RAG) to ground generation in documentation; and (iii) *role specialization* in agent
systems. The loop additionally **harvests its own trajectories as post-training data**.

---

## 2. Architecture

```
   PyTorch / HIP kernel
          │
          ▼
   ┌───────────────┐        ┌────────────────┐        ┌───────────────┐
   │ Code Analyzer │ ─────▶ │ RAG Researcher │ ◀────▶ │ Documentation │
   └───────────────┘        └────────────────┘        └───────────────┘
          │                         │
          │  (analysis)             │ (grounded brief)
          ▼                         ▼
   ┌──────────────────────────────────────┐
   │            Kernel Generator           │ ◀────────────┐
   └──────────────────────────────────────┘              │
          │  (candidate kernel)                           │
          ▼                                               │ (directive)
   ┌───────────────┐        ┌────────────────────┐        │
   │   Evaluator   │ ─────▶ │  Feedback Analyzer  │ ───────┘
   └───────────────┘        └────────────────────┘
          (compile/correct/time)        │
                                         ▼
                          ┌──────────────────────────────┐
                          │ Data collector (post-training)│
                          └──────────────────────────────┘
```

Mapping to the implementation:

| Box                | File / function                              | Kind          |
|--------------------|----------------------------------------------|---------------|
| Code Analyzer      | `agentic/agents.py:analyze_code`             | LLM role      |
| RAG Researcher     | `agentic/agents.py:research`                 | LLM role      |
| Documentation      | `agentic/docs/*` + `agentic/rag.py`          | retrieval     |
| Kernel Generator   | `agentic/agents.py:generate_kernels`         | LLM role      |
| Evaluator          | `agentic/evaluator.py:evaluate_kernel`       | KernelBench   |
| Feedback Analyzer  | `agentic/agents.py:analyze_feedback`         | LLM role      |
| Data collector     | `agentic/datacollector.py`                   | logging       |
| Orchestrator       | `agentic/pipeline.py:run_problem`            | control loop  |

All LLM roles share one `LocalLLM` (`agentic/llm.py`) — the same weights, different
system prompts — so the system stays single-model and reproducible.

---

## 3. The agents

### 3.1 Code Analyzer
Reads the reference PyTorch module and emits a structured analysis: dominant
**operations**, the **bottlenecks** worth replacing, concrete **fusion**
opportunities, an ordered **strategy**, and 3–5 **research queries**. The queries are
the bridge to retrieval — the analyzer decides *what to look up*. Output is parsed by
section header; parsing degrades gracefully if the model omits a header.

### 3.2 RAG Researcher ↔ Documentation
Issues the analyzer's queries against a documentation corpus (`agentic/docs/`:
curated notes on memory coalescing, shared-memory tiling, operator fusion, GEMM/tensor
cores, reductions, online softmax / FlashAttention, `cpp_extension`/`load_inline`,
Triton, and KernelBench pitfalls; PDFs are also indexed when `pypdf` is present).
Retrieval is **BM25 (Okapi)** by default — dependency-free, fully offline — with an
optional dense backend (`sentence-transformers`). The researcher then **synthesizes a
grounded optimization brief** from the retrieved chunks, citing each source tag (e.g.
`[matmul_tiling.md#0]`) to discourage hallucinated APIs. This is the RAG step: the
model conditions on retrieved, task-relevant documentation rather than parametric
memory alone.

### 3.3 Kernel Generator
Produces the full `ModelNew` implementation conditioned on the base prompt, the
analysis, the grounded brief, and (after turn 0) the previous kernel plus the feedback
directive. Supports **best-of-n** sampling: generate *n* candidates, evaluate each, and
keep the best by the ranking (correct + fast) > (correct) > (compiles) > (broken).

### 3.4 Evaluator
The only non-LLM agent. Compiles the candidate, checks numerical correctness against
the reference over several random trials, and times it with CUDA events via the
KernelBench harness. It never raises — failures are returned as a result dict so the
loop can react. It produces both the raw result and a human-readable feedback string.

### 3.5 Feedback Analyzer
Turns raw evaluation output into a **structured diagnosis**: failure *category*
(compilation / correctness / performance / no_code), most-likely *root cause*
(referencing specific APIs/lines), an ordered *fix list*, and a one-line *directive*.
This diagnosis — not the raw error text alone — is what steers the next generation,
making the feedback loop *analytic* rather than merely *reactive*.

### 3.6 Data collector (post-training)
Every trajectory is distilled into post-training records (JSONL under
`experiments/runs/<run_dir>/post_training/`):
- **`sft.jsonl`** — (prompt → best correct kernel): supervised fine-tuning data.
- **`preferences.jsonl`** — (prompt, chosen, rejected): preference/DPO data, built from
  intra-problem candidate ranking (e.g. correct-and-fast ≻ correct ≻ compiles ≻ broken).
- **`reflections.jsonl`** — (broken code, eval feedback, diagnosis, improved code):
  process-supervision / reflection data for training self-correction.
- **`analysis.jsonl`** — the analysis + research brief, for distilling the analyzer/RAG
  behaviour into a base model.

This closes the loop in the figure: the same system that solves tasks **manufactures
the data to improve the model that powers it**.

---

## 4. The control loop

For each task (`agentic/pipeline.py:run_problem`):

1. **Analyze** the reference (if Code Analyzer enabled).
2. **Research**: retrieve + synthesize a grounded brief (if RAG enabled).
3. **Generate** turn 0 (best-of-n), then **Evaluate**.
4. **Iterate** up to `max_turns`: if not stopping, run the Feedback Analyzer on the
   last result to produce a directive, then **regenerate** + **re-evaluate**.
5. **Stop early** when a correct kernel meets the target speedup (or on first
   correct, if configured).
6. **Select** the best turn (correct + fastest) as the final kernel; **collect**
   post-training records.

Stopping, target speedup, turn budget, best-of-n, and each agent are configurable
(`AgenticConfig`), which is exactly how the ablations below are produced from one code
path.

---

## 5. Quality-improving components (and why)

These are deliberate additions over a plain feedback loop; each is independently
ablatable so its value can be measured.

- **Retrieval grounding (RAG).** Conditions generation on documentation, reducing
  hallucinated APIs and surfacing techniques (online softmax, epilogue fusion, bank-
  conflict padding) the model may not reliably recall. *Ablation:* `agentic_no_rag`.
- **Explicit code analysis / planning.** Separates "what to optimize" from "how to
  write it", and produces the retrieval queries. *Ablation:* `agentic_no_analyzer`.
- **Analytic feedback.** A dedicated triage step converts noisy compiler/runtime output
  into a prioritized, actionable directive. *Ablation:* `agentic_no_feedback` (raw
  evaluation text is fed back instead).
- **Best-of-n sampling.** Trades compute for quality by sampling several candidates per
  turn and keeping the best under real evaluation. Part of the full-system backbone
  (n=4). *Ablation:* `agentic_no_bestof` (greedy, n=1); *sweep:* `agentic_bestof2` and
  `agentic_bestof8` map the n=1→2→4→8 compute-for-quality curve.
- **Iterative refinement with execution feedback.** The loop itself. *Ablation:*
  `agentic_single_turn` (one shot, no loop).

---

## 6. Ablation study

Run on the local model over the 30-task subset (levels 1–3 × first 10). Two studies
share one backbone (the full system = all agents on, best-of-4). The **component
ablations** flip exactly one agent off against that backbone; the **best-of-n sweep**
holds all agents on and varies only the sampling width. Variants
(`experiments/config.py:AGENTIC_ABLATIONS`):

| variant                | flips off / changes              | isolates                         |
|------------------------|----------------------------------|----------------------------------|
| `agentic` (full)       | — (all agents, best-of-4)        | the complete system              |
| `agentic_no_rag`       | RAG Researcher + Documentation   | value of retrieval grounding     |
| `agentic_no_analyzer`  | Code Analyzer                    | value of analysis/planning + queries |
| `agentic_no_feedback`  | Feedback Analyzer (raw text only)| value of analytic vs raw feedback|
| `agentic_single_turn`  | loop (`max_turns=1`)             | value of iteration               |
| `agentic_no_bestof`    | best-of-4 → greedy (n=1)         | value of best-of-n sampling      |
| `agentic_bestof2`      | best-of-n = 2 (sweep point)      | compute-for-quality curve        |
| `agentic_bestof8`      | best-of-n = 8 (sweep point)      | compute-for-quality curve        |

External baselines already in the matrix contextualize the ablations: **zero-shot**
(no agents, one shot), **guided** (profiling-enriched prompt, one shot), and
**iterative** (plain feedback loop without specialized agents).

**Metrics** (reported by `experiments/make_report.py`): compilation rate, correctness
rate, **fast_1** (correct *and* faster than eager PyTorch), and geometric-mean speedup
over correct samples. Report per level (difficulty scaling) and pooled.

**Suggested additional analyses for the paper.**
- *Turn-efficiency*: success vs turn index (how often does iteration help, and when do
  returns diminish?). The per-turn data is in `gen_results.json`.
- *Retrieval quality*: do cited sources correlate with correctness/speedup? Sources are
  logged per problem (`research_sources`).
- *Error taxonomy shift*: how the failure distribution moves across ablations
  (`report/data/error_taxonomy.csv`).
- *Compute accounting*: tokens / GPU-seconds per solved task per method (cost of the
  agentic overhead vs payoff).
- *best-of-n scaling*: extend to n ∈ {1,2,4} to chart quality vs compute.

---

## 7. Related work

The method draws on and should be positioned against (verify exact bibliographic
details before submission):

- **KernelBench** — Ouyang et al., 2025. The benchmark and eval protocol; defines
  correctness and the fast_p / speedup metrics used here.
- **Self-Refine** — Madaan et al., NeurIPS 2023. Iterative self-feedback refinement.
- **Reflexion** — Shinn et al., NeurIPS 2023. Verbal reinforcement / reflection from
  environment feedback; motivates the Feedback Analyzer.
- **ReAct** — Yao et al., ICLR 2023. Interleaving reasoning and acting; the analyze →
  research → act structure.
- **Retrieval-Augmented Generation** — Lewis et al., NeurIPS 2020. Grounding generation
  in retrieved documents; the RAG Researcher.
- **Chain-of-Thought / Self-Consistency** — Wei et al., 2022 / Wang et al., 2022.
  Explicit reasoning and sampling multiple solutions (best-of-n).
- **AlphaCode** — Li et al., Science 2022. Large-scale sampling + filtering by
  execution; precedent for best-of-n with an evaluator.
- **STaR** — Zelikman et al., NeurIPS 2022. Bootstrapping training data from a model's
  own (correct) reasoning; motivates the post-training collector.
- **Direct Preference Optimization (DPO)** — Rafailov et al., NeurIPS 2023. Consumes the
  preference pairs produced by the data collector.
- **FlashAttention** — Dao et al., NeurIPS 2022; **Triton** — Tillet et al., MAPL 2019.
  Sources of the concrete kernel-optimization knowledge in the documentation corpus.
- **Voyager** — Wang et al., 2023. Agent with a growing skill/knowledge library; the
  documentation corpus is a static analogue (a learned/grown corpus is future work).

---

## 8. Reproduction

```bash
# Offline orchestration self-test (no GPU/model needed)
python agentic/selftest.py

# Full agentic system on one cell (needs GPUs)
python experiments/run_experiment.py --model qwen --method agentic

# A single ablation
python experiments/run_experiment.py --model qwen --method agentic_no_rag

# Everything (core methods + all ablations) + report, on SLURM
sbatch experiments/run_all_experiments.sh
```

Direct CLI (bypassing the orchestrator):
```bash
python agentic/main.py --input data.json --output agentic/results.json \
  --kernels-dir KernelBench/runs/agentic_qwen --collect-data
```

Key flags: `--no-rag`, `--no-code-analyzer`, `--no-feedback-analyzer`,
`--max-turns N`, `--best-of-n N`, `--rag-backend {bm25,embed}`, `--docs-dir PATH`,
`--collect-data`. See `python agentic/main.py --help`.

---

## 9. Limitations & threats to validity

- **Single model, single seed family.** Generation is stochastic (sampling); report
  variance or fix seeds for the camera-ready. Best-of-n partially controls this.
- **Static documentation corpus.** Retrieval quality is bounded by `agentic/docs/`; a
  larger or task-grown corpus is future work. Out-of-corpus tasks get weaker grounding.
- **Evaluator-in-the-loop cost.** Best-of-n and multi-turn multiply GPU evaluations;
  the compute accounting above is needed for fair comparison.
- **Self-generated post-training data** can amplify the base model's biases; the
  preference/reflection data should be filtered (e.g. require verified correctness,
  which the collector already enforces for SFT) before training.
- **Parsing robustness.** Agents request sectioned output; malformed responses degrade
  to whole-text fallbacks, which can weaken routing between agents.
