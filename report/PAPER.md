# Can a Local 7B Coder Model Optimize GPU Kernels? A Controlled Study of Prompting and Agentic Strategies on KernelBench

*Draft — Qwen2.5-Coder-7B-Instruct on KernelBench levels 1–3.*
*Numbers are auto-synced from `report/data/*.csv`; see `report/REPORT.md` for the machine-generated tables and `docs/AGENTIC_METHOD.md` for the agentic architecture.*

## Abstract

We study how far structured prompting and agentic scaffolding can push a *single local,
open-weight* code model — Qwen2.5-Coder-7B-Instruct — on the task of replacing PyTorch
operators with custom CUDA kernels, using the KernelBench benchmark (levels 1–3, first 10
problems each; 30 tasks). Holding the model fixed, we compare four interaction regimes of
increasing structure: **zero-shot**, **profiling-guided**, **iterative** self-refinement,
and a **multi-agent** loop (Code Analyzer → RAG Researcher → Generator → Evaluator →
Feedback Analyzer). We find three results that we believe generalize beyond this model.
**(1) Correctness, not speed, is the binding constraint:** even kernels that pass
correctness are on average *slower* than eager PyTorch (geometric-mean speedup ≈ 0.97–0.99×
across all methods), because a 7B model cannot out-write cuBLAS/cuDNN on dense GEMM and
convolution. **(2) The best strategy depends on difficulty:** profiling-guided prompting is
strongest overall (43% correct) and uniquely effective on the largest models (Level 3),
whereas the multi-agent loop is the *only* method that solves any Level-2 fusion problem
(3/10 vs. 0/10 for every other method) — at the cost of degraded performance on trivial
Level-1 tasks. **(3) Naïve scaffolding can hurt:** a first implementation of the agentic
loop scored *below* zero-shot (3% correct, 33% compile) because its analysis/retrieval
steps pushed the small model toward ambitious hand-tiled kernels it could not compile;
correcting the loop to prefer "simplest correct first" and spending the extra compute on
best-of-n sampling instead recovered it to parity (30% correct, 73% compile). We release an
ablation study isolating each agent's contribution and a best-of-n test-time-compute curve.

## 1. Introduction

GPU kernel optimization is an attractive testbed for code-generating LLMs: it is
economically important, has an unambiguous correctness oracle (numerical equivalence to a
reference), and an unambiguous quality signal (wall-clock speedup). KernelBench formalizes
this as "rewrite this PyTorch `Model` as a `ModelNew` with custom CUDA, keep it correct, make
it fast." Most prior reporting on KernelBench centers on large frontier models. We instead
ask a question relevant to practitioners with fixed, local compute: **given one mid-sized
open-weight coder model, how much does the *interaction strategy* around it matter, and
where does it help?**

We make the model the single controlled variable and vary only the scaffolding:

1. **Zero-shot** — the raw KernelBench prompt (a capability floor).
2. **Guided** — the prompt enriched with reference GPU profiling and a roofline-aware rule
   ("don't reimplement cuBLAS/cuDNN; fuse the memory-bound epilogue instead").
3. **Iterative** — multi-turn self-refinement on real compile/correctness/timing feedback.
4. **Agentic** — a multi-agent loop on the *same* model (roles differ only by system
   prompt): code analysis, retrieval-augmented research over a kernel-optimization corpus,
   generation, evaluation, and analytic feedback, with optional best-of-n sampling.

**Contributions.** (i) A controlled 1-model × 4-method comparison on 30 KernelBench tasks
with compile/correctness/fast_1/speedup metrics and a per-level difficulty breakdown.
(ii) The finding that *strategy interacts with difficulty* — guided wins overall but agentic
uniquely cracks Level-2 fusion. (iii) A diagnosis of *why naïve agentic scaffolding
underperforms a single shot* for a small model, and a fix (competence-matched ambition +
best-of-n) that recovers it. (iv) A coherent ablation isolating each agent and a best-of-n
test-time-compute scaling curve. (v) As a byproduct, a small post-training dataset
(SFT/preference/reflection trajectories) harvested by the loop.

## 2. Related work

KernelBench [Ouyang et al.] defines the benchmark, the `fast_p` metric (fraction correct
*and* ≥ p× the PyTorch baseline), and difficulty levels (1: single ops; 2: fused operator
sequences; 3: full architectures). Our agentic design draws on retrieval-augmented
generation, execution-feedback refinement (Reflexion/Self-Debug-style loops), and best-of-n
/ test-time-compute scaling. The distinguishing constraint here is the use of a *single,
small, local* model across every role and method, with all evaluation performed by the
real KernelBench harness on a V100. See `docs/AGENTIC_METHOD.md` for the full reference list.

## 3. Methods

All methods share the same base prompt, model, and KernelBench evaluation harness; they
differ only in the information and control flow around generation.

**Zero-shot.** The model receives the reference architecture and is asked for an optimized
`ModelNew`. No examples, no feedback. This is the capability floor.

**Guided.** We profile each reference module with the PyTorch profiler on the target GPU and
inject the top operators by CUDA time into the prompt, plus an explicit roofline rule: keep
torch's tuned matmul/conv for the heavy compute and win only by fusing the cheap
memory-bound epilogue (bias/activation/scale/clamp) or replacing elementwise/reduction/norm
chains. Single shot, no loop.

**Iterative.** The model generates, the harness compiles/checks/times the result, and a
distilled feedback string drives up to two refinements. We select the best turn by a
`(correct, compiled, speedup)` ranking so a later broken turn never displaces an earlier
working one.

**Agentic.** A multi-agent loop on the same model (`docs/AGENTIC_METHOD.md`):

```
Reference → Code Analyzer → RAG Researcher ↔ Doc corpus
                  │               │
                  ▼               ▼
            Kernel Generator ◀────┘  →  Evaluator  →  Feedback Analyzer
                  ▲                                          │
                  └──────────────── loop ────────────────────┘
```

The Code Analyzer names bottlenecks/fusion opportunities and emits retrieval queries; the
RAG Researcher retrieves from a BM25-indexed corpus of kernel-optimization notes and writes
a cited brief; the Generator writes `ModelNew` (optionally best-of-n, with the per-turn
Evaluator keeping the best candidate); the Feedback Analyzer turns raw evaluator output into
a prioritized directive for the next turn. Every component is independently toggleable,
which yields the ablation study (§5.5–5.6). Crucially, the Generator and Analyzer are
**competence-matched**: they are instructed to reach a correct, compiling kernel first
(reuse torch ops, fuse only the epilogue) and to escalate to hand-tiled GEMM/conv only after
a correct baseline exists — a direct response to the failure mode in §5.7.

## 4. Experimental setup

- **Benchmark.** KernelBench levels 1–3, first 10 problems each = **30 tasks**.
- **Model.** Qwen2.5-Coder-7B-Instruct, run locally on GPU via HuggingFace (multi-GPU
  batched generation, fp16).
- **Hardware / baseline.** Evaluation on a Tesla V100-SXM2-32GB; the speed baseline is eager
  PyTorch (`baseline_time_torch`). fp32 precision, 5 correctness trials, 10 timing trials.
- **Metrics.** *Compilation rate* (builds), *correctness rate* (matches reference within
  tolerance), **fast_1** (correct *and* faster than eager PyTorch), and *geometric-mean
  speedup* over correct samples (>1 is faster).
- **Design.** 1 model × 4 methods, plus the agentic ablation/sweep variants (§5.5–5.6). All
  generation is local; all timing is on the V100.

## 5. Results

### 5.1 Headline comparison (30 tasks, pooled)

| Method | Compile % | Correct % | fast_1 % | Geo-mean speedup |
|---|---|---|---|---|
| Zero-Shot | 70 | 30 | 13 | 0.98× |
| **Guided** | **77** | **43** | **27** | **0.99×** |
| Iterative | 67 | 30 | 10 | 0.98× |
| Agentic | 73 | 30 | 10 | 0.97× |

Profiling-guided prompting is the clear overall winner: it lifts correctness from 30% to
43% and *doubles* the faster-than-baseline rate (13% → 27%) over zero-shot, at the same one
shot of compute. Iterative refinement does not help this model — it matches zero-shot on
correctness and is marginally worse on compile rate, indicating that for a 7B model, blind
"try again with the error log" rarely converts a failure into a success. The agentic loop
reaches parity with zero-shot on aggregate correctness but, as §5.2 shows, with a very
different *distribution* of successes.

### 5.2 Difficulty scaling: the best strategy depends on the level

Correct solutions out of 10, per level:

| Method | L1 (single ops) | L2 (fused sequences) | L3 (architectures) |
|---|---|---|---|
| Zero-Shot | 9 | 0 | 0 |
| Guided | **10** | 0 | **3** |
| Iterative | 9 | 0 | 0 |
| Agentic | 5 | **3** | 1 |

This is our central qualitative finding. On **Level 1** (a single matmul/conv/elementwise
op), simpler is better: zero-shot and guided solve 9–10/10, while the agentic loop's
analysis-and-retrieval overhead actively *hurts* (5/10), occasionally talking the model into
an unnecessary custom kernel where a one-line torch op would pass. On **Level 2** (fused
sequences such as `ConvTranspose2d→BiasAdd→Clamp→Scale` or
`Conv3d→Divide→Max→GlobalAvgPool→BiasAdd`), the agentic loop is the **only** method to score
at all (3/10 vs. 0/10 everywhere else): explicit fusion analysis plus iterative feedback is
exactly what these multi-op problems reward. On **Level 3** (full networks), profiling-guided
prompting uniquely succeeds (3/10), because the reference profile points the model straight
at the dominant layer. No single method dominates across difficulty — a result with direct
practical implications for how to route problems to strategies.

### 5.3 The speedup ceiling

Across *every* method, the geometric-mean speedup over correct samples sits at 0.97–0.99×:
correct kernels are, on average, slightly *slower* than eager PyTorch. The reason is
structural — PyTorch already dispatches matmul/linear and convolution to cuBLAS/cuDNN, and a
7B model writing fp32 CUDA cannot beat those libraries on dense compute. The few genuine wins
are exactly the cases where a custom kernel *can* help: e.g., the agentic loop reaches 1.23×
on `7_Matmul_with_small_K_dimension` (a skinny GEMM where cuBLAS is underutilized). The
practical takeaway is that, for a small model, the realistic objective on most of these
problems is *correct fused kernels at ~1×*, and the metric that separates the methods is
correctness/compilation, not raw speedup. Guided's higher fast_1 (27%) comes from the
roofline rule steering it toward the fusible, memory-bound epilogues where wins are possible.

### 5.4 Error taxonomy

Failure counts (correct cells excluded) shift in an informative way:

| Method | compilation | hallucinated_api | shape_mismatch | wrong_output | slow |
|---|---|---|---|---|---|
| Zero-Shot | 7 | 4 | 5 | 4 | 5 |
| Guided | 5 | 6 | 3 | 3 | 5 |
| Iterative | 9 | 3 | 4 | 4 | 6 |
| Agentic | 8 | 8 | 0 | 5 | 6 |

The agentic loop drives `shape_mismatch` to **zero** (the Code Analyzer's explicit shape/
fusion reasoning pays off) but trades into `hallucinated_api` (8): once steered toward
writing more custom code, the model more often invents non-existent CUDA/torch symbols. This
localizes where the remaining headroom is — API grounding (better RAG, or constrained
decoding against a real symbol table) rather than planning.

### 5.5 Component ablation (best-of-4 backbone)

Each row flips exactly one agent off against the full best-of-4 system (30% correct / 73%
compile); the gap to the full system isolates that component's contribution.

| Variant | correct% | fast_1% | geo-mean | compile% | Δ correct vs full |
|---|---|---|---|---|---|
| **Agentic (full)** | 30 | 10 | 0.97× | 73 | — |
| − Code Analyzer | 20 | 10 | 0.99× | 63 | −10 |
| − Feedback Analyzer | 37 | 10 | 0.96× | 73 | **+7** |
| − Refinement loop (1 turn) | 20 | 10 | 0.98× | 70 | −10 |
| − best-of-n (greedy, n=1) | 3 | 0 | 0.95× | 23 | **−27** |
| − RAG Researcher | 20 | 7 | 0.97× | 77 | −10 |

Two results stand out. First, **best-of-n sampling is the load-bearing component by a wide
margin**: removing it (greedy, n=1) collapses the system from 30%→3% correct and 73%→23%
compile — *below even the naïve pre-fix agentic loop* (§5.7). Everything else the agents
contribute is second-order next to simply sampling several candidates and keeping the one
that survives real evaluation. Second, the **Feedback Analyzer is mildly counter-productive**
(removing it *raises* correctness to 37%): for a 7B model, our analytic re-framing of the
compiler/runtime error sometimes over-specifies the next attempt and steers it away from a
simple fix, whereas plain error text leaves more room to recover. The Code Analyzer and the
refinement loop each contribute ≈ +10pp correctness, confirming that planning and iteration
help — but only on top of the sampling backbone, not in place of it. RAG contributes +10pp
correctness as well (20% without it), though slightly *raising* compile rate when removed —
retrieval steers the model toward more custom code, which passes correctness more often but
builds less often.

### 5.6 Test-time compute: best-of-n scaling

Holding all agents on and varying only the number of candidates sampled and evaluated per
turn (the Evaluator keeps the best):

| best-of-n | correct% | fast_1% | geo-mean | compile% |
|---|---|---|---|---|
| n=1 (greedy) | 3 | 0 | 0.95× | 23 |
| n=2 | 27 | 13 | 0.98× | 53 |
| n=4 | 30 | 10 | 0.97× | 73 |
| n=8 | **33** | 13 | 0.98× | **80** |

The curve is steep and monotonic: compilation climbs 23%→53%→73%→80% and correctness
3%→27%→30%→33% as n grows from 1 to 8 (figure: `report/figures/bestof_scaling.png`). The
first doubling (n=1→2) is transformative — it is the difference between a system that almost
never compiles and one that solves a quarter of the tasks — and returns then diminish
(n=4→8 buys only +3pp correctness for 2× the evaluation cost). This is a clean
test-time-compute scaling result: for a small model on a verifiable task, *the single most
cost-effective intervention is to sample a handful of candidates and let the execution oracle
choose*, which extracts far more than elaborate single-sample reasoning. It also explains
the §5.7 recovery — the fix that rescued the agentic loop was, in essence, reallocating
compute from agent turns to best-of-n breadth.

### 5.7 Case study: why naïve agentic scaffolding underperformed zero-shot

Our first agentic implementation scored **3% correct / 33% compile — well below zero-shot's
30% / 70%**, despite three turns and four LLM agents per problem. Inspecting the
trajectories, the cause was *over-ambition*: the Analyzer and RAG brief consistently
recommended "tiling, shared memory, vectorized loads, warp reductions," and the 7B Generator
dutifully attempted hand-tiled CUDA it could not compile (e.g., misusing `load_inline`'s
signature, mismatching the pybind module name). The refinement loop did not recover, because
the feedback kept the model in the same ambitious mode; tellingly, the single-turn ablation
*beat* the full loop. Two changes fixed it: (i) **competence-matched prompting** — instruct
every agent to reach a correct, compiling kernel first (reuse torch ops, fuse only the
epilogue) and escalate only afterward; (ii) **spend the extra compute on best-of-n sampling**
rather than more agent turns. Together these moved agentic from 3%→30% correct and 33%→73%
compile. The lesson generalizes: scaffolding for a weak model must be *calibrated to its
competence ceiling*, or it converts safe single-shot successes into ambitious failures.

## 6. Discussion

- **Structure helps, but only when matched to the model and the problem.** The naïve
  "more agents = better" intuition is false here; the configuration that helped was *lighter*
  prompting plus *more sampling*. Guidance that respects the model's competence ceiling
  (guided's roofline rule, agentic's "simplest-correct-first") is what moved the numbers.
- **Difficulty-aware routing.** Because guided wins L1/L3 and agentic uniquely wins L2, an
  obvious system-level improvement is to route by level/structure rather than commit to one
  method — a cheap ensemble that would, on this set, exceed any single method's correctness.
- **The real objective for small models is correct fusion at ~1×.** Speedup is capped by
  vendor libraries; the differentiated skill is producing *correct* fused kernels, which is
  where the methods actually separate.

## 7. Limitations

- **Scale.** 30 tasks, one model, single seed. Differences of 1–2 problems are within noise;
  we therefore foreground the *qualitative* per-level structure and the large effects
  (the 3%→30% agentic recovery; guided's doubled fast_1) rather than small aggregate gaps.
  The non-sampling methods (zero-shot/guided/iterative) are near-deterministic (greedy
  decoding), so single-seed is defensible for them; sampling variance is characterized by the
  best-of-n sweep (§5.6).
- **One model family.** Findings about competence-matched scaffolding may shift for larger or
  differently-trained models; we make no cross-model claims.
- **fp32 / V100.** Results are specific to fp32 on Volta; cuBLAS/cuDNN dominance (and hence
  the speedup ceiling) could differ for fp16/bf16 or newer GPUs with tensor cores.
- **Benchmark scope.** Levels 1–3 first-10; harder/later problems are unexplored.
- **Evaluation robustness.** Generated CUDA can hang the GPU at runtime (an infinite loop
  or device deadlock), which a Python signal cannot interrupt. We therefore evaluate each
  candidate in a child process with a hard 240 s wall-clock timeout; a timed-out kernel is
  scored as a failure rather than wedging the run. This was necessary: without it, a single
  hanging kernel in the −RAG cell stalled a worker for 14 h. The timeout adds process-spawn
  overhead but bounds worst-case eval time and makes large sweeps safe to run unattended.

## 8. Conclusion

For a fixed local 7B coder model on KernelBench, *how* you wrap the model matters as much as
the prompt content — but more scaffolding is not monotonically better. Profiling-guided
prompting is the best single strategy (43% correct, 27% faster-than-baseline), a corrected
multi-agent loop is the only method that solves fused Level-2 problems, and the decisive
implementation choices were competence-matched ambition and best-of-n sampling rather than
additional agent turns. Correctness — not speed — is the binding constraint, because no
strategy lets a 7B model out-write cuBLAS/cuDNN on dense compute.

## Appendix A. Reproducibility

- One command: `sbatch experiments/run_all_experiments.sh` (core matrix + ablations + report).
- Single cell: `python experiments/run_experiment.py --model qwen --method <m>`.
- Report: `python experiments/make_report.py` (no GPU). Tables/figures in `report/`.
- Single source of truth for the matrix: `experiments/config.py`.

## Appendix B. Post-training byproduct

The full agentic run logs its trajectories as a small post-training dataset under
`experiments/runs/agentic_qwen/post_training/`: 10 SFT records (accepted final kernels), 25
preference pairs (better vs. worse candidate under real evaluation), 110 reflection records
(failure → fix directives), and 55 analysis records. This is a free byproduct of execution-
grounded agentic optimization and a candidate seed for fine-tuning a kernel-specialized model.
