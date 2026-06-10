#!/usr/bin/env python3
"""
Single source of truth for the KernelBench local-model study experiment matrix.

The factorial is 1 model x 4 methods = 4 cells, each run over 30 tasks
(levels 1-3, first 10 problems each). This module defines:

  - MODELS:  the local model(s) under test (Qwen, run on-GPU via HuggingFace)
  - METHODS: the 4 prompting/interaction paradigms
  - run-dir naming (with backward-compatible aliases for already-completed runs)
  - canonical filesystem paths used by the orchestrator and the report builder
  - a heuristic error-taxonomy classifier used for qualitative analysis

Nothing here touches the GPU or the network; it is safe to import anywhere.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RUNS_DIR = EXPERIMENTS_DIR / "runs"          # orchestrator artifacts per cell
REPORT_DIR = REPO_ROOT / "report"            # aggregated report output

KB_DIR = REPO_ROOT / "KernelBench"
KB_RUNS_DIR = KB_DIR / "runs"                # where eval reads kernels / writes results
KB_EVAL_SCRIPT = KB_DIR / "scripts" / "eval_from_generations.py"
KB_ANALYSIS_SCRIPT = KB_DIR / "scripts" / "benchmark_eval_analysis.py"

DATA_ZEROSHOT = REPO_ROOT / "data.json"
DATA_GUIDED = REPO_ROOT / "data_guided.json"

# Evaluation hardware (Tesla V100-SXM2-32GB on this cluster).
HARDWARE = "V100_SXM2_32GB"
GPU_ARCH = "Volta"
BASELINE_NAME = "baseline_time_torch"
BASELINE_FILE = KB_DIR / "results" / "timing" / HARDWARE / f"{BASELINE_NAME}.json"

LEVELS = [1, 2, 3]

# Problems used per level (the first N of each level). To change the study size,
# edit ONLY this line, then regenerate the three derived artifacts (see CLAUDE.md
# §9.6): data.json / data_guided.json, the guided profiles, and the V100 baseline.
PROBLEMS_PER_LEVEL = 10
PROBLEM_IDS = list(range(1, PROBLEMS_PER_LEVEL + 1))
SUBSET = f"(1,{PROBLEMS_PER_LEVEL})"     # passed to KernelBench eval as subset

# Speedup thresholds reported by benchmark_eval_analysis.py (fast_p metric).
FASTP_THRESHOLDS = ["0.0", "0.5", "0.8", "1.0", "1.5", "2.0"]


# ---------------------------------------------------------------------------
# Build toolchain (CUDA + host compiler)
# ---------------------------------------------------------------------------
#
# PyTorch's cpp_extension shells out to $CUDA_HOME/bin/nvcc and the host `c++`
# to build the generated kernels. Without a CUDA toolkit and a GCC>=9 on PATH
# every kernel fails to compile ("CUDA_HOME not set" / "cuda_runtime.h: No such
# file" / "GCC 9 or later"). The SLURM wrappers load the proper Lmod modules
# (see experiments/toolchain.sh); the helper below is the equivalent fallback
# for interactive runs that only activated conda. These are the TWCC install
# locations; CUDA 12.3 matches the installed torch (cu121, major 12).

CUDA_TOOLKIT_CANDIDATES = [
    "/work/HPC_SYS/twnia2/pkg-rocky8/nvidia/cuda/cuda-12.3",
]
GCC_TOOLSET_CANDIDATES = [
    "/work/HPC_SYS/devtoolset/devtoolset-10/root",  # GCC 10.2.1
]


def _gcc_major(env: dict) -> int:
    gcc = shutil.which("gcc", path=env.get("PATH"))
    if not gcc:
        return 0
    try:
        out = subprocess.check_output([gcc, "-dumpversion"], text=True).strip()
        return int(out.split(".")[0])
    except Exception:
        return 0


def _prepend_path(env: dict, key: str, value: str) -> None:
    cur = env.get(key, "")
    env[key] = f"{value}:{cur}" if cur else value


def ensure_build_toolchain(env: dict | None = None) -> dict:
    """Ensure a CUDA toolkit and a GCC>=9 host compiler are reachable via `env`.

    No-op when the launching shell already loaded them (the common SLURM path).
    Otherwise it points CUDA_HOME/PATH/LD_LIBRARY_PATH at the cluster installs so
    `python experiments/run_experiment.py` works after a bare `conda activate`.
    Mutates and returns `env` (defaults to os.environ).
    """
    if env is None:
        env = os.environ

    cuda_home = env.get("CUDA_HOME") or env.get("CUDA_PATH")
    nvcc_ok = bool(cuda_home and os.path.exists(os.path.join(cuda_home, "bin", "nvcc"))) \
        and bool(shutil.which("nvcc", path=env.get("PATH")))
    if not nvcc_ok:
        for cand in CUDA_TOOLKIT_CANDIDATES:
            if os.path.exists(os.path.join(cand, "bin", "nvcc")):
                env["CUDA_HOME"] = env["CUDA_PATH"] = env["CUDA_ROOT"] = cand
                _prepend_path(env, "PATH", os.path.join(cand, "bin"))
                _prepend_path(env, "LD_LIBRARY_PATH", os.path.join(cand, "lib64"))
                break

    if _gcc_major(env) < 9:
        for cand in GCC_TOOLSET_CANDIDATES:
            gccbin = os.path.join(cand, "usr", "bin")
            if os.path.exists(os.path.join(gccbin, "gcc")):
                _prepend_path(env, "PATH", gccbin)
                _prepend_path(env, "LD_LIBRARY_PATH", os.path.join(cand, "usr", "lib"))
                _prepend_path(env, "LD_LIBRARY_PATH", os.path.join(cand, "usr", "lib64"))
                break

    # Pin the host compiler to that GCC>=9 so an inherited CC/CXX (e.g. nvc/nvc++
    # from an nvhpc module exported into the job via --export=ALL) cannot win and
    # drag in a pre-GCC-9 base. torch uses $CXX for the host .cpp and $CC for
    # nvcc's -ccbin (cpp_extension.py), so both must point at gcc.
    if _gcc_major(env) >= 9:
        gcc = shutil.which("gcc", path=env.get("PATH"))
        gxx = shutil.which("g++", path=env.get("PATH"))
        if gcc and gxx:
            env["CC"], env["CXX"] = gcc, gxx
            env["CUDAHOSTCXX"] = gxx

    return env


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    tag: str                 # short id used in run dirs / report (e.g. qwen)
    display: str             # human-readable name for tables
    kind: str                # "local" (HuggingFace, on-GPU)
    model_name: str          # HuggingFace model id


MODELS: dict[str, ModelSpec] = {
    "qwen": ModelSpec(
        tag="qwen",
        display="Qwen2.5-Coder-7B-Instruct",
        kind="local",
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    ),
}


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MethodSpec:
    name: str                # zeroshot | guided | iterative | agentic | agentic_*
    display: str
    main: Path               # path to the method's main.py
    data: Path               # input dataset
    inline_eval: bool        # does generation already run GPU eval per turn?
    multi_turn: bool         # does it loop with feedback?
    # extra CLI args appended for local generation
    local_args: list[str] = field(default_factory=list)
    ablation: bool = False   # True for agentic ablation variants (not in the core matrix)


# Shared launch args for the agentic multi-agent method (and its ablations).
def _agentic_args(*, max_turns: int = 3, extra: tuple[str, ...] = ()) -> list[str]:
    return ["--num-gpus", "8", "--gpus-per-worker", "2",
            "--max-turns", str(max_turns), "--max-new-tokens", "4096",
            "--num-correct-trials", "5", "--num-perf-trials", "10", *extra]


AGENTIC_MAIN = REPO_ROOT / "agentic" / "main.py"


METHODS: dict[str, MethodSpec] = {
    "zeroshot": MethodSpec(
        name="zeroshot",
        display="Zero-Shot",
        main=REPO_ROOT / "zeroshot" / "main.py",
        data=DATA_ZEROSHOT,
        inline_eval=False,
        multi_turn=False,
        local_args=["--batch-size", "4", "--max-new-tokens", "4096"],
    ),
    "guided": MethodSpec(
        name="guided",
        display="Guided",
        main=REPO_ROOT / "guided" / "main.py",
        data=DATA_GUIDED,
        inline_eval=False,
        multi_turn=False,
        local_args=["--batch-size", "4", "--max-new-tokens", "4096"],
    ),
    "iterative": MethodSpec(
        name="iterative",
        display="Iterative",
        main=REPO_ROOT / "iterative" / "main.py",
        data=DATA_ZEROSHOT,
        inline_eval=True,
        multi_turn=True,
        local_args=[
            "--num-gpus", "8", "--gpus-per-worker", "2",
            "--max-turns", "3", "--max-new-tokens", "2048",
            "--num-perf-trials", "10",
        ],
    ),
    "agentic": MethodSpec(
        name="agentic",
        display="Agentic",
        main=AGENTIC_MAIN,
        data=DATA_ZEROSHOT,
        inline_eval=True,
        multi_turn=True,
        # full multi-agent system: Code Analyzer + RAG Researcher + Feedback
        # Analyzer + post-training data collection.
        local_args=_agentic_args(extra=("--collect-data",)),
    ),
}


# ---------------------------------------------------------------------------
# Agentic ablations (paper study)
# ---------------------------------------------------------------------------
#
# Each variant flips exactly one component of the full agentic system so its
# contribution can be isolated. They share agentic/main.py and the report's
# ablation section; they are NOT part of the core model x method matrix.

AGENTIC_ABLATIONS: dict[str, MethodSpec] = {
    "agentic_no_rag": MethodSpec(
        name="agentic_no_rag", display="Agentic − RAG", main=AGENTIC_MAIN,
        data=DATA_ZEROSHOT, inline_eval=True, multi_turn=True, ablation=True,
        local_args=_agentic_args(extra=("--no-rag",)),
    ),
    "agentic_no_analyzer": MethodSpec(
        name="agentic_no_analyzer", display="Agentic − Code Analyzer", main=AGENTIC_MAIN,
        data=DATA_ZEROSHOT, inline_eval=True, multi_turn=True, ablation=True,
        local_args=_agentic_args(extra=("--no-code-analyzer",)),
    ),
    "agentic_no_feedback": MethodSpec(
        name="agentic_no_feedback", display="Agentic − Feedback Analyzer", main=AGENTIC_MAIN,
        data=DATA_ZEROSHOT, inline_eval=True, multi_turn=True, ablation=True,
        local_args=_agentic_args(extra=("--no-feedback-analyzer",)),
    ),
    "agentic_single_turn": MethodSpec(
        name="agentic_single_turn", display="Agentic (1 turn, no loop)", main=AGENTIC_MAIN,
        data=DATA_ZEROSHOT, inline_eval=True, multi_turn=False, ablation=True,
        local_args=_agentic_args(max_turns=1),
    ),
    "agentic_bestof2": MethodSpec(
        name="agentic_bestof2", display="Agentic + best-of-2", main=AGENTIC_MAIN,
        data=DATA_ZEROSHOT, inline_eval=True, multi_turn=True, ablation=True,
        local_args=_agentic_args(extra=("--best-of-n", "2")),
    ),
}


def method_spec(name: str) -> MethodSpec:
    """Look up a method by name across the core matrix and the agentic ablations."""
    if name in METHODS:
        return METHODS[name]
    return AGENTIC_ABLATIONS[name]


def all_method_names() -> list[str]:
    return list(METHODS) + list(AGENTIC_ABLATIONS)


# ---------------------------------------------------------------------------
# Run-dir naming (with backward-compatible aliases)
# ---------------------------------------------------------------------------

# The first Qwen runs were stored under model-agnostic names. Keep reading them
# in place so the report includes them without re-running.
LEGACY_RUN_DIRS: dict[tuple[str, str], str] = {
    ("qwen", "zeroshot"): "zero_shot",
    ("qwen", "guided"): "guided",
    ("qwen", "iterative"): "iterative",
}


def run_dir_name(model_tag: str, method: str) -> str:
    """Canonical KernelBench run-dir name for a (model, method) cell.

    Falls back to a legacy alias when one exists AND the canonical dir has not
    been created yet, so existing results stay usable and new runs are tidy.
    """
    canonical = f"{method}_{model_tag}"
    legacy = LEGACY_RUN_DIRS.get((model_tag, method))
    if legacy is None:
        return canonical
    if (KB_RUNS_DIR / canonical).exists():
        return canonical
    return legacy


def all_cells() -> list[tuple[str, str]]:
    """Every (model_tag, method) pair in the factorial design."""
    return [(m, meth) for m in MODELS for meth in METHODS]


# ---------------------------------------------------------------------------
# Error taxonomy (heuristic, for qualitative analysis in the report)
# ---------------------------------------------------------------------------

ERROR_CATEGORIES = [
    "no_code",            # model produced no extractable code
    "compilation",        # nvcc / cpp_extension build failure
    "hallucinated_api",   # referenced a non-existent symbol / attribute / import
    "memory",             # CUDA OOM / illegal memory access / misaligned address
    "shape_mismatch",     # tensor shape / size mismatch
    "wrong_output",       # compiled & ran but outputs diverge from reference
    "runtime",            # other runtime exception
    "timeout",            # eval timed out
    "slow",               # correct but slower than the PyTorch baseline
    "unknown",
]


def classify_error(status: str | None, eval_dict: dict | None) -> str:
    """Map a generation status + KernelBench eval dict to one taxonomy bucket.

    Used only for descriptive statistics; it is intentionally conservative and
    string-matches the error text that KernelBench surfaces.
    """
    if status in ("no_code", "no_code_extracted"):
        return "no_code"

    ev = eval_dict or {}
    compiled = bool(ev.get("compiled"))
    correct = bool(ev.get("correctness"))

    if correct:
        rt = ev.get("runtime") or -1.0
        ref = ev.get("ref_runtime") or -1.0
        if rt > 0 and ref > 0 and (ref / rt) < 1.0:
            return "slow"
        return "wrong_output"  # placeholder; callers treat correct cells separately

    meta = ev.get("metadata") or {}
    blob = " ".join(
        str(meta.get(k, "")) for k in
        ("compilation_error", "runtime_error", "cuda_error", "other_error",
         "correctness_issue")
    )
    blob += " " + str(ev.get("error", ""))
    low = blob.lower()

    if any(s in low for s in ("out of memory", "illegal memory access",
                              "misaligned address", "cudamalloc")):
        return "memory"
    if "timeout" in low or "timed out" in low:
        return "timeout"
    if any(s in low for s in ("has no attribute", "is not defined", "no module named",
                              "undefined symbol", "unknown identifier",
                              "name '", "cannot find")):
        return "hallucinated_api"
    if any(s in low for s in ("size mismatch", "shape", "must match",
                              "dimension", "expected", "got tensor")):
        if not compiled:
            return "compilation"
        return "shape_mismatch"
    if not compiled:
        return "compilation"
    if not correct:
        return "wrong_output"
    return "unknown"


# Convenience for short tags in tables/filenames.
def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
