#!/usr/bin/env python3
"""The four LLM agent roles of the agentic kernel-optimization loop.

    Code Analyzer    -> understands the reference architecture, names the bottlenecks,
                        and emits targeted research queries.
    RAG Researcher   -> retrieves grounding from the documentation corpus and
                        synthesises an optimisation brief (with citations).
    Kernel Generator -> writes / rewrites the optimised `ModelNew` CUDA-HIP kernel.
    Feedback Analyzer-> turns raw evaluation output into a structured diagnosis and
                        a concrete directive for the next generation.

All roles share one `LocalLLM` (see llm.py); they differ only in system prompt and
the information handed to them. Parsing is deliberately lenient: if a model ignores
the requested section markers we degrade gracefully instead of failing the run.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Code extraction (shared)
# ---------------------------------------------------------------------------

def extract_first_code(text: str) -> Optional[str]:
    if not text or not text.strip():
        return None
    trimmed = text.strip()
    for lang in ("python", ""):
        pattern = rf"```{lang}\s*\n(.*?)```" if lang else r"```(.*?)```"
        m = re.search(pattern, trimmed, re.DOTALL | re.IGNORECASE)
        if m:
            code = m.group(1).strip()
            for hdr in ("python", "cpp"):
                if code.startswith(hdr):
                    code = code[len(hdr):].strip()
            return code
    if "class ModelNew" in trimmed or ("import torch" in trimmed and "def get_inputs" in trimmed):
        return trimmed
    return None


def _section(text: str, header: str) -> str:
    """Pull the body under a `## HEADER` marker (case-insensitive) up to the next `##`."""
    m = re.search(rf"#+\s*{re.escape(header)}\s*\n(.*?)(?=\n#+\s|\Z)", text,
                  re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# 1. Code Analyzer
# ---------------------------------------------------------------------------

ANALYZER_SYSTEM = (
    "You are a senior GPU performance engineer. You analyse PyTorch reference "
    "modules and plan how to replace their operators with fast custom CUDA/HIP "
    "kernels. You are precise, concrete, and you never write the final kernel here."
)

ANALYZER_USER = """\
Analyse the following reference architecture for custom-kernel optimisation.

=== Reference architecture ===
```python
{ref_src}
```

Produce your analysis using EXACTLY these section headers:

## OPERATIONS
List the dominant tensor operations (matmul, conv, elementwise, reduction, norm,
attention, ...) and their rough arithmetic intensity / memory behaviour.

## BOTTLENECKS
Identify the 1-3 operations most worth replacing with a custom kernel and why.

## FUSION
Concrete operator-fusion opportunities (e.g. conv+bias+relu, matmul+bias+gelu,
online-softmax attention). Say which tensors stay in registers / shared memory.

## STRATEGY
A short, ordered optimisation plan (tiling, coalescing, shared memory, vectorised
loads, warp reductions, ...). No code.

## RESEARCH_QUERIES
3-5 short search queries (one per line, no numbering) that would retrieve the most
useful optimisation documentation for implementing this plan.
"""


def analyze_code(llm, ref_src: str, max_new_tokens: int = 1200) -> dict:
    msgs = [
        {"role": "system", "content": ANALYZER_SYSTEM},
        {"role": "user", "content": ANALYZER_USER.format(ref_src=ref_src.strip())},
    ]
    raw = llm.chat(msgs, max_new_tokens=max_new_tokens, temperature=0.3)
    queries_blob = _section(raw, "RESEARCH_QUERIES")
    queries = [q.strip(" -*0123456789.").strip() for q in queries_blob.splitlines() if q.strip()]
    queries = [q for q in queries if len(q) > 3][:5]
    return {
        "raw": raw,
        "operations": _section(raw, "OPERATIONS"),
        "bottlenecks": _section(raw, "BOTTLENECKS"),
        "fusion": _section(raw, "FUSION"),
        "strategy": _section(raw, "STRATEGY"),
        "research_queries": queries,
    }


def analysis_brief(analysis: dict) -> str:
    """Compact human-readable analysis to feed downstream agents."""
    if not analysis:
        return ""
    parts = []
    for key, label in (("bottlenecks", "Bottlenecks"), ("fusion", "Fusion"),
                       ("strategy", "Strategy")):
        if analysis.get(key):
            parts.append(f"{label}:\n{analysis[key]}")
    return "\n\n".join(parts) if parts else analysis.get("raw", "").strip()


# ---------------------------------------------------------------------------
# 2. RAG Researcher
# ---------------------------------------------------------------------------

RESEARCHER_SYSTEM = (
    "You are a research assistant for GPU kernel engineering. You are given "
    "retrieved documentation excerpts and must distil ONLY the techniques that are "
    "directly useful for the task. Ground every claim in the excerpts; never invent "
    "APIs. Cite the source tag (e.g. [memory_optimization.md#2]) after each point."
)

RESEARCHER_USER = """\
Optimisation context:
{context}

Retrieved documentation excerpts:
{excerpts}

Write an OPTIMISATION BRIEF: 4-8 bullet points of concrete, citable techniques the
kernel engineer should apply for this task. Keep it tight and actionable.
"""


def research(llm, retriever, analysis: dict, queries: Optional[list[str]] = None,
             top_k: int = 5, max_new_tokens: int = 900) -> dict:
    """Retrieve grounding chunks and synthesise an optimisation brief."""
    if retriever is None:
        return {"brief": "", "sources": [], "queries": []}

    queries = queries or (analysis.get("research_queries") if analysis else None) or []
    if not queries:
        # fall back to a query built from the analysis / bottlenecks
        queries = [analysis_brief(analysis)[:300] or "cuda kernel optimization fusion tiling"]

    seen: dict[str, object] = {}
    for q in queries:
        for chunk, _score in retriever.search(q, k=top_k):
            seen.setdefault(chunk.cite, chunk)
    chunks = list(seen.values())[: max(top_k, 6)]
    if not chunks:
        return {"brief": "", "sources": [], "queries": queries}

    excerpts = "\n\n".join(f"[{c.cite}]\n{c.text}" for c in chunks)
    context = analysis_brief(analysis) or "Optimise the given PyTorch module with custom CUDA/HIP kernels."
    msgs = [
        {"role": "system", "content": RESEARCHER_SYSTEM},
        {"role": "user", "content": RESEARCHER_USER.format(context=context[:1500], excerpts=excerpts[:6000])},
    ]
    brief = llm.chat(msgs, max_new_tokens=max_new_tokens, temperature=0.3)
    return {"brief": brief.strip(), "sources": [c.cite for c in chunks], "queries": queries}


# ---------------------------------------------------------------------------
# 3. Kernel Generator
# ---------------------------------------------------------------------------

GENERATOR_SYSTEM = (
    "You are an expert CUDA/HIP kernel engineer. You replace PyTorch operators with "
    "correct, high-performance custom kernels using torch.utils.cpp_extension "
    "(load_inline) or Triton. You output ONLY a single ```python``` code block that "
    "defines class ModelNew(nn.Module) with the same interface as Model, plus "
    "get_inputs() and get_init_inputs(). No prose, no tests."
)


def _generator_user(base_prompt: str, analysis: dict, brief: str,
                    previous_code: str = "", feedback: str = "") -> str:
    blocks = [base_prompt.rstrip()]
    if analysis:
        ab = analysis_brief(analysis)
        if ab:
            blocks.append("=== Optimisation analysis ===\n" + ab)
    if brief:
        blocks.append("=== Grounded optimisation brief (from documentation) ===\n" + brief)
    if previous_code:
        blocks.append("=== Your previous implementation ===\n```python\n" + previous_code.strip() + "\n```")
    if feedback:
        blocks.append("=== Evaluation feedback / required fixes ===\n" + feedback.strip())
    blocks.append(
        "Now output the complete optimised ModelNew in ONE ```python``` block. "
        "It must compile and match the reference numerically before being fast."
    )
    return "\n\n".join(blocks)


def generate_kernels(llm, base_prompt: str, analysis: dict, brief: str,
                     previous_code: str = "", feedback: str = "",
                     n: int = 1, max_new_tokens: int = 4096) -> list[dict]:
    """Generate n candidate kernels. n>1 samples with temperature for best-of-n."""
    user = _generator_user(base_prompt, analysis, brief, previous_code, feedback)
    msgs = [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user", "content": user},
    ]
    candidates = []
    for i in range(max(1, n)):
        # first sample greedy-ish for stability; extra samples are more diverse
        temp = 0.2 if (i == 0 and n == 1) else 0.7
        raw = llm.chat(msgs, max_new_tokens=max_new_tokens, temperature=temp,
                       do_sample=(n > 1 or temp > 0))
        candidates.append({"raw_response": raw, "extracted_code": extract_first_code(raw)})
    return candidates


# ---------------------------------------------------------------------------
# 4. Feedback Analyzer
# ---------------------------------------------------------------------------

FEEDBACK_SYSTEM = (
    "You are a debugging and performance triage expert for GPU kernels. Given an "
    "implementation and its evaluation result, you produce a crisp diagnosis and a "
    "concrete, prioritised fix list for the next attempt. You do NOT write the full "
    "kernel; you direct the fix."
)

FEEDBACK_USER = """\
Implementation under review:
```python
{code}
```

Raw evaluation result:
{feedback}

{analysis_ctx}

Respond using EXACTLY these headers:

## CATEGORY
One of: compilation, correctness, performance, no_code.

## ROOT_CAUSE
The most likely root cause, referencing specific lines/APIs where possible.

## FIXES
2-4 concrete, ordered changes to make in the next implementation.

## DIRECTIVE
One sentence telling the generator exactly what to do next.
"""


def analyze_feedback(llm, code: str, raw_feedback: str, analysis: dict,
                     max_new_tokens: int = 900) -> dict:
    analysis_ctx = ""
    ab = analysis_brief(analysis)
    if ab:
        analysis_ctx = "Optimisation analysis for reference:\n" + ab[:1200]
    msgs = [
        {"role": "system", "content": FEEDBACK_SYSTEM},
        {"role": "user", "content": FEEDBACK_USER.format(
            code=(code or "# (no code)")[:6000],
            feedback=(raw_feedback or "(none)")[:4000],
            analysis_ctx=analysis_ctx,
        )},
    ]
    raw = llm.chat(msgs, max_new_tokens=max_new_tokens, temperature=0.3)
    return {
        "raw": raw,
        "category": _section(raw, "CATEGORY"),
        "root_cause": _section(raw, "ROOT_CAUSE"),
        "fixes": _section(raw, "FIXES"),
        "directive": _section(raw, "DIRECTIVE"),
    }


def feedback_directive(diag: dict, raw_feedback: str) -> str:
    """Combine the structured diagnosis into the text fed back to the generator."""
    if not diag:
        return raw_feedback
    parts = []
    if diag.get("root_cause"):
        parts.append("Root cause: " + diag["root_cause"])
    if diag.get("fixes"):
        parts.append("Apply these fixes:\n" + diag["fixes"])
    if diag.get("directive"):
        parts.append("Directive: " + diag["directive"])
    combined = "\n\n".join(parts).strip()
    # always keep the raw measurements visible too
    return (raw_feedback.strip() + "\n\n" + combined).strip() if combined else raw_feedback
