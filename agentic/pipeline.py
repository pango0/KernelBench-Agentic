#!/usr/bin/env python3
"""The agentic kernel-optimisation loop.

    PyTorch/HIP kernel
          |
      Code Analyzer ----> RAG Researcher <----> Documentation
          |                     |
          v                     v
      Kernel Generator <--------+
          |
      Evaluator
          |
      Feedback Analyzer --> Data collector (post-training)
          |  (loop back)
          +--> Kernel Generator

`run_problem` realises exactly this graph for one task. Every box except the
Evaluator and Data collector is the shared LocalLLM in a different role (agents.py).
Each box can be switched off via AgenticConfig, which is how the paper's ablations
are produced from a single code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from agents import analyze_code, research, generate_kernels, analyze_feedback, feedback_directive
from datacollector import build_trajectory
from evaluator import format_eval_feedback


@dataclass
class AgenticConfig:
    max_turns: int = 3
    use_code_analyzer: bool = True
    use_rag: bool = True
    use_feedback_analyzer: bool = True
    best_of_n: int = 1
    rag_top_k: int = 5
    stop_on_correct: bool = False
    target_speedup: Optional[float] = None
    max_new_tokens: int = 4096
    analyzer_max_tokens: int = 1200
    research_max_tokens: int = 900
    feedback_max_tokens: int = 900
    collect_data: bool = False
    # Safety net: if no turn produced compiling code, ship the unoptimised reference
    # (ModelNew = Model) so the agent never emits a broken kernel. OFF by default — a
    # passthrough is trivially correct at ~1x, so counting it as a benchmark "win"
    # would distort the study; turn it on only for a deployment-style floor.
    fallback_to_reference: bool = False


def read_ref_arch(entry: dict) -> str:
    path = entry.get("reference_arch_path") or entry.get("ref_arch_path")
    if not path:
        raise KeyError("reference_arch_path missing in data entry")
    return Path(path).read_text(encoding="utf-8")


def _speedup(ev: dict | None) -> float:
    ev = ev or {}
    rt = ev.get("runtime") or -1.0
    ref = ev.get("ref_runtime") or -1.0
    return (ref / rt) if (rt > 0 and ref > 0) else -1.0


def _rank(ev: dict | None) -> tuple:
    ev = ev or {}
    correct = bool(ev.get("correctness"))
    compiled = bool(ev.get("compiled"))
    return (2 if correct else (1 if compiled else 0), _speedup(ev) if correct else 0.0)


def should_stop_early(ev: dict, target_speedup: Optional[float], stop_on_correct: bool) -> bool:
    if not ev or not ev.get("correctness"):
        return False
    if stop_on_correct and target_speedup is None:
        return True
    if target_speedup is None:
        return False
    return _speedup(ev) >= target_speedup


def pick_best_turn(turns: list) -> Optional[dict]:
    best, best_rank = None, (-1, -1.0)
    for t in turns:
        if not t.get("extracted_code"):
            continue
        r = _rank(t.get("eval"))
        if r >= best_rank:
            best_rank, best = r, t
    if best is not None:
        return best
    return turns[-1] if turns else None


def _select_candidate(candidates: list[dict], ref_src: str,
                      evaluate: Callable[[str, str], dict]) -> tuple[dict, list[dict]]:
    """Evaluate each candidate (best-of-n) and return (chosen, summaries)."""
    coded = [c for c in candidates if c.get("extracted_code")]
    if not coded:
        return candidates[0] if candidates else {"raw_response": "", "extracted_code": None}, []

    summaries = []
    best, best_rank = None, (-2, -1.0)
    for c in coded:
        ev = evaluate(ref_src, c["extracted_code"])
        c["eval"] = ev
        summaries.append({
            "compiled": bool(ev.get("compiled")), "correct": bool(ev.get("correctness")),
            "speedup": _speedup(ev),
        })
        r = _rank(ev)
        if r > best_rank:
            best_rank, best = r, c
    return best, (summaries if len(coded) > 1 else [])


def _status_from_eval(code: Optional[str], ev: Optional[dict]) -> str:
    if not code:
        return "no_code"
    if not ev:
        return "error"
    if ev.get("correctness"):
        return "correct"
    return "compile_failed" if not ev.get("compiled") else "incorrect"


def run_problem(llm, evaluate: Callable[[str, str], dict], retriever,
                level_key: str, pid: str, entry: dict, cfg: AgenticConfig) -> dict:
    """Run the full agentic loop for one KernelBench task. Returns a result entry."""
    base_prompt = entry["prompt"]
    ref_src = read_ref_arch(entry)
    tag = f"L{level_key} P{pid}"

    # --- Code Analyzer ---
    analysis: dict = {}
    if cfg.use_code_analyzer:
        print(f"  {tag} agentic: analysing code...", flush=True)
        try:
            analysis = analyze_code(llm, ref_src, cfg.analyzer_max_tokens)
        except Exception as e:  # noqa: BLE001
            print(f"  {tag} analyzer error: {e}", flush=True)

    # --- RAG Researcher <-> Documentation ---
    rb = {"brief": "", "sources": [], "queries": []}
    if cfg.use_rag and retriever is not None:
        print(f"  {tag} agentic: researching docs...", flush=True)
        try:
            rb = research(llm, retriever, analysis, top_k=cfg.rag_top_k,
                          max_new_tokens=cfg.research_max_tokens)
        except Exception as e:  # noqa: BLE001
            print(f"  {tag} researcher error: {e}", flush=True)
    brief = rb.get("brief", "")

    # --- Generate / Evaluate / (Feedback) loop ---
    turns: list[dict] = []
    prev_code = ""
    feedback_text = ""

    for turn in range(cfg.max_turns):
        kind = "initial" if turn == 0 else "refine"
        print(f"  {tag} agentic: generating ({kind}, turn {turn})...", flush=True)
        try:
            cands = generate_kernels(
                llm, base_prompt, analysis, brief,
                previous_code=prev_code, feedback=feedback_text,
                n=cfg.best_of_n, max_new_tokens=cfg.max_new_tokens,
            )
        except Exception as e:  # noqa: BLE001
            turns.append({"turn": turn, "status": "error", "error": str(e)})
            break

        chosen, cand_summ = _select_candidate(cands, ref_src, evaluate)
        code = chosen.get("extracted_code")
        ev = chosen.get("eval")
        rec: dict = {
            "turn": turn,
            "prompt_kind": kind,
            "raw_response": chosen.get("raw_response", ""),
            "extracted_code": code,
            "eval": ev,
            "feedback": format_eval_feedback(ev, turn) if ev else "No code extracted.",
            "status": _status_from_eval(code, ev),
        }
        if cand_summ:
            rec["candidates"] = cand_summ
        turns.append(rec)

        if code:
            prev_code = code

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        if ev and should_stop_early(ev, cfg.target_speedup, cfg.stop_on_correct):
            print(f"  {tag} early stop after turn {turn}", flush=True)
            break

        # --- Feedback Analyzer (prepares the next generation) ---
        if turn < cfg.max_turns - 1:
            if cfg.use_feedback_analyzer:
                try:
                    diag = analyze_feedback(llm, code or "", rec["feedback"], analysis,
                                            cfg.feedback_max_tokens)
                    rec["diagnosis"] = diag
                    feedback_text = feedback_directive(diag, rec["feedback"])
                except Exception as e:  # noqa: BLE001
                    print(f"  {tag} feedback error: {e}", flush=True)
                    feedback_text = rec["feedback"]
            else:
                feedback_text = rec["feedback"]

    # --- Optional reference fallback (deployment floor; off by default) ---
    if cfg.fallback_to_reference and not any((t.get("eval") or {}).get("compiled") for t in turns):
        print(f"  {tag} no compiling turn -> reference fallback", flush=True)
        fb_code = ref_src.rstrip() + (
            "\n\n# Fallback: the agent could not produce a compiling custom kernel; "
            "ship the unoptimised reference unchanged.\nModelNew = Model\n"
        )
        fb_eval = evaluate(ref_src, fb_code)
        turns.append({
            "turn": len(turns),
            "prompt_kind": "fallback_reference",
            "raw_response": "",
            "extracted_code": fb_code,
            "eval": fb_eval,
            "feedback": format_eval_feedback(fb_eval, len(turns)),
            "status": "fallback_reference",
        })

    out = _build_result(entry, base_prompt, analysis, rb, turns)
    if cfg.collect_data:
        out["_post_training"] = build_trajectory(entry, base_prompt, analysis, brief, turns)
    return out


def _build_result(entry: dict, base_prompt: str, analysis: dict, rb: dict, turns: list) -> dict:
    best = pick_best_turn(turns)
    out = {
        **entry,
        "base_prompt": base_prompt,
        "analysis": {k: analysis.get(k) for k in
                     ("operations", "bottlenecks", "fusion", "strategy", "research_queries")}
        if analysis else None,
        "research_sources": rb.get("sources", []),
        "research_brief": rb.get("brief", ""),
        "turns": turns,
        "num_turns": len(turns),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if best:
        out["final_turn"] = best.get("turn")
        out["final_status"] = best.get("status")
        out["extracted_code"] = best.get("extracted_code")
        out["final_eval"] = best.get("eval")
    any_correct = any((t.get("eval") or {}).get("correctness") for t in turns)
    if best and best.get("extracted_code"):
        out["status"] = "success" if any_correct else "best_effort"
    else:
        out["status"] = "failed"
    return out
