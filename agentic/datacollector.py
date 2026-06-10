#!/usr/bin/env python3
"""Data collector for post-training.

The agentic loop produces rich supervision for free. For each solved (or attempted)
problem we emit three kinds of records, written as JSONL so they can feed standard
post-training recipes:

  sft.jsonl          (prompt -> best correct kernel)        supervised fine-tuning
  preferences.jsonl  (prompt, chosen, rejected)             DPO / reward modelling
  reflections.jsonl  (code, eval feedback, diagnosis, fix)  reflection / process data

To avoid multi-process write races, workers return their trajectory and the single
main process appends through one collector instance.
"""

from __future__ import annotations

import json
from pathlib import Path


def _speedup(ev: dict | None) -> float:
    ev = ev or {}
    rt = ev.get("runtime") or -1.0
    ref = ev.get("ref_runtime") or -1.0
    return (ref / rt) if (rt > 0 and ref > 0) else -1.0


def build_trajectory(entry: dict, base_prompt: str, analysis: dict, brief: str,
                     turns: list[dict]) -> dict:
    """Distil a problem's turns into post-training records (pure, no I/O)."""
    sft, prefs, reflections = [], [], []

    scored = []
    for t in turns:
        code = t.get("extracted_code")
        if not code:
            continue
        ev = t.get("eval") or {}
        scored.append({
            "turn": t.get("turn"),
            "code": code,
            "correct": bool(ev.get("correctness")),
            "compiled": bool(ev.get("compiled")),
            "speedup": _speedup(ev),
        })

    meta = {
        "problem_name": entry.get("problem_name"),
        "problem_id": entry.get("problem_id"),
    }

    # SFT: best correct (fastest) kernel for this prompt
    correct = [s for s in scored if s["correct"]]
    best = max(correct, key=lambda s: s["speedup"]) if correct else None
    if best:
        sft.append({**meta, "prompt": base_prompt, "completion": best["code"],
                    "speedup": best["speedup"]})

    # Preferences: best vs each strictly worse candidate (rank: correct+fast > correct > compiled > broken)
    def rank(s):
        return (2 if s["correct"] else (1 if s["compiled"] else 0),
                s["speedup"] if s["correct"] else 0.0)
    if len(scored) >= 2:
        best_overall = max(scored, key=rank)
        for s in scored:
            if s is best_overall:
                continue
            if rank(s) < rank(best_overall):
                prefs.append({**meta, "prompt": base_prompt,
                              "chosen": best_overall["code"], "rejected": s["code"]})

    # Reflections: (broken code -> feedback -> diagnosis -> improved code) chains
    for prev, nxt in zip(turns, turns[1:]):
        if not prev.get("extracted_code") or not nxt.get("extracted_code"):
            continue
        reflections.append({
            **meta,
            "code": prev.get("extracted_code"),
            "feedback": prev.get("feedback", ""),
            "diagnosis": (prev.get("diagnosis") or {}).get("raw", ""),
            "improved_code": nxt.get("extracted_code"),
        })

    return {"sft": sft, "preferences": prefs, "reflections": reflections,
            "analysis": analysis_summary(analysis, brief, meta)}


def analysis_summary(analysis: dict, brief: str, meta: dict) -> dict | None:
    if not analysis and not brief:
        return None
    return {**meta, "analysis_raw": (analysis or {}).get("raw", ""), "research_brief": brief}


class PostTrainingCollector:
    """Single-writer JSONL appender for post-training records."""

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.counts = {"sft": 0, "preferences": 0, "reflections": 0, "analysis": 0}

    def _append(self, name: str, rows: list[dict]):
        if not rows:
            return
        with (self.out_dir / f"{name}.jsonl").open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.counts[name] += len(rows)

    def add(self, trajectory: dict):
        if not trajectory:
            return
        self._append("sft", trajectory.get("sft", []))
        self._append("preferences", trajectory.get("preferences", []))
        self._append("reflections", trajectory.get("reflections", []))
        a = trajectory.get("analysis")
        if a:
            self._append("analysis", [a])

    def summary(self) -> dict:
        return dict(self.counts)
