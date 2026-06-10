#!/usr/bin/env python3
"""
Unified experiment runner for ONE (model, method) cell of the KernelBench study.

This is the single entry point referenced in the report workflow. Given a model
and a method it runs the whole pipeline end to end and saves everything a report
needs:

    generate  ->  stage kernels  ->  GPU eval  ->  analysis  ->  summary

Outputs (under experiments/runs/<run_dir>/):
    generate.log          full stdout/stderr of the generation step
    eval.log              full stdout/stderr of KernelBench eval + analysis
    gen_results.json      raw generation results (prompts, raw responses, code)
    eval_results_level*.json   per-problem correctness/runtime from KernelBench
    analysis_level*.json  KernelBench aggregate metrics per level
    summary.json          one tidy record for this cell (consumed by make_report)
    summary.md            human-readable per-cell summary

Examples
--------
Local Qwen, zero-shot (needs GPUs):
    python experiments/run_experiment.py --model qwen --method zeroshot

Local Qwen, agentic:
    python experiments/run_experiment.py --model qwen --method agentic

Re-summarize an already-evaluated cell (no GPU):
    python experiments/run_experiment.py --model qwen --method iterative --stage summarize
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_logged(cmd: list[str], log_path: Path, cwd: Path, env: dict, append=False) -> int:
    """Run a command, streaming combined stdout/stderr to console and a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    header = f"\n{'='*100}\n$ {' '.join(cmd)}\n  cwd={cwd}\n  at={now()}\n{'='*100}\n"
    print(header, flush=True)
    with log_path.open(mode, encoding="utf-8") as log:
        log.write(header)
        log.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        proc.wait()
        log.write(f"\n[exit code: {proc.returncode}] at {now()}\n")
    return proc.returncode


def gpu_count() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return len([l for l in out.splitlines() if l.strip()])
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def build_generate_cmd(model: C.ModelSpec, method: C.MethodSpec,
                        gen_results: Path, kernels_dir: Path,
                        prompts_dir: Path, py: str) -> list[str]:
    cmd = [py, str(method.main),
           "--input", str(method.data),
           "--output", str(gen_results),
           "--kernels-dir", str(kernels_dir)]

    # iterative writes per-turn prompt artifacts; keep them with the cell.
    if method.name == "iterative":
        cmd += ["--prompts-dir", str(prompts_dir)]

    cmd += ["--model", model.model_name, "--dtype", "float16"]
    cmd += method.local_args
    return cmd


def stage_generate(model, method, paths, env, py) -> int:
    # Kernels are written straight into the KernelBench run dir the eval reads.
    paths["kb_run"].mkdir(parents=True, exist_ok=True)
    cmd = build_generate_cmd(model, method, paths["gen_results"], paths["kb_run"],
                             paths["cell_dir"] / "prompts", py)
    rc = run_logged(cmd, paths["generate_log"], cwd=C.REPO_ROOT, env=env)
    if rc != 0:
        print(f"WARNING: generation exited with code {rc}", file=sys.stderr)
    return rc


def stage_eval(method, paths, env, py, num_gpu_devices, levels) -> int:
    rc_all = 0
    n_kernels = len(list(paths["kb_run"].glob("level_*_kernel.py")))
    if n_kernels == 0:
        print(f"ERROR: no kernels in {paths['kb_run']}; run --stage generate first.",
              file=sys.stderr)
        return 2
    print(f"Evaluating {n_kernels} kernels in {paths['kb_run']}")

    for level in levels:
        eval_results = paths["kb_run"] / "eval_results.json"
        eval_results.unlink(missing_ok=True)
        cmd = [py, str(C.KB_EVAL_SCRIPT),
               f"run_name={paths['run_dir']}",
               "dataset_src=local",
               f"level={level}",
               f"subset={C.SUBSET}",
               f"num_gpu_devices={num_gpu_devices}",
               f"gpu_arch={C.GPU_ARCH}"]
        rc = run_logged(cmd, paths["eval_log"], cwd=C.KB_DIR, env=env, append=True)
        rc_all = rc_all or rc
        if eval_results.exists():
            snap = paths["kb_run"] / f"eval_results_level{level}.json"
            shutil.copy(eval_results, snap)
            print(f"Saved {snap}")
        else:
            print(f"WARNING: no eval_results.json produced for level {level}",
                  file=sys.stderr)
    return rc_all


def stage_analyze(paths, env, py, levels) -> int:
    rc_all = 0
    for level in levels:
        eval_file = paths["kb_run"] / f"eval_results_level{level}.json"
        if not eval_file.exists():
            print(f"WARNING: missing {eval_file}; skipping analysis for level {level}",
                  file=sys.stderr)
            continue
        out_file = paths["kb_run"] / f"analysis_level{level}.json"
        cmd = [py, str(C.KB_ANALYSIS_SCRIPT),
               f"run_name={paths['run_dir']}",
               f"level={level}",
               f"hardware={C.HARDWARE}",
               f"baseline={C.BASELINE_NAME}",
               f"baseline_file={C.BASELINE_FILE}",
               f"eval_results_file={eval_file}",
               f"output_file={out_file}"]
        rc = run_logged(cmd, paths["eval_log"], cwd=C.KB_DIR, env=env, append=True)
        rc_all = rc_all or rc
        if out_file.exists():
            shutil.copy(out_file, paths["cell_dir"] / f"analysis_level{level}.json")
    return rc_all


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def _load(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _gen_lookup(gen_results: dict | None) -> dict[tuple[str, str], dict]:
    """(level, pid) -> generation entry, regardless of method's result schema."""
    out = {}
    if not gen_results:
        return out
    for lk, probs in gen_results.get("levels", {}).items():
        for pid, entry in probs.items():
            out[(str(lk), str(pid))] = entry
    return out


def build_cell_summary(model, method, kb_run: Path, run_dir: str,
                       levels, gen_results_path: Path | None = None) -> dict:
    """Pure summary builder: reads KernelBench analysis + eval_results for a cell.

    gen_results_path is optional (used for generation status, turn counts, and
    problem names); the report still works without it.
    """
    gen = _load(gen_results_path) if gen_results_path else None
    gen_by_id = _gen_lookup(gen)

    per_level = {}
    per_problem = []
    error_counts = {c: 0 for c in C.ERROR_CATEGORIES}

    for level in levels:
        analysis = _load(kb_run / f"analysis_level{level}.json")
        evalr = _load(kb_run / f"eval_results_level{level}.json") or {}
        if analysis:
            per_level[str(level)] = {
                "total_eval": analysis.get("total_eval"),
                "compiled_count": analysis.get("compiled_count"),
                "correct_count": analysis.get("correct_count"),
                "compilation_rate": analysis.get("compilation_rate"),
                "correctness_rate": analysis.get("correctness_rate"),
                "geo_mean_speedup": analysis.get("geo_mean_speedup"),
                "fast_p": analysis.get("fast_p", {}),
            }

        # per-problem detail + error taxonomy
        for pid in (str(i) for i in C.PROBLEM_IDS):
            ev_list = evalr.get(pid)
            ev = None
            if isinstance(ev_list, list) and ev_list:
                ev = next((e for e in ev_list if e.get("sample_id") == 0), ev_list[0])
            gen_entry = gen_by_id.get((str(level), pid), {})
            status = gen_entry.get("status")

            compiled = bool(ev.get("compiled")) if ev else False
            correct = bool(ev.get("correctness")) if ev else False
            runtime = (ev or {}).get("runtime")
            ref_runtime = (ev or {}).get("ref_runtime")
            speedup = None
            if runtime and ref_runtime and runtime > 0:
                speedup = ref_runtime / runtime

            if correct:
                category = "slow" if (speedup is not None and speedup < 1.0) else "correct"
            else:
                category = C.classify_error(status, ev)
            if category in error_counts:
                error_counts[category] += 1

            per_problem.append({
                "level": level,
                "problem_id": pid,
                "problem_name": gen_entry.get("problem_name"),
                "gen_status": status,
                "compiled": compiled,
                "correct": correct,
                "runtime_ms": runtime,
                "ref_runtime_ms": ref_runtime,
                "speedup": speedup,
                "category": category,
                "num_turns": gen_entry.get("num_turns"),
            })

    # cross-level aggregates
    n = len(per_problem)
    n_compiled = sum(1 for p in per_problem if p["compiled"])
    n_correct = sum(1 for p in per_problem if p["correct"])
    n_faster = sum(1 for p in per_problem if p["speedup"] and p["speedup"] >= 1.0)
    speedups_correct = [p["speedup"] for p in per_problem if p["correct"] and p["speedup"]]
    import math
    geo = (math.exp(sum(math.log(s) for s in speedups_correct) / len(speedups_correct))
           if speedups_correct else 0.0)

    summary = {
        "model_tag": model.tag,
        "model_display": model.display,
        "model_kind": model.kind,
        "method": method.name,
        "method_display": method.display,
        "run_dir": run_dir,
        "hardware": C.HARDWARE,
        "levels": levels,
        "n_tasks": n,
        "overall": {
            "compiled": n_compiled,
            "correct": n_correct,
            "faster_than_baseline": n_faster,
            "compilation_rate": n_compiled / n if n else 0.0,
            "correctness_rate": n_correct / n if n else 0.0,
            "fast_1_rate": n_faster / n if n else 0.0,
            "geo_mean_speedup_correct": geo,
        },
        "per_level": per_level,
        "error_taxonomy": error_counts,
        "per_problem": per_problem,
        "summarized_at": now(),
    }
    if gen:
        summary["gen_meta"] = gen.get("meta", {})
    return summary


def stage_summarize(model, method, paths, levels) -> dict:
    summary = build_cell_summary(
        model, method, paths["kb_run"], paths["run_dir"], levels,
        gen_results_path=paths["gen_results"],
    )
    paths["cell_dir"].mkdir(parents=True, exist_ok=True)
    (paths["cell_dir"] / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_md(summary, paths["cell_dir"] / "summary.md")
    print(f"\nWrote {paths['cell_dir']/'summary.json'}")
    print(f"Wrote {paths['cell_dir']/'summary.md'}")
    return summary


def _write_summary_md(s: dict, path: Path) -> None:
    o = s["overall"]
    lines = [
        f"# {s['model_display']} x {s['method_display']}",
        "",
        f"- Run dir: `{s['run_dir']}`  |  Hardware: {s['hardware']}  |  Tasks: {s['n_tasks']}",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "|---|---|",
        f"| Compilation rate | {o['compilation_rate']*100:.1f}% ({o['compiled']}/{s['n_tasks']}) |",
        f"| Correctness rate | {o['correctness_rate']*100:.1f}% ({o['correct']}/{s['n_tasks']}) |",
        f"| Faster than baseline (fast_1) | {o['fast_1_rate']*100:.1f}% ({o['faster_than_baseline']}/{s['n_tasks']}) |",
        f"| Geo-mean speedup (correct only) | {o['geo_mean_speedup_correct']:.3f}x |",
        "",
        "## Per level",
        "",
        "| level | correct | compile% | correct% | geo-mean speedup | fast_1.0 |",
        "|---|---|---|---|---|---|",
    ]
    for lvl, d in sorted(s["per_level"].items()):
        fast1 = (d.get("fast_p") or {}).get("1.0")
        lines.append(
            f"| {lvl} | {d.get('correct_count')}/{d.get('total_eval')} | "
            f"{(d.get('compilation_rate') or 0)*100:.0f}% | "
            f"{(d.get('correctness_rate') or 0)*100:.0f}% | "
            f"{(d.get('geo_mean_speedup') or 0):.3f}x | "
            f"{fast1 if fast1 is not None else '-'} |"
        )
    lines += ["", "## Error taxonomy", "", "| category | count |", "|---|---|"]
    for cat, cnt in s["error_taxonomy"].items():
        if cnt:
            lines.append(f"| {cat} | {cnt} |")
    lines += ["", "## Per-problem", "",
              "| L | P | name | compiled | correct | speedup | category |",
              "|---|---|---|---|---|---|---|"]
    for p in s["per_problem"]:
        sp = f"{p['speedup']:.2f}x" if p["speedup"] else "-"
        nm = (p["problem_name"] or "")[:34]
        lines.append(
            f"| {p['level']} | {p['problem_id']} | {nm} | "
            f"{'Y' if p['compiled'] else 'N'} | {'Y' if p['correct'] else 'N'} | "
            f"{sp} | {p['category']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--method", required=True, choices=C.all_method_names())
    ap.add_argument("--stage", default="all",
                    choices=["all", "generate", "eval", "analyze", "summarize"])
    ap.add_argument("--levels", default="1,2,3")
    ap.add_argument("--num-gpu-devices", type=int, default=None,
                    help="GPUs for KernelBench batch eval (default: detected, capped at 4)")
    ap.add_argument("--python", default=sys.executable, help="python interpreter to use")
    args = ap.parse_args()

    model = C.MODELS[args.model]
    method = C.method_spec(args.method)
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    run_dir = C.run_dir_name(model.tag, method.name)

    if args.num_gpu_devices is None:
        args.num_gpu_devices = min(gpu_count(), 4) or 1

    paths = {
        "run_dir": run_dir,
        "kb_run": C.KB_RUNS_DIR / run_dir,
        "cell_dir": C.RUNS_DIR / run_dir,
        "gen_results": C.RUNS_DIR / run_dir / "gen_results.json",
        "generate_log": C.RUNS_DIR / run_dir / "generate.log",
        "eval_log": C.RUNS_DIR / run_dir / "eval.log",
    }
    paths["cell_dir"].mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Make nvcc + a GCC>=9 host compiler reachable for cpp_extension builds when
    # the launching shell did not load them (e.g. interactive run after conda).
    C.ensure_build_toolchain(env)

    print(f"=== CELL: {model.display} x {method.display} ===")
    print(f"    run_dir={run_dir}  stage={args.stage}  levels={levels}  "
          f"eval_gpus={args.num_gpu_devices}")

    stages = (["generate", "eval", "analyze", "summarize"]
              if args.stage == "all" else [args.stage])

    rc = 0
    for st in stages:
        if st == "generate":
            rc = stage_generate(model, method, paths, env, args.python)
            # API-eval methods already produced gen_results with kernels; even if
            # rc!=0 we continue to try eval on whatever kernels exist.
        elif st == "eval":
            rc = stage_eval(method, paths, env, args.python, args.num_gpu_devices, levels)
        elif st == "analyze":
            rc = stage_analyze(paths, env, args.python, levels)
        elif st == "summarize":
            stage_summarize(model, method, paths, levels)

    print(f"\n=== DONE: {run_dir} (last rc={rc}) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
