#!/usr/bin/env python3
"""
Aggregate every (model, method) cell into a report-ready bundle.

Walks the full 3x4 experiment matrix, (re)builds each cell summary from the
KernelBench analysis/eval JSON, and emits:

    report/data/master_metrics.csv     one row per (model, method, level)
    report/data/summary_by_cell.csv    one row per (model, method), all levels
    report/data/error_taxonomy.csv     error-category counts per cell
    report/figures/*.png               model x method heatmaps (if matplotlib)
    report/REPORT.md                   the written report scaffold with tables

Runs entirely from already-computed eval results: no GPU and no API calls.

Usage:
    python experiments/make_report.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from run_experiment import build_cell_summary, _write_summary_md  # noqa: E402

LEGACY_GEN_RESULTS = {
    ("qwen", "zeroshot"): C.REPO_ROOT / "zeroshot" / "results.json",
    ("qwen", "guided"): C.REPO_ROOT / "guided" / "results.json",
    ("qwen", "iterative"): C.REPO_ROOT / "iterative" / "results.json",
    ("qwen", "agentic"): C.REPO_ROOT / "agentic" / "results.json",
}


def find_gen_results(model_tag: str, method: str, run_dir: str) -> Path | None:
    cand = C.RUNS_DIR / run_dir / "gen_results.json"
    if cand.exists():
        return cand
    legacy = LEGACY_GEN_RESULTS.get((model_tag, method))
    if legacy and legacy.exists():
        return legacy
    return None


def ordered_items(summaries: dict):
    """Iterate cells in canonical (model, method) order, not alphabetically."""
    for mt in C.MODELS:
        for me in C.METHODS:
            if (mt, me) in summaries:
                yield (mt, me), summaries[(mt, me)]


def cell_has_results(run_dir: str, levels) -> bool:
    return any((C.KB_RUNS_DIR / run_dir / f"analysis_level{l}.json").exists()
               for l in levels)


def _all_cells_with_specs():
    """(model_tag, method_name, MethodSpec) for the core matrix + agentic ablations."""
    cells = [(mt, me, C.METHODS[me]) for (mt, me) in C.all_cells()]
    cells += [(mt, ab, spec) for mt in C.MODELS
              for ab, spec in C.AGENTIC_ABLATIONS.items()]
    return cells


def collect() -> dict[tuple[str, str], dict]:
    """Build (refresh) a summary for every cell (core + ablation) with eval results."""
    summaries = {}
    for model_tag, method, meth in _all_cells_with_specs():
        model = C.MODELS[model_tag]
        run_dir = C.run_dir_name(model_tag, method)
        if not cell_has_results(run_dir, C.LEVELS):
            continue
        gen = find_gen_results(model_tag, method, run_dir)
        s = build_cell_summary(model, meth, C.KB_RUNS_DIR / run_dir, run_dir,
                               C.LEVELS, gen_results_path=gen)
        # persist refreshed per-cell artifacts
        cell_dir = C.RUNS_DIR / run_dir
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        _write_summary_md(s, cell_dir / "summary.md")
        summaries[(model_tag, method)] = s
    return summaries


# ---------------------------------------------------------------------------
# CSV outputs
# ---------------------------------------------------------------------------

def write_csvs(summaries: dict, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    # master: per (model, method, level)
    with (data_dir / "master_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "model_tag", "method", "level", "total_eval",
                    "compiled", "correct", "compilation_rate", "correctness_rate",
                    "geo_mean_speedup", "fast_0.0", "fast_1.0", "fast_2.0"])
        for (mt, me), s in ordered_items(summaries):
            for lvl, d in sorted(s["per_level"].items()):
                fp = d.get("fast_p") or {}
                w.writerow([s["model_display"], mt, me, lvl, d.get("total_eval"),
                            d.get("compiled_count"), d.get("correct_count"),
                            _r(d.get("compilation_rate")), _r(d.get("correctness_rate")),
                            _r(d.get("geo_mean_speedup")),
                            fp.get("0.0"), fp.get("1.0"), fp.get("2.0")])

    # per cell (all levels)
    with (data_dir / "summary_by_cell.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "model_tag", "method", "n_tasks", "compiled",
                    "correct", "faster_than_baseline", "compilation_rate",
                    "correctness_rate", "fast_1_rate", "geo_mean_speedup_correct"])
        for (mt, me), s in ordered_items(summaries):
            o = s["overall"]
            w.writerow([s["model_display"], mt, me, s["n_tasks"], o["compiled"],
                        o["correct"], o["faster_than_baseline"],
                        _r(o["compilation_rate"]), _r(o["correctness_rate"]),
                        _r(o["fast_1_rate"]), _r(o["geo_mean_speedup_correct"])])

    # error taxonomy
    with (data_dir / "error_taxonomy.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cats = ["correct"] + C.ERROR_CATEGORIES
        w.writerow(["model", "method"] + cats)
        for (mt, me), s in ordered_items(summaries):
            tax = dict(s["error_taxonomy"])
            tax["correct"] = s["overall"]["correct"]
            w.writerow([mt, me] + [tax.get(c, 0) for c in cats])

    # agentic ablation study (full agentic + each ablation variant)
    rows = ablation_rows(summaries)
    if rows:
        with (data_dir / "ablation_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "variant", "display", "n_tasks", "compilation_rate",
                        "correctness_rate", "fast_1_rate", "geo_mean_speedup_correct"])
            for mt, me, disp, s in rows:
                o = s["overall"]
                w.writerow([mt, me, disp, s["n_tasks"], _r(o["compilation_rate"]),
                            _r(o["correctness_rate"]), _r(o["fast_1_rate"]),
                            _r(o["geo_mean_speedup_correct"])])


def ablation_rows(summaries: dict):
    """Full agentic followed by each ablation variant, per model, where results exist."""
    order = ["agentic"] + list(C.AGENTIC_ABLATIONS)
    rows = []
    for mt in C.MODELS:
        for me in order:
            s = summaries.get((mt, me))
            if s:
                rows.append((mt, me, C.method_spec(me).display, s))
    return rows


def _r(x, nd=4):
    return round(x, nd) if isinstance(x, (int, float)) else x


# ---------------------------------------------------------------------------
# Matrix helpers (model rows x method cols)
# ---------------------------------------------------------------------------

def matrix(summaries: dict, getter):
    models = list(C.MODELS)
    methods = list(C.METHODS)
    grid = []
    for mt in models:
        row = []
        for me in methods:
            s = summaries.get((mt, me))
            row.append(getter(s) if s else None)
        grid.append(row)
    return models, methods, grid


def md_matrix(summaries: dict, getter, fmt) -> str:
    models, methods, grid = matrix(summaries, getter)
    head = "| model \\ method | " + " | ".join(C.METHODS[m].display for m in methods) + " |"
    sep = "|" + "---|" * (len(methods) + 1)
    lines = [head, sep]
    for mt, row in zip(models, grid):
        cells = [fmt(v) if v is not None else "—" for v in row]
        lines.append(f"| {C.MODELS[mt].display} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_heatmaps(summaries: dict, fig_dir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"(matplotlib unavailable: {e}; skipping figures)")
        return []

    fig_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("correctness_rate", "Correctness rate", lambda s: s["overall"]["correctness_rate"],
         "{:.0%}", "YlGn"),
        ("geo_mean_speedup", "Geo-mean speedup (correct)",
         lambda s: s["overall"]["geo_mean_speedup_correct"], "{:.2f}x", "RdYlGn"),
        ("fast_1_rate", "Fraction faster than baseline (fast_1)",
         lambda s: s["overall"]["fast_1_rate"], "{:.0%}", "Blues"),
    ]
    written = []
    methods = list(C.METHODS)
    models = list(C.MODELS)
    for key, title, getter, cellfmt, cmap in specs:
        _, _, grid = matrix(summaries, getter)
        arr = np.array([[(v if v is not None else np.nan) for v in row] for row in grid],
                       dtype=float)
        fig, ax = plt.subplots(figsize=(1.6 * len(methods) + 2, 1.1 * len(models) + 1.5))
        im = ax.imshow(arr, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(methods)), [C.METHODS[m].display for m in methods])
        ax.set_yticks(range(len(models)), [C.MODELS[m].display for m in models])
        for i in range(len(models)):
            for j in range(len(methods)):
                v = arr[i, j]
                txt = "—" if np.isnan(v) else cellfmt.format(v)
                ax.text(j, i, txt, ha="center", va="center", fontsize=10)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = fig_dir / f"heatmap_{key}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        written.append(out.name)
        print(f"Wrote {out}")
    return written


# ---------------------------------------------------------------------------
# REPORT.md
# ---------------------------------------------------------------------------

def write_report(summaries: dict, figures: list[str], report_dir: Path) -> None:
    done = set(summaries)
    all_cells = set(C.all_cells())
    missing = sorted(all_cells - done)

    L = []
    A = L.append
    A("# LLM-Based Kernel Improvement on KernelBench — Results Report\n")
    A("Auto-generated by `experiments/make_report.py`. Tables and figures refresh "
      "from the latest eval results; the prose sections are scaffolds to expand.\n")

    A("## 1. Experimental setup\n")
    A(f"- **Benchmark:** KernelBench, levels {C.LEVELS}, first {C.PROBLEMS_PER_LEVEL} "
      f"problems each (**{len(C.LEVELS)*C.PROBLEMS_PER_LEVEL} tasks**).")
    A(f"- **Hardware:** {C.HARDWARE} (Volta); baseline = eager PyTorch "
      f"(`{C.BASELINE_NAME}`).")
    A("- **Models:** " + "; ".join(f"{m.display} (`{m.tag}`, {m.kind})"
                                    for m in C.MODELS.values()) + ".")
    A("- **Methods:** " + "; ".join(m.display for m in C.METHODS.values()) + ".")
    A(f"- **Design:** {len(C.MODELS)}×{len(C.METHODS)} = {len(all_cells)} cells; "
      f"**{len(done)} complete**, {len(missing)} pending.")
    if missing:
        A("- **Pending cells:** " +
          ", ".join(f"{C.MODELS[mt].display}/{C.METHODS[me].display}"
                    for mt, me in missing) + ".")
    A("")
    A("**Metrics.** *Compilation rate* = fraction that build; *correctness rate* = "
      "fraction matching the reference within tolerance; *fast_1* = fraction that are "
      "both correct and faster than eager PyTorch; *geo-mean speedup* = geometric mean "
      "of (baseline/kernel) runtime over correct samples (>1 is faster).\n")

    A("## 2. Headline comparison (all levels pooled)\n")
    A("### 2.1 Correctness rate\n")
    A(md_matrix(summaries, lambda s: s["overall"]["correctness_rate"],
                lambda v: f"{v:.0%}") + "\n")
    A("### 2.2 Geometric-mean speedup (correct samples)\n")
    A(md_matrix(summaries, lambda s: s["overall"]["geo_mean_speedup_correct"],
                lambda v: f"{v:.2f}x") + "\n")
    A("### 2.3 Faster-than-baseline rate (fast_1)\n")
    A(md_matrix(summaries, lambda s: s["overall"]["fast_1_rate"],
                lambda v: f"{v:.0%}") + "\n")
    A("### 2.4 Compilation rate\n")
    A(md_matrix(summaries, lambda s: s["overall"]["compilation_rate"],
                lambda v: f"{v:.0%}") + "\n")

    if figures:
        A("## 3. Heatmaps\n")
        for fn in figures:
            A(f"![{fn}](figures/{fn})\n")

    A("## 4. Per-level breakdown\n")
    A(f"Correctness count (out of {C.PROBLEMS_PER_LEVEL}) per level, per cell.\n")
    A("| model | method | L1 | L2 | L3 |")
    A("|---|---|---|---|---|")
    for (mt, me), s in ordered_items(summaries):
        pl = s["per_level"]
        c = [str(pl.get(str(l), {}).get("correct_count", "—")) for l in C.LEVELS]
        A(f"| {C.MODELS[mt].display} | {C.METHODS[me].display} | " + " | ".join(c) + " |")
    A("")

    A("## 5. Error taxonomy\n")
    A("Counts per failure category (correct/slow cells excluded from failure buckets).\n")
    cats = [c for c in C.ERROR_CATEGORIES if c != "correctness" or True]
    A("| model | method | " + " | ".join(cats) + " |")
    A("|---|---|" + "---|" * len(cats))
    for (mt, me), s in ordered_items(summaries):
        tax = s["error_taxonomy"]
        A(f"| {C.MODELS[mt].display} | {C.METHODS[me].display} | " +
          " | ".join(str(tax.get(c, 0)) for c in cats) + " |")
    A("")

    rows = ablation_rows(summaries)
    if rows:
        A("## 6. Agentic ablation study\n")
        A("The full agentic system is the multi-agent loop (Code Analyzer + RAG "
          "Researcher + Kernel Generator + Evaluator + Feedback Analyzer). Each "
          "variant below flips exactly one component; comparing it to the full system "
          "isolates that component's contribution. See `docs/AGENTIC_METHOD.md`.\n")
        A("| variant | correct% | fast_1% | geo-mean speedup | compile% | tasks |")
        A("|---|---|---|---|---|---|")
        for _mt, _me, disp, s in rows:
            o = s["overall"]
            A(f"| {disp} | {o['correctness_rate']*100:.0f}% | {o['fast_1_rate']*100:.0f}% | "
              f"{o['geo_mean_speedup_correct']:.2f}x | {o['compilation_rate']*100:.0f}% | "
              f"{s['n_tasks']} |")
        A("")

    A("## 7. Discussion (to write)\n")
    A("- How much does richer guidance (guided / iterative / agentic) lift the local "
      "Qwen model over plain zero-shot?")
    A("- Iterative vs agentic: does explicit planning + self-critique beat plain "
      "feedback loops?")
    A("- Ablations: which agent (RAG, Code Analyzer, Feedback Analyzer) contributes "
      "most? Does best-of-n trade compute for quality favourably?")
    A("- How does each method degrade from level 1 → 3 as problems get harder?")
    A("- Reasoning overhead vs payoff: turns spent vs speedup gained.\n")

    A("## 8. Artifacts\n")
    A("- Per-cell detail: `experiments/runs/<run_dir>/summary.md` (per-problem "
      "table, turns, categories).")
    A("- Machine-readable: `report/data/*.csv`.")
    A("- Raw generations + logs: `experiments/runs/<run_dir>/`.\n")

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {report_dir/'REPORT.md'}")


def main() -> int:
    summaries = collect()
    if not summaries:
        print("No completed cells found. Run experiments first.", file=sys.stderr)
        return 1
    print(f"Collected {len(summaries)} cells: "
          + ", ".join(f"{mt}/{me}" for mt, me in summaries))
    write_csvs(summaries, C.REPORT_DIR / "data")
    figures = make_heatmaps(summaries, C.REPORT_DIR / "figures")
    write_report(summaries, figures, C.REPORT_DIR)
    print("\nReport ready at report/REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
