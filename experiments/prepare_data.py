#!/usr/bin/env python3
"""
Make sure every data artifact matches experiments/config.py before a matrix run.

Idempotent: each step checks whether the artifact already covers
config.PROBLEMS_PER_LEVEL for config.LEVELS and only (re)builds it if it does not.
At the default size (10) with everything already present this does nothing.

Artifacts handled:
  1. data.json            zero-shot prompts          (CPU)
  2. guided profiles      reference GPU profiles     (GPU)
  3. data_guided.json     profiling-enriched prompts (CPU; rebuilt if 1 or 2 changed)
  4. V100 speed baseline  baseline_time_torch.json   (GPU; needed for speedup metrics)

If the problem count changed, a flag file (experiments/runs/.force_rerun) is written
so run_all.sh knows to re-run every cell (old kernels only cover the old count).

Usage:
    python experiments/prepare_data.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

N = C.PROBLEMS_PER_LEVEL
LEVELS_CSV = ",".join(str(l) for l in C.LEVELS)
FORCE_FLAG = C.RUNS_DIR / ".force_rerun"
GUIDED_DIR = C.REPO_ROOT / "guided"
PROFILES_DIR = GUIDED_DIR / "results" / "profiles"
EXPORT_SCRIPT = C.KB_DIR / "scripts" / "export_prompts_json.py"
BASELINE_SCRIPT = C.KB_DIR / "scripts" / "generate_baseline_time_v100.py"
PY = sys.executable


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def gpu_count() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return len([x for x in out.splitlines() if x.strip()])
    except Exception:
        return 0


def run(cmd, cwd, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n  (cwd={cwd})", flush=True)
    rc = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env).returncode
    if rc != 0:
        raise RuntimeError(f"command failed (rc={rc}): {' '.join(str(c) for c in cmd)}")


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def prompts_ok(path: Path) -> bool:
    d = _load(path)
    if not d:
        return False
    levels = d.get("levels", {})
    return all(len(levels.get(str(l), {})) >= N for l in C.LEVELS)


def profiles_ok() -> bool:
    for l in C.LEVELS:
        d = _load(PROFILES_DIR / f"reference_profiles_level{l}.json")
        if not d or len(d.get(f"level{l}", {})) < N:
            return False
    return True


def baseline_ok() -> bool:
    d = _load(C.BASELINE_FILE)
    if not d:
        return False
    return all(len(d.get(f"level{l}", {})) >= N for l in C.LEVELS)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def require_gpu(what: str):
    if gpu_count() == 0:
        sys.exit(f"ERROR: '{what}' needs a GPU but none is visible. "
                 f"Run inside a SLURM allocation (e.g. via experiments/run_all.sh).")


def export_prompts():
    run([PY, EXPORT_SCRIPT, "--levels", LEVELS_CSV,
         "--start-id", "1", "--end-id", str(N), "-o", str(C.DATA_ZEROSHOT)],
        cwd=C.REPO_ROOT)


def regen_profiles():
    require_gpu("guided profiling")
    ngpu = gpu_count()
    procs = []
    for i, level in enumerate(C.LEVELS):
        env = dict(os.environ)
        env["NUM_PROBLEMS_PER_LEVEL"] = str(N)
        env["CUDA_VISIBLE_DEVICES"] = str(i % ngpu)
        print(f"\n$ profile_reference.py {level} 0  (gpu={i % ngpu}, N={N})", flush=True)
        procs.append(subprocess.Popen(
            [PY, "profile_reference.py", str(level), "0"],
            cwd=str(GUIDED_DIR), env=env))
    rc = max((p.wait() for p in procs), default=0)
    if rc != 0:
        raise RuntimeError("profiling failed")


def build_guided():
    run([PY, "build_prompt.py"], cwd=GUIDED_DIR)


def regen_baseline():
    require_gpu("V100 speed baseline")
    run([PY, "scripts/generate_baseline_time_v100.py"],
        cwd=C.KB_DIR, env_extra={"NUM_PROBLEMS_PER_LEVEL": str(N)})


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    C.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Preparing data for {N} problems/level on levels {C.LEVELS} "
          f"({N * len(C.LEVELS)} tasks).")

    data_changed = False

    if prompts_ok(C.DATA_ZEROSHOT):
        print(f"[ok] {C.DATA_ZEROSHOT.name} already covers {N}/level")
    else:
        print(f"[build] re-exporting {C.DATA_ZEROSHOT.name} for {N}/level")
        export_prompts()
        data_changed = True

    profiles_changed = False
    if profiles_ok():
        print(f"[ok] guided profiles already cover {N}/level")
    else:
        print(f"[build] profiling references for {N}/level (GPU)")
        regen_profiles()
        profiles_changed = True

    if data_changed or profiles_changed or not prompts_ok(C.DATA_GUIDED):
        print(f"[build] rebuilding {C.DATA_GUIDED.name}")
        build_guided()
    else:
        print(f"[ok] {C.DATA_GUIDED.name} already covers {N}/level")

    if baseline_ok():
        print(f"[ok] V100 baseline already covers {N}/level")
    else:
        print(f"[build] regenerating V100 speed baseline for {N}/level (GPU)")
        regen_baseline()

    if data_changed:
        FORCE_FLAG.write_text("size changed; rerun all cells\n", encoding="utf-8")
        print(f"[note] problem count changed -> wrote {FORCE_FLAG} "
              f"(run_all.sh will re-run every cell)")

    print("\nData prepared. ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
