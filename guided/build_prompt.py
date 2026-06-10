#!/usr/bin/env python3
"""
Merge reference profiling results into data.json → data_guided.json.

Profiles are keyed by problem filename (e.g. 1_Square_matrix_multiplication_.py);
data.json is keyed by problem_id. We join on problem_name.

Usage (from repo root or guided/):
  python guided/build_prompt.py
  python guided/build_prompt.py --input ../data.json --profiles-dir guided/results/profiles -o ../data_guided.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

GUIDED_DIR = Path(__file__).resolve().parent
REPO_ROOT = GUIDED_DIR.parent

PROFILING_HEADER = """\
Here is GPU profiling information for the reference PyTorch implementation (measured on the target GPU):

{body}

Use this profiling data to guide your optimization: focus on the highest-cost operators, consider kernel fusion where multiple hot ops appear in sequence, and target memory-bound vs compute-bound behavior implied by the dominant kernels.
"""


def format_profile_entry(profile: Dict[str, Any], max_ops: int = 6) -> str:
    if profile.get("error"):
        return f"Profiling was not available for this problem: {profile['error']}"

    lines = []  # type: List[str]
    total = profile.get("total_cuda_time_ms")
    if total is not None:
        lines.append(f"Total CUDA time (reference forward): {total:.2f} ms")
        lines.append("")

    lines.append("Top operators by CUDA time (self time when available):")
    seen = set()  # type: Set[str]
    rank = 0
    for op in profile.get("top_ops", []):
        name = op.get("name", "")
        if not name or name in seen:
            continue
        self_ms = float(op.get("self_cuda_time_ms") or 0)
        cuda_ms = float(op.get("cuda_time_ms") or 0)
        time_ms = self_ms if self_ms > 0 else cuda_ms
        if time_ms <= 0:
            continue
        seen.add(name)
        rank += 1
        mem = op.get("cuda_memory_usage_mb", 0)
        mem_note = f", peak CUDA mem ~{mem:.1f} MB" if mem and mem > 0 else ""
        lines.append(f"  {rank}. {name}: {time_ms:.3f} ms{mem_note}")
        if rank >= max_ops:
            break

    if rank == 0:
        lines.append("  (no CUDA operators recorded)")

    return "\n".join(lines)


def load_profiles(profiles_dir, levels):
    # type: (Path, List[int]) -> Dict[Tuple[int, str], dict]
    """Map (level, problem_name) -> profile dict."""
    out = {}  # type: Dict[Tuple[int, str], dict]
    for level in levels:
        path = profiles_dir / f"reference_profiles_level{level}.json"
        if not path.exists():
            print(f"Warning: missing {path}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        level_key = f"level{level}"
        for problem_name, profile in blob.get(level_key, {}).items():
            out[(level, problem_name)] = profile
    return out


def inject_profiling(prompt: str, profile_body: str) -> str:
    block = PROFILING_HEADER.format(body=profile_body)
    marker = "\n\nOptimize the architecture named Model"
    if marker in prompt:
        return prompt.replace(marker, f"\n\n{block}{marker}", 1)
    return prompt.rstrip() + "\n\n" + block + "\n"


def build_guided_dataset(data, profiles_by_key, missing="warn"):
    # type: (dict, Dict[Tuple[int, str], dict], str) -> dict
    out = {
        "meta": {
            **data.get("meta", {}),
            "prompt_option": "profiling_guided",
            "profiling_source": "reference_torch_profiler",
        },
        "levels": {},
    }

    for level_key, problems in data.get("levels", {}).items():
        level = int(level_key)
        out["levels"][level_key] = {}
        for pid, entry in problems.items():
            new_entry = dict(entry)
            problem_name = entry.get("problem_name", "")
            profile = profiles_by_key.get((level, problem_name))

            if profile is None:
                msg = f"level {level} problem {pid} ({problem_name}): no profile"
                if missing == "error":
                    raise KeyError(msg)
                if missing == "warn":
                    print(f"Warning: {msg}", file=sys.stderr)
                profile_body = "Profiling data not found for this problem."
            else:
                profile_body = format_profile_entry(profile)

            new_entry["profiling"] = profile if profile is not None else None
            new_entry["profiling_summary"] = profile_body
            new_entry["prompt"] = inject_profiling(entry["prompt"], profile_body)
            out["levels"][level_key][pid] = new_entry

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data.json",
        help="Base prompts (zero-shot export)",
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=GUIDED_DIR / "results" / "profiles",
        help="Directory with reference_profiles_level{N}.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "data_guided.json",
    )
    parser.add_argument(
        "--levels",
        default="1,2,3",
        help="Comma-separated levels that have profile JSON files",
    )
    parser.add_argument(
        "--missing",
        choices=("warn", "error", "skip"),
        default="warn",
        help="If a problem has no profile file entry",
    )
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    profiles_by_key = load_profiles(args.profiles_dir, levels)

    if args.missing == "skip":
        # Filter profiles only; still build all problems from data.json
        pass

    guided = build_guided_dataset(data, profiles_by_key, missing=args.missing)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(guided, f, indent=2, ensure_ascii=False)

    n = sum(len(p) for p in guided["levels"].values())
    print(f"Wrote {args.output} ({n} problems)")
    print("Use with zeroshot: python main.py --input ../data_guided.json --kernels-dir runs/profiling_guided")


if __name__ == "__main__":
    main()
