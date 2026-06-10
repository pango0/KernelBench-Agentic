#!/usr/bin/env python3
"""
Export best kernels and prompts from iterative/results.json.

Usage:
  python export_artifacts.py
  python export_artifacts.py --results results.json --print
  python export_artifacts.py --level 3 --problem 4 --print
"""

import argparse
import json
import sys
from pathlib import Path

ITERATIVE_DIR = Path(__file__).resolve().parent


def load_results(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print("ERROR: cannot parse {}: {}".format(path, e), file=sys.stderr)
        print(
            "Re-run iterative/main.py (JSON save is fixed) or restore a complete results.json.",
            file=sys.stderr,
        )
        sys.exit(1)


def kernel_path(kernels_dir, level_key, pid, sample_id=0):
    return (
        Path(kernels_dir)
        / "level_{}_problem_{}_sample_{}_kernel.py".format(
            level_key, pid, sample_id
        )
    )


def write_kernel(kernels_dir, level_key, pid, code, sample_id=0):
    kp = kernel_path(kernels_dir, level_key, pid, sample_id)
    kp.parent.mkdir(parents=True, exist_ok=True)
    kp.write_text(code, encoding="utf-8")
    return kp


def export_prompts(entry, prompts_dir, level_key, pid):
    from main import export_prompts_for_problem

    return export_prompts_for_problem(entry, prompts_dir, level_key, pid)


def print_prompts(level_key, pid, entry):
    from main import print_prompts_for_problem

    print_prompts_for_problem(level_key, pid, entry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=ITERATIVE_DIR / "results.json",
    )
    parser.add_argument(
        "--kernels-dir",
        type=Path,
        default=ITERATIVE_DIR / "iterative",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=ITERATIVE_DIR / "prompts",
    )
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--problem", type=int, default=None)
    parser.add_argument(
        "--print",
        dest="do_print",
        action="store_true",
        help="Print prompts to stdout",
    )
    parser.add_argument(
        "--kernels-only",
        action="store_true",
        help="Only export kernel files",
    )
    args = parser.parse_args()

    if not args.results.exists():
        print("Missing {}".format(args.results), file=sys.stderr)
        sys.exit(1)

    results = load_results(args.results)
    n_kernels = 0
    n_prompts = 0

    for level_key, problems in sorted(
        results.get("levels", {}).items(), key=lambda x: int(x[0])
    ):
        if args.level is not None and int(level_key) != args.level:
            continue
        for pid, entry in sorted(problems.items(), key=lambda x: int(x[0])):
            if args.problem is not None and int(pid) != args.problem:
                continue

            code = entry.get("extracted_code")
            status = entry.get("status", "?")
            if code:
                kp = write_kernel(
                    args.kernels_dir,
                    level_key,
                    pid,
                    code,
                    args.sample_id,
                )
                n_kernels += 1
                print(
                    "kernel L{} P{} -> {} ({})".format(
                        level_key, pid, kp, status
                    )
                )
            else:
                print(
                    "skip kernel L{} P{} (no extracted_code, status={})".format(
                        level_key, pid, status
                    )
                )

            if not args.kernels_only:
                paths = export_prompts(
                    entry, args.prompts_dir, level_key, pid
                )
                n_prompts += len(paths)
                print(
                    "prompts L{} P{} -> {} files in {}".format(
                        level_key,
                        pid,
                        len(paths),
                        entry.get("prompt_export_dir", args.prompts_dir),
                    )
                )

            if args.do_print:
                print_prompts(level_key, pid, entry)

    print(
        "\nDone: {} kernels -> {}, prompts under {}".format(
            n_kernels, args.kernels_dir, args.prompts_dir
        )
    )


if __name__ == "__main__":
    main()
