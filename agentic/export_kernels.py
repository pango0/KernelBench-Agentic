#!/usr/bin/env python3
"""Export best kernels from agentic/results.json to a flat directory for KernelBench eval."""

import argparse
import json
import sys
from pathlib import Path

AGENTIC_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=AGENTIC_DIR / "results.json",
    )
    parser.add_argument(
        "--kernels-dir",
        type=Path,
        default=AGENTIC_DIR / "agentic",
    )
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--problem", type=int, default=None)
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Missing {args.results}", file=sys.stderr)
        sys.exit(1)

    with args.results.open(encoding="utf-8") as f:
        results = json.load(f)

    n_kernels = 0
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
                kp = (
                    args.kernels_dir
                    / f"level_{level_key}_problem_{pid}_sample_{args.sample_id}_kernel.py"
                )
                kp.parent.mkdir(parents=True, exist_ok=True)
                kp.write_text(code, encoding="utf-8")
                n_kernels += 1
                print(f"kernel L{level_key} P{pid} -> {kp} ({status})")
            else:
                print(f"skip L{level_key} P{pid} (no code, status={status})")

    print(f"\nDone: {n_kernels} kernels -> {args.kernels_dir}")


if __name__ == "__main__":
    main()
