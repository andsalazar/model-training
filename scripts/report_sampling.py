"""Summarize the sampling comparison and show failure modes."""

from __future__ import annotations

import json
import sys


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/sampling_test.json"
    with open(path, encoding="utf-8") as f:
        results = json.load(f)

    print(f"{'condition':14s} {'acc':>7s} {'correct':>9s} {'capped':>7s} {'secs':>7s}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['label']:14s} {r['accuracy']:7.3f} "
            f"{r['correct']:4d}/{r['total']:<4d} {r['hit_token_cap']:7d} "
            f"{r['seconds']:7.0f}"
        )

    print("\n=== failure modes (greedy) ===")
    greedy = next(r for r in results if r["label"] == "greedy")
    for s in greedy["samples"]:
        if not s["correct"]:
            reason = "hit token cap" if s["hit_token_cap"] else "wrong answer"
            print(
                f"  #{s['index']:2d} {reason:14s} "
                f"tokens={s['tokens']:5d} pred={str(s['prediction'])[:30]!r}"
            )

    print("\n=== failure modes (recommended) ===")
    rec = next(r for r in results if r["label"] == "recommended")
    for s in rec["samples"]:
        if not s["correct"]:
            reason = "hit token cap" if s["hit_token_cap"] else "wrong answer"
            print(
                f"  #{s['index']:2d} {reason:14s} "
                f"tokens={s['tokens']:5d} pred={str(s['prediction'])[:30]!r}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
