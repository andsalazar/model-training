"""Summarize the blind-vs-weights ablation across editors."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ARMS = ("astra", "kimi", "deepseek", "glm", "qwen")


def main() -> int:
    print(f"{'editor':10s} {'weights':>9s} {'blind':>7s}  verdict")
    print("-" * 78)

    rows = []
    for arm in ARMS:
        path = Path(f"runs/ablation-{arm}.json")
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        rows.append(d)
        print(
            f"{arm:10s} {str(d['weights_succeeded']):>9s} "
            f"{str(d['blind_succeeded']):>7s}  {d['verdict']}"
        )

    print()
    declined = sum(1 for d in rows if not d["weights_succeeded"])
    print(f"declined on the weights arm: {declined}/{len(rows)}")

    # Every editor that declined named the same reason.
    print("\n=== stated reason (weights arm, final lines) ===")
    for d in rows:
        tail = d["weights_response"].strip().splitlines()
        last = " ".join(tail[-3:])[:220] if tail else ""
        print(f"\n{d['editor']}:\n  {last}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
