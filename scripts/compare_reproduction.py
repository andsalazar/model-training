"""Compare our reproduction against the paper's released scores."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    ours_base = load("runs/vllm_base.json")
    ours_edit = load("runs/vllm_edited.json")
    paper_base = load("runs/paper/scores_base_math500.json")
    paper_edit = load("runs/paper/scores_edited_math500.json")

    rows = [
        ("paper  base", paper_base["summary"]),
        ("ours   base", ours_base["summary"]),
        ("paper  edited", paper_edit["summary"]),
        ("ours   edited", ours_edit["summary"]),
    ]

    print(f"{'run':16s} {'acc':>7s} {'cut':>7s} {'mean_tok':>10s} {'tok/s':>9s}")
    print("-" * 54)
    for label, s in rows:
        print(
            f"{label:16s} {s['acc']:7.3f} {s['cut_share']:7.3f} "
            f"{s['mean_tokens']:10.0f} {s.get('tok_per_s', 0):9.0f}"
        )

    print("\n=== deltas (edited - base) ===")
    for name, b, e in (
        ("paper", paper_base["summary"], paper_edit["summary"]),
        ("ours", ours_base["summary"], ours_edit["summary"]),
    ):
        print(
            f"{name:6s} acc {b['acc']:.3f} -> {e['acc']:.3f} "
            f"({e['acc'] - b['acc']:+.3f})   "
            f"cut {b['cut_share']:.3f} -> {e['cut_share']:.3f} "
            f"({e['cut_share'] - b['cut_share']:+.3f})"
        )

    # Per-problem agreement on the practice subset.
    paper_by_id = {p["id"]: p["correct"] for p in paper_edit["per_problem"]}
    ours_by_id = {p["id"]: p["correct"] for p in ours_edit["per_problem"]}

    agree = 0
    total = 0
    print("\n=== per-problem, edited model (paper vs ours) ===")
    for pid, ours in ours_by_id.items():
        if pid not in paper_by_id:
            continue
        p = paper_by_id[pid]
        pa = sum(p) / len(p)
        oa = sum(ours) / len(ours)
        total += 1
        if abs(pa - oa) < 0.01:
            agree += 1
        print(f"  {pid:40s} paper={pa:.2f} ours={oa:.2f}")

    print(f"\nexact agreement: {agree}/{total}")

    # Correlation between the two runs.
    pairs = [
        (sum(paper_by_id[p]) / len(paper_by_id[p]), sum(o) / len(o))
        for p, o in ours_by_id.items()
        if p in paper_by_id
    ]
    if len(pairs) > 2:
        xs = [a for a, _ in pairs]
        ys = [b for _, b in pairs]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in pairs)
        den = (
            sum((x - mx) ** 2 for x in xs) ** 0.5
            * sum((y - my) ** 2 for y in ys) ** 0.5
        )
        print(f"per-problem correlation: {num / den:.3f}" if den else "correlation: n/a")

    return 0


if __name__ == "__main__":
    sys.exit(main())
