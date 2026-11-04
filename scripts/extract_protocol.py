"""Extract the paper's exact eval protocol from its released score files.

The base file covers only the 25-problem practice subset; the edited file covers
all 500. Both name problems by MATH-500 id, which is what lets a rerun select
the identical problems instead of guessing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    base = json.loads(Path("runs/paper/scores_base_math500.json").read_text(encoding="utf-8"))
    edited = json.loads(
        Path("runs/paper/scores_edited_math500.json").read_text(encoding="utf-8")
    )

    print("=== base summary (practice subset) ===")
    for k, v in base["summary"].items():
        if k != "sampling":
            print(f"  {k}: {v}")
    print("  sampling:", json.dumps(base["summary"]["sampling"]))

    print("\n=== edited summary (full 500) ===")
    for k, v in edited["summary"].items():
        if k != "sampling":
            print(f"  {k}: {v}")

    practice_ids = [p["id"] for p in base["per_problem"]]
    print(f"\npractice problems: {len(practice_ids)}")
    for pid in practice_ids:
        print(f"  {pid}")

    # Per-problem accuracy on the practice subset, from the edited run.
    by_id = {p["id"]: p["correct"] for p in edited["per_problem"]}
    print("\n=== practice subset: base vs edited (per-problem, 4 samples) ===")
    base_acc = []
    edit_acc = []
    for p in base["per_problem"]:
        pid = p["id"]
        b = sum(p["correct"]) / len(p["correct"])
        e = sum(by_id[pid]) / len(by_id[pid]) if pid in by_id else float("nan")
        base_acc.append(b)
        edit_acc.append(e)
        print(f"  {pid:42s} base={b:.2f} edited={e:.2f}")

    print(f"\npractice mean: base={sum(base_acc) / len(base_acc):.3f} "
          f"edited={sum(edit_acc) / len(edit_acc):.3f}")

    Path("runs/paper/practice_ids.json").write_text(
        json.dumps(practice_ids, indent=2), encoding="utf-8"
    )
    print("\nwritten: runs/paper/practice_ids.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
