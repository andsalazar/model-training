"""Print per-seed wall clock and speedup from a seed sweep report."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/speed10s_seeds.json")
    data = json.loads(path.read_text())

    by_cond: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for r in data.get("results", []):
        seconds = r.get("run", {}).get("seconds")
        if seconds is None:
            continue
        by_cond.setdefault((r["horizon"], r["steps"]), []).append(
            (r.get("seed", 0), float(seconds))
        )

    base_steps = data.get("default_steps")
    print("wall clock per generation (seconds)")
    for (horizon, steps), rows in sorted(by_cond.items()):
        rows.sort()
        times = [t for _, t in rows]
        base = dict(by_cond.get((horizon, base_steps), [])).get(rows[0][0])
        speedups = []
        base_map = dict(by_cond.get((horizon, base_steps), []))
        for seed, t in rows:
            if seed in base_map and base_map[seed]:
                speedups.append(base_map[seed] / t)
        print(
            f"  {horizon} {steps:>3} steps  seeds={[s for s, _ in rows]}  "
            f"times={[round(t, 1) for t in times]}  "
            f"mean={sum(times) / len(times):.1f}  "
            f"speedup_per_seed={[round(s, 2) for s in speedups]}"
        )


if __name__ == "__main__":
    main()
