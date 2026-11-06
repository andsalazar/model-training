"""Print the per-condition summary from a seed sweep report."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/speed10s_seeds.json")
    data = json.loads(path.read_text())

    print(f"prompt: {data.get('prompt', '')[:70]}")
    print(f"default steps: {data.get('default_steps')}\n")

    print("per-condition summary across seeds")
    for s in data.get("summary", []):
        print(
            f"  {s['horizon']:<14} {s['steps']:>3} steps  n={s['n_seeds']}  "
            f"speedup={s['speedup_median']}  all_ok={s['acceptable_all_seeds']}  "
            f"any_ok={s['acceptable_any_seed']}  outside={s['outside_counts']}"
        )

    # The colour question specifically: the distribution per condition.
    print("\ncolour_stability per generation")
    by_cond: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for r in data.get("results", []):
        if "quality" not in r:
            continue
        key = (r["horizon"], r["steps"])
        by_cond.setdefault(key, []).append(
            (r.get("seed", 0), r["quality"]["colour_stability"])
        )
    for (horizon, steps), rows in sorted(by_cond.items()):
        values = sorted(v for _, v in rows)
        seeds = [s for s, _ in rows]
        spread = values[-1] - values[0] if len(values) > 1 else 0.0
        print(
            f"  {horizon} {steps:>3} steps  seeds={seeds}  values={[round(v, 2) for v in values]}  "
            f"range={spread:.2f}  mean={sum(values) / len(values):.2f}"
        )

    # Do the distributions overlap? That is the question the seed run exists to answer.
    print("\noverlap check (does the low-step colour range intersect the baseline's?)")
    for horizon in sorted({h for h, _ in by_cond}):
        base_steps = data.get("default_steps")
        base = sorted(v for _, v in by_cond.get((horizon, base_steps), []))
        if not base:
            continue
        print(f"  baseline {base_steps} steps: {[round(v, 2) for v in base]}")
        for (h, steps), rows in sorted(by_cond.items()):
            if h != horizon or steps == base_steps:
                continue
            vals = sorted(v for _, v in rows)
            overlaps = not (vals[-1] < base[0] or vals[0] > base[-1])
            print(
                f"    {steps:>3} steps: {[round(v, 2) for v in vals]}  "
                f"{'OVERLAPS baseline' if overlaps else 'disjoint from baseline'}"
            )


if __name__ == "__main__":
    main()
