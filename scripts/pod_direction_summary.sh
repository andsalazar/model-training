#!/bin/bash
# Summarise a direction report across layers, so the shape of the result is visible
# rather than a wall of per-step rows.
set -uo pipefail
source /local/ovi-env/bin/activate
REPORT="${1:-/local/captures/direction_20v30.json}"
python3 - "$REPORT" <<'PY'
import json
import sys

from collections import defaultdict

data = json.load(open(sys.argv[1]))
rows = data["rows"]
print(f"stream={data['stream']}  steps {data['steps_a']} -> {data['steps_b']}")
print(f"pairs={data['n_pairs']}  directions={data['n_directions']}  "
      f"identical={data['n_identical']}  skipped={data['n_skipped']}")
print(f"delta_norm min/median/max: "
      f"{data['delta_norm_min']:.4f} / {data['delta_norm_median']:.4f} / "
      f"{data['delta_norm_max']:.4f}")
print()

by_layer = defaultdict(list)
for row in rows:
    if "delta_norm" in row and row["delta_norm"] > 0:
        by_layer[row["layer"]].append(row)

print(f"{'layer':>5} {'n':>3} {'rel delta %':>12} {'cosine':>10} {'state norm':>11}")
for layer in sorted(by_layer):
    entries = by_layer[layer]
    rel = [e["delta_norm"] / ((e["state_norm_a"] + e["state_norm_b"]) / 2) * 100
           for e in entries]
    cos = [e["cosine"] for e in entries]
    norm = [e["state_norm_b"] for e in entries]
    print(f"{layer:>5} {len(entries):>3} {sum(rel)/len(rel):>11.2f}% "
          f"{sum(cos)/len(cos):>10.6f} {sum(norm)/len(norm):>11.3f}")

identical = [r for r in rows if r.get("delta_norm") == 0.0]
print(f"\nidentical states: {len(identical)} "
      f"(steps {[r['step_a'] for r in identical]} at layers "
      f"{sorted({r['layer'] for r in identical})})")
PY
