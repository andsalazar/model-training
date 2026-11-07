#!/bin/bash
# Inspect the prior attention run's contrast rows.
#
# The prior run predates the timestep mapping: its summaries carry no "timesteps" key,
# so its 50 rows were aligned by step index. That is the confounded comparison -- at
# index 18 the 20-step run is at t=233 and the 30-step run at t=756. This shows what
# the rows actually contain so the old result can be labelled correctly.
set -uo pipefail
source /local/ovi-env/bin/activate

python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("/local/captures/attention_20v30.json").read_text())
rows = data["contrast"]["rows"]
print(f"rows: {len(rows)}")
print("row keys:", list(rows[0].keys()))
print()

print("--- max_effect by layer ---")
by_layer = {}
for r in rows:
    if "max_effect" in r:
        by_layer.setdefault(r["layer"], []).append(r["max_effect"])
for layer in sorted(by_layer):
    vals = by_layer[layer]
    print(f"  layer {layer:>2}: max {max(vals):.4f}  mean {sum(vals)/len(vals):.4f}  n={len(vals)}")
print()

print("--- max_effect by step ---")
by_step = {}
for r in rows:
    if "max_effect" in r:
        by_step.setdefault(r.get("step"), []).append(r["max_effect"])
for step in sorted(by_step):
    vals = by_step[step]
    print(f"  step {step:>2}: max {max(vals):.4f}  mean {sum(vals)/len(vals):.4f}")
print()

print("--- does any row carry a timestep? ---")
print("timestep_a present:", sum(1 for r in rows if "timestep_a" in r))
print("timestep_gap present:", sum(1 for r in rows if "timestep_gap" in r))
print()

print("--- strongest rows ---")
ranked = sorted(
    (r for r in rows if r.get("max_effect")),
    key=lambda r: r["max_effect"],
    reverse=True,
)[:6]
for r in ranked:
    top = r["top_heads"][0] if r.get("top_heads") else {}
    print(
        f"  layer {r['layer']:>2} step {r['step']:>2}: "
        f"max {r['max_effect']:.4f}  heads_changed {r['n_heads_changed']}  "
        f"top head {top.get('head')} ({top.get('effect_size', 0):.4f})"
    )
PY
