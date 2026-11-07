#!/bin/bash
# Check whether the attention contrast compares the same noise level or different ones.
#
# The residual derivation aligned by timestep. The attention driver aligns by step index.
# If a 20-step run's step 18 and a 30-step run's step 18 sit at different timesteps, then
# "the effect grows with step" could be an artefact of comparing different noise levels
# rather than a measurement of drift. This prints the timestep table so the confound is
# visible instead of assumed away.
set -uo pipefail

source /local/ovi-env/bin/activate

python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("/local/captures/attention_20v30.json").read_text())
summaries = data.get("summaries", {})

print("=== timesteps per rollout, per layer ===")
for steps in sorted(summaries, key=int):
    per_layer = {}
    for key, s in summaries[steps].items():
        per_layer.setdefault(s["layer"], []).append((s["timestep"], s["row_entropy_mean"]))
    print(f"\n--- {steps} steps ---")
    for layer in sorted(per_layer):
        rows = sorted(per_layer[layer])
        steps_seen = [r[0] for r in rows]
        print(f"  layer {layer:>2}: timesteps {steps_seen}")

print()
print("=== do the two rollouts share timesteps? ===")
a, b = sorted(summaries, key=int)[:2]
ta = {s["timestep"] for s in summaries[a].values()}
tb = {s["timestep"] for s in summaries[b].values()}
print(f"{a} steps timesteps: {sorted(ta)}")
print(f"{b} steps timesteps: {sorted(tb)}")
print(f"shared: {sorted(ta & tb)}")
print(f"shared count: {len(ta & tb)} of {len(ta)} and {len(tb)}")

print()
print("=== row entropy by layer and step (the raw signal) ===")
for steps in sorted(summaries, key=int):
    print(f"\n--- {steps} steps ---")
    per_layer = {}
    for key, s in summaries[steps].items():
        per_layer.setdefault(s["layer"], []).append((s["timestep"], s["row_entropy_mean"]))
    for layer in sorted(per_layer):
        vals = [f"{e:.3f}" for _, e in sorted(per_layer[layer])]
        print(f"  layer {layer:>2}: {' '.join(vals)}")
PY
