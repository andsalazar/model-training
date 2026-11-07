#!/bin/bash
# Summarise the attention contrast: which heads separate drifting from stable, and how
# strongly.
#
# The key question is not "did anything change" but "does the change survive the
# timestep-alignment guard". Rows dropped for a gap beyond tolerance are reported
# separately, so a surviving row is a near-matched comparison.
set -uo pipefail
source /local/ovi-env/bin/activate

python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("/local/captures/attention_20v30.json").read_text())
contrast = data["contrast"]
rows = contrast["rows"]

kept = [r for r in rows if "max_effect" in r]
dropped = [r for r in rows if "skipped" in r]

print(f"=== alignment: {contrast.get('aligned_by')} ===")
print(f"rows: {len(rows)}  kept: {len(kept)}  dropped: {len(dropped)}")
print()

if dropped:
    print("--- dropped for timestep gap ---")
    for r in dropped:
        print(f"  step_a {r.get('step_a')} t={r.get('timestep_a')} -> "
              f"step_b {r.get('step_b')} t={r.get('timestep_b')} "
              f"gap {r.get('timestep_gap')}")
    print()

if not kept:
    print("no near-matched pairs survived; nothing to report")
    raise SystemExit(0)

print("--- kept pairs ---")
for r in kept:
    print(f"  layer {r['layer']:>2}  t_a {r['timestep_a']:>8.2f}  "
          f"t_b {r['timestep_b']:>8.2f}  gap {r['timestep_gap']:>6.2f}  "
          f"max {r['max_effect']:.4f}  heads {r['n_heads_changed']}")
print()

print("--- max_effect by layer (kept only) ---")
by_layer = {}
for r in kept:
    by_layer.setdefault(r["layer"], []).append(r["max_effect"])
for layer in sorted(by_layer):
    vals = by_layer[layer]
    print(f"  layer {layer:>2}: max {max(vals):.4f}  mean {sum(vals)/len(vals):.4f}  n={len(vals)}")
print()

print("--- strongest rows ---")
ranked = sorted(kept, key=lambda r: r["max_effect"], reverse=True)[:8]
for r in ranked:
    top = r["top_heads"][0] if r.get("top_heads") else {}
    print(f"  layer {r['layer']:>2} t={r['timestep_a']:>8.2f}: "
          f"max {r['max_effect']:.4f}  heads {r['n_heads_changed']}  "
          f"top head {top.get('head')} ({top.get('effect_size', 0):.4f})")
print()

print("--- head recurrence across kept rows ---")
counts = {}
for r in kept:
    for h in r.get("top_heads", [])[:3]:
        key = (r["layer"], h.get("head"))
        counts[key] = counts.get(key, 0) + 1
for (layer, head), n in sorted(counts.items(), key=lambda kv: -kv[1])[:10]:
    print(f"  layer {layer:>2} head {head:>2}: appears {n}x")
PY
