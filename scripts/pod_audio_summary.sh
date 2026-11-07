#!/bin/bash
# Summarise the audio direction report, layer by layer.
#
# Run as a script file: nested quoting through PowerShell -> ssh -> bash mangles
# anything containing quotes or brackets.
set -uo pipefail

source /local/ovi-env/bin/activate

python3 - <<'PY'
import json

d = json.load(open("/local/captures/direction_audio.json"))
print("stream:", d["stream"])
print("rows:", len(d["rows"]))
print()
print(f"{'layer':>5} {'stepA':>5} {'stepB':>5} {'timestep':>9} {'delta':>9} {'cosine':>10} {'normA':>9} {'rel%':>7}")
for r in d["rows"]:
    rel = 100.0 * r["delta_norm"] / max(r["state_norm_a"], 1e-9)
    print(
        f"{r['layer']:>5} {r['step_a']:>5} {r['step_b']:>5} {r['timestep_a']:>9.1f} "
        f"{r['delta_norm']:>9.3f} {r['cosine']:>10.6f} {r['state_norm_a']:>9.2f} {rel:>7.2f}"
    )

print()
print("=== per-layer summary ===")
by_layer = {}
for r in d["rows"]:
    by_layer.setdefault(r["layer"], []).append(r)
for layer in sorted(by_layer):
    rows = by_layer[layer]
    rels = [100.0 * r["delta_norm"] / max(r["state_norm_a"], 1e-9) for r in rows]
    cosines = [r["cosine"] for r in rows]
    identical = [r for r in rows if r["delta_norm"] == 0.0]
    print(
        f"L{layer:2d}: {len(rows)} pairs, rel delta {sum(rels)/len(rels):.2f}% "
        f"(max {max(rels):.2f}%), cosine {min(cosines):.6f}-{max(cosines):.6f}, "
        f"identical {len(identical)}"
    )
PY
