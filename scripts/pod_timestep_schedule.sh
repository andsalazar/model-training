#!/bin/bash
# What timesteps does a 20-step run actually visit, versus a 30-step run?
#
# The attention contrast aligns by step index. If the two runs visit different noise
# levels at the same index, then "the effect grows with step" could be an artefact of
# comparing different points on the trajectory rather than a measurement of drift. This
# prints the real schedule so the confound is visible instead of assumed away.
#
# The solver must be the one generate() uses: 'unipc'. An earlier version of this script
# used 'euler', which produced the wrong table (t=233 vs t=756 at index 18) that was then
# quoted in docs. Under unipc the 20-step even indices and the 30-step every-third indices
# coincide exactly: 999, 978, 952, 920, 882, 833, 768, 681, 555, 356.
set -uo pipefail

source /local/ovi-env/bin/activate
export PYTHONPATH=/local/wsp/src:/local/Ovi
cd /local/Ovi

python3 - <<'PY'
from ovi.ovi_fusion_engine import OviFusionEngine

for steps in (20, 30):
    # get_scheduler_time_steps is a method; call it on a bare instance without loading
    # the model. The scheduler construction does not touch weights.
    engine = OviFusionEngine.__new__(OviFusionEngine)
    try:
        _sched, timesteps = engine.get_scheduler_time_steps(
            sampling_steps=steps, solver_name="unipc", device="cpu", shift=5.0
        )
        values = [float(t) for t in timesteps]
        print(f"{steps} steps -> {len(values)} timesteps")
        print(f"  first 5: {[round(v, 2) for v in values[:5]]}")
        print(f"  last 5:  {[round(v, 2) for v in values[-5:]]}")
        print(f"  range:   {round(min(values), 2)} .. {round(max(values), 2)}")
    except Exception as exc:
        print(f"{steps} steps -> could not build schedule: {type(exc).__name__}: {exc}")
    print()

# Compare the timestep at each step index.
try:
    engine = OviFusionEngine.__new__(OviFusionEngine)
    _s20, t20 = engine.get_scheduler_time_steps(
        sampling_steps=20, solver_name="unipc", device="cpu", shift=5.0
    )
    _s30, t30 = engine.get_scheduler_time_steps(
        sampling_steps=30, solver_name="unipc", device="cpu", shift=5.0
    )
    v20 = [float(t) for t in t20]
    v30 = [float(t) for t in t30]
    print("=== same step index, different timestep? ===")
    print(f"{'idx':>4} {'20-step t':>10} {'30-step t':>10} {'diff':>8}")
    for i in (0, 2, 4, 6, 8, 10, 12, 14, 16, 18):
        if i < len(v20) and i < len(v30):
            print(f"{i:>4} {v20[i]:>10.2f} {v30[i]:>10.2f} {abs(v20[i]-v30[i]):>8.2f}")
except Exception as exc:
    print(f"comparison failed: {type(exc).__name__}: {exc}")
PY
