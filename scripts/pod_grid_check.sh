#!/bin/bash
source /local/ovi-env/bin/activate
cd /local/Ovi
export PYTHONPATH=/local/wsp/src:/local/Ovi
python3 - <<'PY'
import inspect
from ovi.ovi_fusion_engine import OviFusionEngine
import ovi.ovi_fusion_engine as m
e = OviFusionEngine.__new__(OviFusionEngine)
s12, t12 = e.get_scheduler_time_steps(12, "unipc", "cpu", 5.0)
s48, t48 = e.get_scheduler_time_steps(48, "unipc", "cpu", 5.0)
print("t12", [round(float(x), 3) for x in t12])
print("t48[::4]", [round(float(x), 3) for x in t48[::4]])
print("sigmas12", [round(float(x), 5) for x in s12.sigmas])
print("len sigmas", len(s12.sigmas), "len t", len(t12))
print("t/1000 == sigma[:-1]?", max(abs(float(a)/1000 - float(b)) for a, b in zip(t12, s12.sigmas[:-1])))
print(type(s12).__name__, "step sig:", inspect.signature(s12.step))
src = inspect.getsource(type(s12).convert_model_output)
print([l.strip() for l in src.splitlines() if "sigma" in l and ("x0" in l or "model_output" in l)][:6])
print("predict:", getattr(s12.config, "prediction_type", None), getattr(s12.config, "solver_order", None))
PY