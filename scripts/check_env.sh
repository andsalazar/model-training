#!/usr/bin/env bash
# Report whether the pod's venv has vLLM and math-verify.
set -u

/workspace/venv2/bin/python - <<'PY'
for name in ("vllm", "math_verify", "datasets", "transformers"):
    try:
        mod = __import__(name)
        print(f"{name:14s} OK  {getattr(mod, '__version__', '')}")
    except Exception as exc:
        print(f"{name:14s} MISSING ({type(exc).__name__})")
PY
