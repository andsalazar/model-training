"""Verify the activation tooling imports and works on the pod.

Run on the pod: PYTHONPATH=/workspace/src /workspace/venv2/bin/python scripts/pod_check.py
"""

from __future__ import annotations

import sys


def main() -> int:
    import numpy as np

    from wsp.activations import (
        direction_from_contrast,
        subtract_direction,
        top_aligned_columns,
    )
    from wsp.audit import audit_run
    from wsp.tools import TOOL_SCHEMA

    print("imports OK")

    looping = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    calm = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    vec = direction_from_contrast(looping, calm)
    print("direction:", vec.tolist())

    weight = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    edited = subtract_direction(weight, None, vec, scale=1.0)
    print("edited row0:", edited[0].tolist())

    ranked = top_aligned_columns(weight, vec, k=2)
    print("ranked:", [(c, round(s, 4)) for c, s in ranked])

    names = {t["name"] for t in TOOL_SCHEMA}
    expected = {
        "capture_activations",
        "capture_direction",
        "aligned_columns",
        "list_directions",
    }
    missing = expected - names
    if missing:
        print("MISSING TOOLS:", sorted(missing))
        return 1
    print("tool schema OK:", len(names), "tools")

    # The auditor must reject a conclusion that claims activation numbers with
    # no capture call behind it.
    run = {
        "conclusion": "The activation norms peak at 3.4 in layer 12.",
        "tool_calls": [{"name": "tensor_stats", "ok": True}],
        "fabricated": [],
    }
    result = audit_run(run)
    print("audit clean:", result.clean)
    if result.clean:
        print("FAILED: auditor accepted an unlicensed activation claim")
        return 1
    print("audit correctly rejected the unlicensed claim")

    print("\nALL POD CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
