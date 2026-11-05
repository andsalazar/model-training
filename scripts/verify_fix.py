"""Verify the fixed code on the pod, and prove the tools read per-layer data.

Run on the pod: /workspace/venv2/bin/python /workspace/scripts/verify_fix.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/workspace/src")

import numpy as np

from wsp.agent import _FABRICATED_RESULT, _extract_calls
from wsp.weights.store import WeightStore

CKPT = "/workspace/qwen/model.safetensors-00001-of-00001.safetensors"


def main() -> int:
    print("=== parser: CALL written mid-line ===")
    text = 'Let me check layer 23 and 12CALL tensor_stats\n{"tensor": "w", "cols": "0:8"}'
    calls = _extract_calls(text)
    print(f"calls: {calls}")
    assert calls == [("tensor_stats", '{"tensor": "w", "cols": "0:8"}')], calls

    print("\n=== parser: nested braces in arguments ===")
    text = 'CALL run_model\n{"prompt": "value is {a: 1} ok"}'
    calls = _extract_calls(text)
    print(f"calls: {calls}")
    assert calls[0][1] == '{"prompt": "value is {a: 1} ok"}', calls

    print("\n=== fabrication detector ===")
    text = 'CALL list_tensors\n{}\n\nRESULT of tensor_stats:\n{"top_norms": []}'
    found = [m.group(0) for m in _FABRICATED_RESULT.finditer(text)]
    print(f"fabricated: {found}")
    assert found == ["RESULT of tensor_stats"], found

    print("\n=== tools read per-layer data ===")
    with WeightStore(CKPT) as store:
        layers = {}
        for layer in (0, 6, 12, 23):
            name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
            array = store.get(name)
            window = np.asarray(array[:, 1150:1250], dtype=np.float32)
            norms = np.linalg.norm(window, axis=0)
            order = np.argsort(norms)[::-1][:3]
            layers[layer] = [
                (int(1150 + i), round(float(norms[i]), 4)) for i in order
            ]
            print(f"layer {layer:2d}: top cols {layers[layer]}")

        distinct = {tuple(v) for v in layers.values()}
        print(f"\ndistinct top-3 across 4 layers: {len(distinct)}")
        assert len(distinct) == 4, "layers returned identical statistics"

        first_rows = {}
        for layer in (0, 12):
            name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
            first_rows[layer] = np.asarray(
                store.get(name)[0:4, 1180:1205], dtype=np.float32
            )
        same = np.array_equal(first_rows[0], first_rows[12])
        print(f"read_slice identical for layer 0 vs 12: {same}")
        print(f"  layer 0  row0[:5]: {first_rows[0][0][:5]}")
        print(f"  layer 12 row0[:5]: {first_rows[12][0][:5]}")
        assert not same, "read_slice returned identical values for two layers"

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
