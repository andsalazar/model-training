"""Check that WeightStore resolves short tensor names on the real checkpoint.

Run on the pod:
    PYTHONPATH=/workspace/src /workspace/venv2/bin/python scripts/pod_resolve_check.py
"""

from __future__ import annotations

import sys

from wsp.weights.store import WeightStore

CHECKPOINT = "/workspace/qwen/model.safetensors-00001-of-00001.safetensors"
SHORT = "model.layers.7.mlp.down_proj.weight"
EXACT = "model.language_model.layers.7.mlp.down_proj.weight"


def main() -> int:
    with WeightStore(CHECKPOINT) as store:
        print(f"tensors in checkpoint: {len(store.keys)}")

        resolved = store.resolve(SHORT)
        print(f"short  {SHORT!r}")
        print(f"    -> {resolved!r}")
        if resolved != EXACT:
            print(f"FAILED: expected {EXACT!r}")
            return 1

        if store.resolve(EXACT) != EXACT:
            print("FAILED: exact key did not resolve to itself")
            return 1

        array = store.get(SHORT)
        print(f"get(short) shape: {array.shape}")

        info = store.info(SHORT)
        print(f"info(short).name: {info.name!r}")
        if info.name != EXACT:
            print("FAILED: info did not report the resolved key")
            return 1

    print("\nRESOLVE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
