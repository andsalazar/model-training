"""Probe the pod's checkpoint: tensor names and the WeightStore API."""

from __future__ import annotations

import inspect

from wsp.weights.store import WeightStore

CHECKPOINT = "/workspace/qwen/model.safetensors-00001-of-00001.safetensors"


def main() -> int:
    store = WeightStore(CHECKPOINT)
    print("WeightStore methods:", [n for n, _ in inspect.getmembers(store, inspect.ismethod) if not n.startswith("_")])
    print("keys attr type:", type(store.keys).__name__, "len:", len(store.keys))

    down = [k for k in store.keys if "down_proj" in k]
    print("\ndown_proj keys:", len(down))
    for k in down[:6]:
        print("  ", k)

    # Find the layer-23 down_proj under whatever prefix this checkpoint uses.
    l23 = [k for k in down if "layers.23." in k]
    print("\nlayer 23 down_proj:", l23)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
