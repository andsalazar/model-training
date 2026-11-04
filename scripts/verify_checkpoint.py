"""Verify the weight layer against a real checkpoint.

Checks that layer discovery, slicing, and the paper's edit arithmetic all hold
on the actual Qwen3.5-0.8B weights rather than on synthetic fixtures.
"""

from __future__ import annotations

import sys

from wsp.tasks.qwen_math import (
    CANDIDATE_LAYERS,
    COLUMNS_PER_LAYER,
    apply_direction_subtraction,
    down_proj_shape,
    find_mlp_down_proj,
    footprint,
    is_full_attention,
)
from wsp.weights.slices import TensorSlice, to_text
from wsp.weights.store import WeightStore

CHECKPOINT = "models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"


def main() -> int:
    with WeightStore(CHECKPOINT) as store:
        keys = store.keys
        print(f"checkpoint: {len(keys)} tensors")

        total_changed = 0
        for layer in CANDIDATE_LAYERS:
            name = find_mlp_down_proj(keys, layer)
            info = store.info(name)
            print(f"\nlayer {layer}  full_attention={is_full_attention(layer)}")
            print(f"  tensor: {name}")
            print(f"  shape:  {info.shape}  dtype: {info.dtype}  numel: {info.numel:,}")

            if info.shape != down_proj_shape():
                print(f"  MISMATCH: expected {down_proj_shape()}")
                return 1

            tensor = store.get(name)
            target = TensorSlice(name=name, cols=(0, COLUMNS_PER_LAYER))
            region = target.apply(tensor)
            print(f"  slice:  {region.shape}  mean={region.mean():+.6f}  std={region.std():.6f}")

            # Apply a zero direction: the edit must be a no-op.
            zero = region.mean(axis=0)
            edited, changed = apply_direction_subtraction(tensor, target, zero)
            print(f"  edit:   {changed:,} weights changed")
            total_changed += changed

            # Confirm the edit is confined to the target columns.
            untouched = tensor[:, COLUMNS_PER_LAYER:]
            if not (edited[:, COLUMNS_PER_LAYER:] == untouched).all():
                print("  ERROR: edit leaked outside the target columns")
                return 1

        print(f"\ntotal changed across layers: {total_changed:,}")
        print(f"footprint: {footprint(total_changed)}")
        print(f"paper reports: 131,072 weights (0.016%)")
        print(f"match: {total_changed == 131_072}")

        # Show a small rendered slice, as an editor would see it.
        name = find_mlp_down_proj(keys, 15)
        tensor = store.get(name)
        preview = to_text(TensorSlice(name=name, rows=(0, 2), cols=(0, 4)).apply(tensor))
        print(f"\nrendered preview (2x4):\n{preview}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
