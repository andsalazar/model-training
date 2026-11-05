"""Exercise the activation capturer against the real Qwen checkpoint on the pod.

This is the check that could not be done locally: it needs the real 4B model and
a GPU. It captures activations from a real layer, derives a direction from a
looping/calm contrast, and ranks the weight columns most aligned with it.

Run on the pod:
    PYTHONPATH=/workspace/src /workspace/venv2/bin/python scripts/pod_capture_check.py
"""

from __future__ import annotations

import sys
import time

MODEL = "/workspace/qwen"
CHECKPOINT = "/workspace/qwen/model.safetensors-00001-of-00001.safetensors"
TENSOR = "model.language_model.layers.23.mlp.down_proj.weight"


def main() -> int:
    import numpy as np

    from wsp.activations import (
        contrast_cosine,
        direction_from_contrast,
        summarize_direction,
        top_aligned_columns,
    )
    from wsp.runner import TransformersActivationCapturer
    from wsp.weights.store import WeightStore

    print("loading store...")
    store = WeightStore(CHECKPOINT)
    print("tensor present:", TENSOR in store.keys)
    weight = store.get(TENSOR)
    print("weight shape:", weight.shape, weight.dtype)

    print("\nloading capturer (this loads the model onto the GPU)...")
    t0 = time.time()
    cap = TransformersActivationCapturer(MODEL, device="cuda", dtype="bfloat16")
    cap.reload(MODEL)
    print(f"loaded in {time.time() - t0:.1f}s")

    # The checkpoint key; the capturer maps it onto the loaded model's path.
    module = TENSOR
    looping = "Let me check. Let me check. Let me check. Let me check again."
    calm = "The answer is 42. Done."

    print("\ncapturing looping activations...")
    t0 = time.time()
    a_loop = cap.capture(looping, module)
    print(f"  shape {a_loop.shape} in {time.time() - t0:.1f}s")

    print("capturing calm activations...")
    t0 = time.time()
    a_calm = cap.capture(calm, module)
    print(f"  shape {a_calm.shape} in {time.time() - t0:.1f}s")

    # The two prompts tokenize to different lengths; compare the last token of
    # each, which is the state the model would generate from.
    last_loop = a_loop[-1]
    last_calm = a_calm[-1]
    print("\nlast-token norms:", round(float(np.linalg.norm(last_loop)), 4),
          round(float(np.linalg.norm(last_calm)), 4))

    vec = direction_from_contrast(a_loop, a_calm)
    contrast = contrast_cosine(a_loop, a_calm)
    summary = summarize_direction(
        "loopdir", 23, vec, n_tokens=a_loop.shape[0], contrast=contrast
    )
    print("\n" + summary.as_text())

    ranked = top_aligned_columns(weight, vec, k=8)
    print("\ntop aligned columns:")
    for col, score in ranked:
        print(f"  col {col:5d}  alignment {score:+.4f}")

    # A real direction must be non-degenerate: not all zeros, and not so tiny
    # that the ranking is noise.
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        print(f"\nFAILED: direction is degenerate (norm {norm})")
        return 1
    if len({c for c, _ in ranked}) < 2:
        print("\nFAILED: ranking collapsed to one column")
        return 1

    print("\nALL CAPTURE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
