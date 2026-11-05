"""Check whether the derived direction is stable across prompt pairs.

A weak contrast cosine means each pair's direction is partly noise. That is only
fatal if the noise dominates: if independent prompt pairs yield directions that
agree with each other, the shared component is the real signal and the editor can
act on it. This measures agreement between directions from different pairs.
"""

from __future__ import annotations

import sys

MODEL = "/workspace/qwen"
TENSOR = "model.language_model.layers.23.mlp.down_proj.weight"

LOOPING = [
    "Let me check. Let me check. Let me check. Let me check again.",
    "Let me verify this. Let me verify this. Let me verify this. "
    "Let me verify this. Let me verify this. Let me verify this.",
    "Wait, let me reconsider. Actually, let me reconsider. "
    "Hmm, let me reconsider. But wait, let me reconsider.",
    "I should double-check. I should double-check. I should double-check. "
    "I should double-check. I should double-check.",
]

CALM = [
    "The answer is 42. Done.",
    "Compute 17 * 23 = 391. The result is 391. Finished.",
    "391",
    "The sum is 391. That is final.",
]


def main() -> int:
    import numpy as np

    from wsp.activations import contrast_cosine, direction_from_contrast
    from wsp.runner import TransformersActivationCapturer

    cap = TransformersActivationCapturer(MODEL, device="cuda", dtype="bfloat16")
    cap.reload(MODEL)
    print("model loaded\n")

    dirs = []
    for i, (loop, calm) in enumerate(zip(LOOPING, CALM)):
        a = cap.capture(loop, TENSOR)
        b = cap.capture(calm, TENSOR)
        vec = direction_from_contrast(a, b)
        dirs.append(vec)
        print(f"pair {i}: contrast_cosine {contrast_cosine(a, b):+.4f}")

    print("\npairwise agreement between directions:")
    n = len(dirs)
    cosines = []
    for i in range(n):
        for j in range(i + 1, n):
            c = float(np.dot(dirs[i], dirs[j]))
            cosines.append(c)
            print(f"  {i} vs {j}: {c:+.4f}")

    mean = float(np.mean(cosines))
    print(f"\nmean pairwise cosine: {mean:+.4f}")
    if mean < 0.0:
        print("WARNING: directions from different pairs disagree; the contrast")
        print("is dominated by prompt-specific noise, not a shared looping signal.")
    else:
        print("Directions agree in sign across pairs: a shared component exists.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
