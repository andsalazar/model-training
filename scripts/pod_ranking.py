"""How brittle is the column ranking at the top-k cut?

Stage 2 surfaced a fragility that needs quantifying. Two runs used the same
layer (12), same scale (0.5), same top_k (12) and the same capture prompts, yet
produced materially different edits and opposite verdicts:

  Stage 1   cols ... 1575, 46    (RTX 4000 Ada)     -- validated
  Claim A   cols ... 1176, 2177  (RTX PRO 4500)    -- harmful

Measurement is perfectly deterministic on fixed hardware (7 repeats, stdev 0.0),
so the difference is not sampling. It is either cross-GPU numerics changing the
ranking, or genuine near-ties at the cut boundary.

This prints the alignment score spectrum around top_k so the size of the tie can
be read directly. If column 12 and column 40 differ by 0.001 in cosine, then the
"top 12" is an arbitrary choice inside a plateau, and any result that depends on
exactly 12 columns is not reproducible across hardware.

Run as a file, not a heredoc: vLLM spawns a core process that re-imports main.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

from wsp.activations import column_alignment, direction_from_contrast
from wsp.runner import TransformersActivationCapturer
from wsp.weights.store import WeightStore

MODEL = "/local/models/qwen"
CKPT = "/local/models/qwen/model.safetensors-00001-of-00001.safetensors"

LAYER = 12
TENSOR = f"model.language_model.layers.{LAYER}.mlp.down_proj.weight"

#: The four columns whose membership differed between the two runs.
DISPUTED = {1575, 46, 1176, 2177}


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/ranking.json"

    capturer = TransformersActivationCapturer(MODEL)
    verbose = capturer.capture(
        "What is 12 times 11? Reply with only the digits.", TENSOR, max_tokens=2048
    )
    terse = capturer.capture(
        "What is 12 times 11? Answer with just the number.", TENSOR, max_tokens=2048
    )
    direction = direction_from_contrast(verbose, terse)

    store = WeightStore(Path(CKPT))
    weight = store.get(store.resolve(TENSOR))
    scores = column_alignment(weight, direction)

    order = sorted(range(len(scores)), key=lambda i: -scores[i])

    print("rank\tcol\tcosine\t\tdisputed", flush=True)
    for rank, col in enumerate(order[:40], start=1):
        mark = "  <-- DISPUTED" if int(col) in DISPUTED else ""
        print(f"{rank:4d}\t{col}\t{scores[col]:+.6f}{mark}", flush=True)

    # How much does the score fall off across the cut?
    s12 = float(scores[order[11]])
    s13 = float(scores[order[12]])
    s20 = float(scores[order[19]])
    s40 = float(scores[order[39]])
    print(f"\nscore at rank 12: {s12:+.6f}", flush=True)
    print(f"score at rank 13: {s13:+.6f}  gap {s12 - s13:+.6f}", flush=True)
    print(f"score at rank 20: {s20:+.6f}  gap from 12 {s12 - s20:+.6f}", flush=True)
    print(f"score at rank 40: {s40:+.6f}  gap from 12 {s12 - s40:+.6f}", flush=True)

    # Where the disputed columns actually rank.
    print("\ndisputed column ranks:", flush=True)
    rank_of = {int(c): i + 1 for i, c in enumerate(order)}
    for col in sorted(DISPUTED):
        print(f"  col {col:5d} rank {rank_of.get(col)} cosine {scores[col]:+.6f}", flush=True)

    payload = {
        "top40": [
            {"rank": i + 1, "col": int(c), "cosine": float(scores[c])}
            for i, c in enumerate(order[:40])
        ],
        "score_rank12": s12,
        "score_rank13": s13,
        "score_rank20": s20,
        "score_rank40": s40,
        "disputed_ranks": {str(c): rank_of.get(c) for c in sorted(DISPUTED)},
        "disputed_cosines": {str(c): float(scores[c]) for c in sorted(DISPUTED)},
    }
    Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
