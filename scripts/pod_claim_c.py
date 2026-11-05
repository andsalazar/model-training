"""Stage 2 Claim C: is the over-thinking direction shared, or layer/behaviour specific?

Stage 1 found direction_similarity near -0.06 between layers 6, 10 and 12, yet
edits at both 6 and 12 validated. Two readings are possible and they mean very
different things:

  shared      one global over-thinking direction exists and any layer can
              express it; cross-layer disagreement would be an artefact
  local       each layer carries its own working direction; the method is
              "find a working layer", not "find the canonical direction"

This script settles it by capturing directions on a grid of behaviour x layer and
comparing every pair. "Within" compares one behaviour across layers; "across"
compares different behaviours at the same layer. If the direction were global,
both would be high; if it is local, both should be near zero.

The survey showed the model over-thinks on far more than arithmetic: 232 tokens
to answer "Paris", 1516 to answer a yes/no logic question. So "behaviour" here
means three genuinely different question types, not three phrasings of one.

Run as a file, not a heredoc: vLLM spawns a core process that re-imports main.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

import numpy as np

from wsp.activations import contrast_cosine, direction_from_contrast
from wsp.runner import TransformersActivationCapturer

MODEL = "/local/models/qwen"
LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 23]

#: Each behaviour is a (verbose phrasing, terse phrasing) pair of the SAME
#: question, so the contrast is over-thinking rather than topic.
BEHAVIOURS: dict[str, tuple[str, str]] = {
    "arith": (
        "What is 12 times 11? Reply with only the digits.",
        "What is 12 times 11? Answer with just the number.",
    ),
    "logic": (
        "All cats are mammals. All mammals are animals. Are all cats animals? "
        "Reply with only yes or no.",
        "All cats are mammals. All mammals are animals. Are all cats animals? "
        "Answer yes or no.",
    ),
    "fact": (
        "Which planet is closest to the Sun? Reply with only the name.",
        "Which planet is closest to the Sun? Answer with just the name.",
    ),
}


def tensor_for(layer: int) -> str:
    return f"model.language_model.layers.{layer}.mlp.down_proj.weight"


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/stage2_claimC.json"
    capturer = TransformersActivationCapturer(MODEL)

    directions: dict[tuple[str, int], np.ndarray] = {}
    contrasts: dict[tuple[str, int], float] = {}

    # Capture the grid. max_tokens is generous so the verbose prompt is not cut
    # off: a truncated state would fold "where it stopped" into the direction.
    for behaviour, (verbose, terse) in BEHAVIOURS.items():
        for layer in LAYERS:
            tensor = tensor_for(layer)
            try:
                a = capturer.capture(verbose, tensor, max_tokens=2048)
                b = capturer.capture(terse, tensor, max_tokens=2048)
                directions[(behaviour, layer)] = direction_from_contrast(a, b)
                contrasts[(behaviour, layer)] = contrast_cosine(a, b)
            except Exception as exc:  # noqa: BLE001 - report, keep going
                print(f"FAIL {behaviour} L{layer}: {exc}", flush=True)
                continue
            print(
                f"{behaviour:8s} L{layer:2d} contrast_cosine {contrasts[(behaviour, layer)]:+.4f}",
                flush=True,
            )

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return 0.0 if denom == 0 else float(np.dot(a, b) / denom)

    # Within-behaviour, across layers: is one behaviour's direction stable?
    within: dict[str, list[float]] = {}
    for behaviour in BEHAVIOURS:
        scores = []
        keys = [k for k in directions if k[0] == behaviour]
        for i, ka in enumerate(keys):
            for kb in keys[i + 1 :]:
                scores.append(cos(directions[ka], directions[kb]))
        within[behaviour] = scores
        if scores:
            print(
                f"\nwithin {behaviour:8s} across layers: "
                f"mean {np.mean(scores):+.3f}  min {min(scores):+.3f}  max {max(scores):+.3f}",
                flush=True,
            )

    # Across behaviours at the same layer: is there one shared direction?
    across: dict[int, list[float]] = {}
    for layer in LAYERS:
        keys = [k for k in directions if k[1] == layer]
        scores = []
        for i, ka in enumerate(keys):
            for kb in keys[i + 1 :]:
                scores.append(cos(directions[ka], directions[kb]))
        across[layer] = scores
        if scores:
            print(
                f"across behaviours at L{layer:2d}: "
                f"mean {np.mean(scores):+.3f}",
                flush=True,
            )

    payload = {
        "layers": LAYERS,
        "behaviours": list(BEHAVIOURS),
        "contrasts": {f"{b}@{l}": v for (b, l), v in contrasts.items()},
        "within_behaviour_across_layers": {
            b: {"mean": float(np.mean(s)) if s else None, "scores": s}
            for b, s in within.items()
        },
        "across_behaviours_same_layer": {
            str(l): {"mean": float(np.mean(s)) if s else None, "scores": s}
            for l, s in across.items()
        },
        "pairwise": {
            f"{b1}@{l1}|{b2}@{l2}": cos(directions[(b1, l1)], directions[(b2, l2)])
            for (b1, l1) in directions
            for (b2, l2) in directions
            if (b1, l1) < (b2, l2)
        },
    }
    Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
