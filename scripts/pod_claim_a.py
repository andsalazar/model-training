"""Stage 2 Claim A: does an edit derived from one behaviour fix another?

Claim C found the over-thinking direction is layer-local (+0.006 across layers)
but behaviour-general within a layer (+0.429 across behaviours). That makes a
falsifiable prediction: a direction captured from arithmetic should suppress
over-thinking on logic and factual questions too, because at any given layer the
three behaviours share a direction.

This tests it directly, in the only way that counts: capture from arithmetic,
edit, then verify with the fixed rule on probes covering all three behaviours.

The reverse is also worth knowing. If an arith-derived edit fixes logic and fact,
the feature is general. If it only fixes arith, then the +0.429 similarity is
shared direction but not shared *causation*, and the edit is behaviour-specific
after all.

Run as a file, not a heredoc: vLLM spawns a core process that re-imports main.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

import numpy as np

from wsp.activations import direction_from_contrast, top_aligned_columns
from wsp.runner import (
    TransformersActivationCapturer,
    VLLMRunner,
    classify,
    write_edited_checkpoint,
)
from wsp.weights.store import WeightStore

MODEL = "/local/models/qwen"
CKPT = "/local/models/qwen/model.safetensors-00001-of-00001.safetensors"

LAYER = 12
TENSOR = f"model.language_model.layers.{LAYER}.mlp.down_proj.weight"
TOP_K = 12
SCALE = 0.5
MAX_TOKENS = 2048

#: The direction is captured from ARITHMETIC only. The probes below cover three
#: behaviours, and only the first is the one the direction came from.
CAPTURE_VERBOSE = "What is 12 times 11? Reply with only the digits."
CAPTURE_TERSE = "What is 12 times 11? Answer with just the number."

PROBES = [
    # the behaviour the direction was derived from
    ("arith_target", "What is 12 times 11? Reply with only the digits.", "132"),
    ("arith_heldout", "What is 17 times 23? Reply with only the digits.", "391"),
    # different behaviours, same over-thinking habit
    (
        "logic_transfer",
        "All cats are mammals. All mammals are animals. Are all cats animals? "
        "Reply with only yes or no.",
        "yes",
    ),
    (
        "fact_transfer",
        "Which planet is closest to the Sun? Reply with only the name.",
        "Mercury",
    ),
    # must not move: unrelated, and a global terseness edit would shorten it
    (
        "guard_hammer",
        "List three uses of a hammer, one per line.",
        "nail",
    ),
]


def measure(runner: VLLMRunner, label: str) -> list[dict]:
    rows = []
    for name, prompt, expected in PROBES:
        text = runner.generate(prompt, max_tokens=MAX_TOKENS)
        n = runner.count_tokens(text)
        signal = classify(text, n, MAX_TOKENS)
        row = {
            "probe": name,
            "expected": expected,
            "correct": expected.lower() in text.lower(),
            **signal.as_dict(),
        }
        rows.append(row)
        print(
            f"  {label:10s} {name:16s} tokens={signal.tokens:5d} "
            f"cap={str(signal.hit_cap):5s} ok={row['correct']}",
            flush=True,
        )
    return rows


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/claim_a.json"

    print("capturing direction from ARITHMETIC only...", flush=True)
    capturer = TransformersActivationCapturer(MODEL)
    verbose = capturer.capture(CAPTURE_VERBOSE, TENSOR, max_tokens=2048)
    terse = capturer.capture(CAPTURE_TERSE, TENSOR, max_tokens=2048)
    direction = direction_from_contrast(verbose, terse)
    print(f"  direction dim {direction.shape[0]}", flush=True)

    ckpt_file = Path(CKPT)
    store = WeightStore(ckpt_file)
    original = store.get(store.resolve(TENSOR)).copy()

    ranked = top_aligned_columns(original, direction, k=TOP_K)
    cols = [int(c) for c, _ in ranked]
    print(f"  ranked columns: {cols}", flush=True)

    runner = VLLMRunner(MODEL)

    print("\n=== baseline (unmodified) ===", flush=True)
    baseline = measure(runner, "baseline")

    print(f"\n=== applying edit: L{LAYER}, {TOP_K} ranked cols, scale {SCALE} ===", flush=True)
    edited = original.copy()
    edited[:, cols] -= SCALE * direction[:, None]
    changed = int(np.count_nonzero(~np.isclose(edited, original, rtol=0.0, atol=0.0)))
    ckpt_dir = write_edited_checkpoint(store.path, TENSOR, edited, Path("/local/edit_claimA"))
    print(f"  wrote {ckpt_dir} ({changed} weights changed)", flush=True)

    runner.reload(str(ckpt_dir))
    print("\n=== after edit ===", flush=True)
    after = measure(runner, "edited")

    # The same fixed rule as wsp.verify, applied here so the script is
    # self-contained: improved < 0.75x, regressed > 1.25x, 25-token noise floor.
    print("\n=== comparison ===", flush=True)
    verdicts = []
    for b, a in zip(baseline, after):
        if b["correct"] and not a["correct"]:
            state = "broken"
        elif a["hit_cap"] and not b["hit_cap"]:
            state = "regressed"
        elif abs(a["tokens"] - b["tokens"]) < 25:
            state = "unchanged"
        elif a["tokens"] / max(b["tokens"], 1) <= 0.75:
            state = "improved"
        elif a["tokens"] / max(b["tokens"], 1) >= 1.25:
            state = "regressed"
        else:
            state = "unchanged"
        verdicts.append({"probe": b["probe"], "baseline": b["tokens"],
                         "edited": a["tokens"], "state": state})
        print(
            f"  {b['probe']:16s} {b['tokens']:5d} -> {a['tokens']:5d}  {state}",
            flush=True,
        )

    payload = {
        "layer": LAYER,
        "cols": cols,
        "scale": SCALE,
        "baseline": baseline,
        "edited": after,
        "verdicts": verdicts,
    }
    Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
