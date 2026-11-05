"""Stage 2: was Stage 1's VALIDATED verdict an artifact of a narrow probe set?

Claim A used the same layer (12), the same scale (0.5) and 10 of the same 12
ranked columns as Stage 1's validated ranked-column sweep -- yet it regressed the
factual probe and the long-form guard to the token cap.

The difference is the probe set. Stage 1 tested target + three arithmetic
held-outs + one SHORT non-math guard (204 tokens). Claim A tested target + one
arithmetic held-out + logic (621) + fact (482) + a LONG guard (1204).

This measures both probe sets on the SAME edited checkpoint, so probe-set choice
is isolated from edit choice. If Stage 1's probes validate while Claim A's fail on
one identical checkpoint, then Stage 1's verdict was true of its probe set and
over-generalised in its wording.

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

CAPTURE_VERBOSE = "What is 12 times 11? Reply with only the digits."
CAPTURE_TERSE = "What is 12 times 11? Answer with just the number."

#: The probe set Stage 1 used. Almost entirely arithmetic, with one short guard.
STAGE1_PROBES = [
    ("s1_target", "What is 12 times 11? Reply with only the digits.", "132"),
    ("s1_control", "What is 12 times 11? Answer with just the number.", "132"),
    ("s1_17x23", "What is 17 times 23? Reply with only the digits.", "391"),
    ("s1_13x14", "What is 13 times 14? Reply with only the digits.", "182"),
    ("s1_9x8", "What is 9 times 8? Reply with only the digits.", "72"),
    ("s1_guard_capital", "Name the capital of France. Reply with only the city.", "Paris"),
]

#: A broader set: different behaviours and a long-form task.
BROAD_PROBES = [
    ("b_12x11", "What is 12 times 11? Reply with only the digits.", "132"),
    ("b_17x23", "What is 17 times 23? Reply with only the digits.", "391"),
    (
        "b_logic",
        "All cats are mammals. All mammals are animals. Are all cats animals? "
        "Reply with only yes or no.",
        "yes",
    ),
    (
        "b_fact_planet",
        "Which planet is closest to the Sun? Reply with only the name.",
        "Mercury",
    ),
    ("b_guard_hammer", "List three uses of a hammer, one per line.", "nail"),
    (
        "b_guard_poem",
        "Write two lines of a poem about the sea.",
        "sea",
    ),
]


def measure(runner: VLLMRunner, probes, label: str) -> list[dict]:
    rows = []
    for name, prompt, expected in probes:
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
            f"  {label:10s} {name:18s} tokens={signal.tokens:5d} "
            f"cap={str(signal.hit_cap):5s} ok={row['correct']}",
            flush=True,
        )
    return rows


def verdicts_for(baseline: list[dict], after: list[dict]) -> list[dict]:
    out = []
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
        out.append(
            {
                "probe": b["probe"],
                "baseline": b["tokens"],
                "edited": a["tokens"],
                "state": state,
            }
        )
    return out


def summarise(rows: list[dict], title: str) -> None:
    print(f"\n=== {title} ===", flush=True)
    for r in rows:
        print(
            f"  {r['probe']:18s} {r['baseline']:5d} -> {r['edited']:5d}  {r['state']}",
            flush=True,
        )
    harmful = [r for r in rows if r["state"] in ("regressed", "broken")]
    if harmful:
        print(
            f"  -> HARMFUL ({len(harmful)} probe(s)): "
            + ", ".join(r["probe"] for r in harmful),
            flush=True,
        )
    else:
        improved = [r for r in rows if r["state"] == "improved"]
        print(f"  -> no damage; {len(improved)} improved", flush=True)


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/probe_set.json"

    print("capturing direction (arith, layer 12)...", flush=True)
    capturer = TransformersActivationCapturer(MODEL)
    verbose = capturer.capture(CAPTURE_VERBOSE, TENSOR, max_tokens=2048)
    terse = capturer.capture(CAPTURE_TERSE, TENSOR, max_tokens=2048)
    direction = direction_from_contrast(verbose, terse)

    store = WeightStore(Path(CKPT))
    original = store.get(store.resolve(TENSOR)).copy()
    cols = [int(c) for c, _ in top_aligned_columns(original, direction, k=TOP_K)]
    print(f"  ranked columns: {cols}", flush=True)

    runner = VLLMRunner(MODEL)

    print("\n=== baseline: Stage 1 probe set ===", flush=True)
    s1_base = measure(runner, STAGE1_PROBES, "baseline")
    print("\n=== baseline: broad probe set ===", flush=True)
    b_base = measure(runner, BROAD_PROBES, "baseline")

    # One edit, shared by both measurements.
    edited = original.copy()
    edited[:, cols] -= SCALE * direction[:, None]
    ckpt_dir = write_edited_checkpoint(store.path, TENSOR, edited, Path("/local/edit_probeset"))
    print(f"\nwrote {ckpt_dir}", flush=True)
    runner.reload(str(ckpt_dir))

    print("\n=== after edit: Stage 1 probe set ===", flush=True)
    s1_after = measure(runner, STAGE1_PROBES, "edited")
    print("\n=== after edit: broad probe set ===", flush=True)
    b_after = measure(runner, BROAD_PROBES, "edited")

    s1_v = verdicts_for(s1_base, s1_after)
    b_v = verdicts_for(b_base, b_after)
    summarise(s1_v, "Stage 1 probe set")
    summarise(b_v, "Broad probe set")

    payload = {
        "layer": LAYER,
        "cols": cols,
        "scale": SCALE,
        "stage1": {"baseline": s1_base, "edited": s1_after, "verdicts": s1_v},
        "broad": {"baseline": b_base, "edited": b_after, "verdicts": b_v},
    }
    Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
