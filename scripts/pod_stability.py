"""Stage 3: does threshold-based column selection reproduce where top-k did not?

Stage 2 failed because the top-12 cut sat inside a plateau: alignment cosines tie
to within 0.0012 at rank 12 while ranks 1..40 span only 0.049, so different
hardware picked different columns and inverted the verdict.

`select_stable_columns` replaces the rank cut with a *score threshold*: keep every
candidate column whose cosine is at least `ratio` times the best cosine. A
column's membership then depends on its own score relative to the maximum, not on
how many columns happen to sit slightly above it.

This script tests the fix three ways:

  1. STABILITY -- select on the real weight, then on the same weight perturbed by
     tiny numerical noise. Report Jaccard overlap. A rank cut should score badly;
     a threshold cut should score well. Report both so the comparison is explicit.

  2. EFFECT -- apply the threshold-selected edit and measure a BROAD probe set
     (not just arithmetic), so an edit that damages other behaviours is caught.

  3. SIZE HONESTY -- report how many columns the threshold admits. If the ranking
     is a plateau the threshold admits many, and the result should say so rather
     than pretend 12 was meaningful.

Run as a file, not a heredoc: vLLM spawns a core process that re-imports main.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

import numpy as np

from wsp.activations import (
    direction_from_contrast,
    selection_overlap,
    select_stable_columns,
    top_aligned_columns,
)
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
SCALE = 0.5
MAX_TOKENS = 2048

CAPTURE_VERBOSE = "What is 12 times 11? Reply with only the digits."
CAPTURE_TERSE = "What is 12 times 11? Answer with just the number."

#: A BROAD probe set, because Stage 1's arithmetic-only set licensed a VALIDATED
#: verdict for an edit that damaged unrelated behaviour. The two guards include a
#: long-form task on purpose: a global verbosity change hides behind a short guard.
PROBES = [
    ("arith_target", "What is 12 times 11? Reply with only the digits.", "132"),
    ("arith_17x23", "What is 17 times 23? Reply with only the digits.", "391"),
    ("arith_347x28", "What is 347 times 28? Reply with only the digits.", "9716"),
    ("arith_9x8", "What is 9 times 8? Reply with only the digits.", "72"),
    (
        "logic",
        "All cats are mammals. All mammals are animals. Are all cats animals? "
        "Reply with only yes or no.",
        "yes",
    ),
    (
        "fact_planet",
        "Which planet is closest to the Sun? Reply with only the name.",
        "Mercury",
    ),
    ("guard_capital", "Name the capital of France. Reply with only the city.", "Paris"),
    ("guard_hammer", "List three uses of a hammer, one per line.", "nail"),
    ("guard_poem", "Write two lines of a poem about the sea.", "sea"),
]

#: Ratios to try. A tighter ratio means fewer, more-confident columns.
RATIOS = [0.95, 0.90, 0.80]


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


def verdicts(baseline: list[dict], after: list[dict]) -> list[dict]:
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


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/stability.json"

    print("capturing direction at layer 12...", flush=True)
    capturer = TransformersActivationCapturer(MODEL)
    verbose = capturer.capture(CAPTURE_VERBOSE, TENSOR, max_tokens=2048)
    terse = capturer.capture(CAPTURE_TERSE, TENSOR, max_tokens=2048)
    direction = direction_from_contrast(verbose, terse)

    store = WeightStore(Path(CKPT))
    weight = store.get(store.resolve(TENSOR)).copy()

    # -- 1. stability: rank cut vs threshold cut under tiny numerical jitter ----
    rng = np.random.default_rng(0)
    jitter = rng.normal(scale=1e-6, size=weight.shape).astype(np.float32)
    weight_j = weight + jitter

    rank_a = [c for c, _ in top_aligned_columns(weight, direction, k=12)]
    rank_b = [c for c, _ in top_aligned_columns(weight_j, direction, k=12)]
    rank_overlap = len(set(rank_a) & set(rank_b)) / max(len(set(rank_a) | set(rank_b)), 1)
    print(
        f"\nRANK cut  (k=12): overlap under 1e-6 jitter = {rank_overlap:.3f}  "
        f"({len(set(rank_a) & set(rank_b))}/12 shared)",
        flush=True,
    )

    ratio_report = {}
    for ratio in RATIOS:
        sa = select_stable_columns(weight, direction, ratio=ratio, candidate_depth=200)
        sb = select_stable_columns(weight_j, direction, ratio=ratio, candidate_depth=200)
        overlap = selection_overlap(sa, sb)
        ratio_report[ratio] = {
            "k": sa.k,
            "k_jittered": sb.k,
            "overlap": overlap,
            "columns": list(sa.columns),
            "describe": sa.describe(),
        }
        print(f"RATIO {ratio:.2f}: {sa.describe()}", flush=True)
        print(f"          overlap under 1e-6 jitter = {overlap:.3f}", flush=True)

    # -- 2. effect of each ratio-selected edit on the broad probe set ----------
    runner = VLLMRunner(MODEL)
    print("\n=== baseline ===", flush=True)
    baseline = measure(runner, "baseline")

    results = {}
    for ratio in RATIOS:
        cols = list(ratio_report[ratio]["columns"])
        testable = [c for c in cols if 0 <= c < weight.shape[1]]
        if not testable:
            print(f"\nratio {ratio}: no usable columns, skipping", flush=True)
            continue

        print(
            f"\n=== ratio {ratio:.2f}: editing {len(testable)} columns ===",
            flush=True,
        )
        edited = weight.copy()
        edited[:, testable] -= SCALE * direction[:, None]
        changed = int(np.count_nonzero(~np.isclose(edited, weight, rtol=0.0, atol=0.0)))
        ckpt = write_edited_checkpoint(
            store.path, TENSOR, edited, Path(f"/local/edit_ratio_{int(ratio * 100)}")
        )
        print(f"  wrote {ckpt} ({changed} weights changed)", flush=True)

        runner.reload(str(ckpt))
        after = measure(runner, f"r{int(ratio * 100)}")
        verdict = verdicts(baseline, after)
        results[str(ratio)] = {
            "k": len(testable),
            "columns": testable,
            "baseline": baseline,
            "edited": after,
            "verdicts": verdict,
        }

        print(f"\n  --- ratio {ratio:.2f} comparison ---", flush=True)
        for v in verdict:
            print(
                f"  {v['probe']:16s} {v['baseline']:5d} -> {v['edited']:5d}  {v['state']}",
                flush=True,
            )
        harmful = [v for v in verdict if v["state"] in ("regressed", "broken")]
        if harmful:
            print(
                f"  -> HARMFUL: " + ", ".join(v["probe"] for v in harmful),
                flush=True,
            )
        else:
            improved = [v for v in verdict if v["state"] == "improved"]
            print(f"  -> no damage; {len(improved)} improved", flush=True)

    payload = {
        "layer": LAYER,
        "scale": SCALE,
        "rank_cut_overlap": rank_overlap,
        "rank_cut_columns": rank_a,
        "ratios": {str(k): v for k, v in ratio_report.items()},
        "effects": results,
    }
    Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
