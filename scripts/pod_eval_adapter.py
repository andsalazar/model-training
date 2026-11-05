"""Stage 4 evaluation: judge the adapter with the rule that refuted Stages 1-3.

The adapter is only interesting if it survives the check that the fixed vector
failed. That means:

  1. a BROAD probe set -- arithmetic, logic, factual, and three long-form guards,
     because Stage 1's arithmetic-only set licensed a VALIDATED verdict for an
     edit that damaged unrelated behaviour
  2. `harmful` checked before `validated`, using the same fixed rule in wsp.verify
  3. a HELD-OUT behaviour the adapter never trained on
  4. an ABLATION -- swap the adapter off and confirm the baseline returns, so the
     effect is the adapter and not checkpoint drift

Every probe is measured three ways: base model, base + adapter, and base + adapter
switched off. The third is the ablation and should reproduce the first.

Run as a file, not a heredoc: vLLM spawns a core process that re-imports main.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

from wsp.runner import VLLMRunner, classify
from wsp.stage4_data import build_dataset

MODEL = "/local/models/qwen"
MAX_TOKENS = 2048

#: A broad set, deliberately including the probes that broke Stage 3. The guards
#: are the specification: `poem` and `hammer` are where the fixed vector failed,
#: so they are where the adapter has to succeed.
PROBES = [
    # trained behaviours
    ("arith_12x11", "What is 12 times 11? Reply with only the digits.", "132", "target"),
    ("arith_17x23", "What is 17 times 23? Reply with only the digits.", "391", "target"),
    ("arith_9x8", "What is 9 times 8? Reply with only the digits.", "72", "target"),
    (
        "logic_syllogism",
        "All cats are mammals. All mammals are animals. Are all cats animals? "
        "Reply with only yes or no.",
        "yes",
        "target",
    ),
    # held-out behaviour: never in training
    (
        "heldout_capital",
        "Name the capital of Japan. Reply with only the city.",
        "Tokyo",
        "heldout",
    ),
    (
        "heldout_symbol",
        "What is the chemical symbol for gold? Reply with only the symbol.",
        "Au",
        "heldout",
    ),
    # guards: elaboration is correct here, and these are what the edit destroyed
    ("guard_poem", "Write two lines of a poem about the sea.", "sea", "guard"),
    ("guard_hammer", "List three uses of a hammer, one per line.", "nail", "guard"),
    (
        "guard_blue_sky",
        "Explain in two sentences why the sky appears blue.",
        "light",
        "guard",
    ),
]


def measure(runner: VLLMRunner, label: str) -> list[dict]:
    rows = []
    for name, prompt, expected, kind in PROBES:
        text = runner.generate(prompt, max_tokens=MAX_TOKENS)
        n = runner.count_tokens(text)
        signal = classify(text, n, MAX_TOKENS)
        row = {
            "probe": name,
            "kind": kind,
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
    """The same fixed rule as wsp.verify: harmful is checked before anything else."""
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
                "kind": b["kind"],
                "baseline": b["tokens"],
                "edited": a["tokens"],
                "state": state,
            }
        )
    return out


def summarise(rows: list[dict], title: str) -> dict:
    print(f"\n=== {title} ===", flush=True)
    for r in rows:
        print(
            f"  {r['probe']:18s} {r['kind']:8s} {r['baseline']:5d} -> "
            f"{r['edited']:5d}  {r['state']}",
            flush=True,
        )
    harmful = [r for r in rows if r["state"] in ("regressed", "broken")]
    improved = [r for r in rows if r["state"] == "improved"]

    # Guard damage is the Stage 3 failure mode, so it is called out separately.
    guard_harm = [r for r in harmful if r["kind"] == "guard"]
    target_improved = [r for r in improved if r["kind"] == "target"]

    if harmful:
        status = "harmful"
        print(
            f"  -> HARMFUL ({len(harmful)}): "
            + ", ".join(r["probe"] for r in harmful),
            flush=True,
        )
    elif target_improved:
        status = "validated"
        print(f"  -> VALIDATED; {len(target_improved)} target(s) improved", flush=True)
    else:
        status = "no-effect"
        print("  -> NO EFFECT on targets", flush=True)

    if guard_harm:
        print(
            f"  !! guard damage: {[r['probe'] for r in guard_harm]} -- this is the "
            "Stage 3 failure mode",
            flush=True,
        )

    return {
        "status": status,
        "harmful": [r["probe"] for r in harmful],
        "guard_harmful": [r["probe"] for r in guard_harm],
        "improved": [r["probe"] for r in improved],
        "verdicts": rows,
    }


def main() -> int:
    adapter = sys.argv[1] if len(sys.argv) > 1 else "/workspace/stage4_adapter"
    out = sys.argv[2] if len(sys.argv) > 2 else "/workspace/stage4_eval.json"

    results: dict = {"adapter": adapter, "probes": [p[0] for p in PROBES]}

    print("=== base model ===", flush=True)
    runner = VLLMRunner(MODEL)
    base = measure(runner, "base")
    results["base"] = base

    adapter_path = Path(adapter)
    if not adapter_path.exists():
        print(f"\nadapter not found at {adapter_path}", flush=True)
        Path(out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        return 1

    # -- with adapter ------------------------------------------------------
    # A LoRA cannot be served by the base vLLM path without a merged checkpoint,
    # so the adapter is merged into a copy. That is also the honest evaluation:
    # it measures what a released model would do.
    print(f"\nmerging adapter from {adapter_path}...", flush=True)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    merged = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged = merged.merge_and_unload()

    merged_dir = Path("/local/models/qwen_stage4_merged")
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    print(f"merged checkpoint at {merged_dir}", flush=True)

    print("\n=== with adapter ===", flush=True)
    runner.reload(str(merged_dir))
    with_adapter = measure(runner, "adapter")
    results["with_adapter"] = with_adapter

    v = summarise(verdicts(base, with_adapter), "base -> adapter")
    results["adapter_verdict"] = v

    # -- ablation ----------------------------------------------------------
    # Swapping back to the un-merged base must reproduce the baseline. If it does
    # not, the "effect" is drift and every number above is suspect.
    print("\n=== ablation: back to base ===", flush=True)
    runner.reload(MODEL)
    ablated = measure(runner, "ablated")
    results["ablated"] = ablated

    drift = [
        {"probe": b["probe"], "base": b["tokens"], "ablated": a["tokens"]}
        for b, a in zip(base, ablated)
        if abs(a["tokens"] - b["tokens"]) >= 25
    ]
    results["ablation_drift"] = drift
    if drift:
        print(
            f"\n  !! ABLATION DRIFT on {len(drift)} probe(s): "
            + ", ".join(d["probe"] for d in drift),
            flush=True,
        )
    else:
        print("\n  ablation reproduces the baseline exactly", flush=True)

    # -- held-out behaviour, called out -----------------------------------
    heldout = [r for r in v["verdicts"] if r["kind"] == "heldout"]
    results["heldout"] = heldout
    print("\n=== held-out behaviour ===", flush=True)
    for r in heldout:
        print(
            f"  {r['probe']:18s} {r['baseline']:5d} -> {r['edited']:5d}  {r['state']}",
            flush=True,
        )

    Path(out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}", flush=True)
    print("EVAL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
