"""Separate the adapter's effect from the sampler's.

The adapter reduced length under temperature 0.6 with presence penalty 1.5, but
still ran to the cap under greedy. Two explanations remain, and they matter very
differently for the claim:

  adapter   the adapter genuinely stops over-thinking, and greedy decoding is
            simply outside its training distribution
  sampler   the presence penalty is doing the work, penalising the repeated
            re-verification that is the model's failure mode; the adapter
            contributes little

This measures a 2x2 grid: {base, adapter} x {with penalty, without penalty},
holding temperature at 0.6 so only the penalty varies. If the adapter helps
without the penalty, the mechanism is real. If it only helps with the penalty, the
result is a decoding artefact and the Stage 4 claim shrinks.

Run as a file, not a heredoc: vLLM spawns a core process that re-imports main.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

MODEL = "/local/models/qwen"
MERGED = "/local/models/qwen_stage4_merged"
MAX_TOKENS = 2048

PROBES = [
    ("arith_12x11", "What is 12 times 11? Reply with only the digits.", "132"),
    ("arith_17x23", "What is 17 times 23? Reply with only the digits.", "391"),
    ("logic", "All cats are mammals. All mammals are animals. Are all cats "
              "animals? Reply with only yes or no.", "yes"),
    ("guard_poem", "Write two lines of a poem about the sea.", "poem"),
    ("guard_hammer", "List three uses of a hammer, one per line.", "use"),
]


def configs():
    from vllm import SamplingParams

    # Temperature fixed so only the penalty varies.
    return {
        "T0.6_pp0": SamplingParams(
            max_tokens=MAX_TOKENS, temperature=0.6, top_p=0.95, top_k=20,
            presence_penalty=0.0, seed=0,
        ),
        "T0.6_pp1.5": SamplingParams(
            max_tokens=MAX_TOKENS, temperature=0.6, top_p=0.95, top_k=20,
            presence_penalty=1.5, seed=0,
        ),
    }


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/sampler.json"
    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    rows = []
    cfg = configs()

    for label, path in (("base", MODEL), ("adapter", MERGED)):
        if not Path(path).exists():
            print(f"SKIP {label}: {path} missing", flush=True)
            continue
        llm = LLM(
            model=path, max_model_len=4096, gpu_memory_utilization=0.85,
            enforce_eager=True, disable_log_stats=True,
        )
        for name, prompt, expected in PROBES:
            messages = [{"role": "user", "content": prompt}]
            rendered = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
            for cfg_name, params in cfg.items():
                o = llm.generate([rendered], params)[0]
                text = o.outputs[0].text
                n = len(o.outputs[0].token_ids)
                rows.append({
                    "model": label, "probe": name, "config": cfg_name,
                    "tokens": n, "correct": expected.lower() in text.lower(),
                    "text": text,
                })
                print(
                    f"  {label:8s} {name:14s} {cfg_name:12s} {n:5d} tok "
                    f"ok={expected.lower() in text.lower()}",
                    flush=True,
                )
        del llm

    # The contrast that decides it: does the adapter help WITHOUT the penalty?
    print("\n=== adapter effect, by penalty ===", flush=True)
    by = {(r["model"], r["probe"], r["config"]): r for r in rows}
    summary = {}
    for cfg_name in cfg:
        gains = []
        for name, _, _ in PROBES:
            b = by.get(("base", name, cfg_name))
            a = by.get(("adapter", name, cfg_name))
            if b and a:
                gains.append(b["tokens"] - a["tokens"])
        mean_gain = sum(gains) / len(gains) if gains else 0.0
        summary[cfg_name] = {"mean_token_reduction": mean_gain}
        print(
            f"  {cfg_name:12s} mean reduction {mean_gain:7.1f} tok  "
            f"per-probe {gains}",
            flush=True,
        )

    Path(out).write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(f"\nwritten: {out}", flush=True)
    print("SAMPLER_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
