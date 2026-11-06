"""Resolve a discrepancy that decides whether the adapter result is real.

Two measurements of the same merged adapter disagree:

  vLLM eval        guard_poem -> 116 tokens
  transformers greedy (in-memory merge)  guard_poem -> 2048 tokens

That is a 17x difference. Either the decoding config is doing the work (vLLM runs
temperature 0.6 with presence_penalty 1.5, which heavily discourages repetition)
or one of the two measurements is not measuring what it claims.

This runs the SAME checkpoint through vLLM with the runner's exact sampling params
and prints the full text, so "116 tokens" can be judged as content rather than as
a number. If the short output is a real poem the adapter works under that config;
if it is a truncated fragment, the earlier VALIDATED verdict was a substring
artefact.

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
    ("guard_poem", "Write two lines of a poem about the sea."),
    ("guard_hammer", "List three uses of a hammer, one per line."),
    ("guard_blue_sky", "Explain in two sentences why the sky appears blue."),
]


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/decode.json"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = []
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # Two configurations, to separate "the adapter" from "the sampler".
    configs = {
        "sampled": SamplingParams(
            max_tokens=MAX_TOKENS,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            presence_penalty=1.5,
            seed=0,
        ),
        "greedy": SamplingParams(max_tokens=MAX_TOKENS, temperature=0),
    }

    for label, path in (("base", MODEL), ("adapter", MERGED)):
        if not Path(path).exists():
            print(f"SKIP {label}: {path} missing", flush=True)
            continue
        llm = LLM(
            model=path,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
            disable_log_stats=True,
        )
        for name, prompt in PROBES:
            messages = [{"role": "user", "content": prompt}]
            rendered = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
            for cfg_name, params in configs.items():
                o = llm.generate([rendered], params)[0]
                text = o.outputs[0].text
                n = len(o.outputs[0].token_ids)
                rows.append(
                    {
                        "model": label,
                        "probe": name,
                        "config": cfg_name,
                        "tokens": n,
                        "text": text,
                    }
                )
                print(f"\n--- {label} / {name} / {cfg_name} : {n} tok ---", flush=True)
                print(text[-400:], flush=True)
        del llm

    Path(out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}", flush=True)
    print("DECODE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
