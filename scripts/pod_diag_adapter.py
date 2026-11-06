"""Diagnostic: what does the adapter actually generate?

The eval reported 1-3 tokens on every probe, with several probes still marked
"correct" because the expected string is short enough to appear in a 1-token
output. That is a substring artifact, not a success. This prints the real text so
the failure mode is identified from evidence rather than inferred.

Run as a file, not a heredoc: vLLM spawns a core process that re-imports main.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

MODEL = "/local/models/qwen"
MERGED = "/local/models/qwen_stage4_merged"
MAX_TOKENS = 256

PROMPTS = [
    ("arith_12x11", "What is 12 times 11? Reply with only the digits.", "132"),
    ("arith_17x23", "What is 17 times 23? Reply with only the digits.", "391"),
    ("logic", "All cats are mammals. All mammals are animals. Are all cats "
              "animals? Reply with only yes or no.", "yes"),
    ("guard_poem", "Write two lines of a poem about the sea.", "a poem"),
    ("guard_hammer", "List three uses of a hammer, one per line.", "three uses"),
    ("guard_blue_sky", "Explain in two sentences why the sky appears blue.", "an explanation"),
]


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/diag.json"
    from vllm import LLM, SamplingParams

    rows = []
    for label, path in (("base", MODEL), ("adapter", MERGED)):
        if not Path(path).exists():
            print(f"SKIP {label}: {path} missing", flush=True)
            continue
        print(f"\n=== {label} ===", flush=True)
        llm = LLM(
            model=path,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
            disable_log_stats=True,
        )
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        for name, prompt, _ in PROMPTS:
            messages = [{"role": "user", "content": prompt}]
            rendered = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            o = llm.generate([rendered], SamplingParams(max_tokens=MAX_TOKENS, temperature=0))
            text = o[0].outputs[0].text
            n = len(o[0].outputs[0].token_ids)
            rows.append({"model": label, "probe": name, "tokens": n, "text": text})
            print(f"  {name:16s} {n:4d} tok  {text[-160:]!r}", flush=True)
        del llm

    Path(out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
