"""Does the target-only adapter survive greedy decoding?

Stage 4's decode check showed the adapter's effect was larger without the presence
penalty, but greedy still hit the cap on guards. That check predates 4c/4d. This
repeats it for the target-only adapter, which is the one that works.

Greedy is the strict case: no sampling means no chance of taking a stop branch by
luck. If the adapter stops under greedy, the effect is structural.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

MODEL = "/local/models/qwen"
ADAPTER = "/workspace/s4d_adapter"
MERGED = "/local/models/qwen_s4d_merged"
MAX_TOKENS = 2048

PROBES = [
    ("arith_12x11", "What is 12 times 11? Reply with only the digits.", "132"),
    ("arith_17x23", "What is 17 times 23? Reply with only the digits.", "391"),
    ("logic_syllogism", "All cats are mammals. All mammals are animals. Are all cats animals? Reply with only yes or no.", "yes"),
    ("heldout_capital", "Name the capital of Japan. Reply with only the city.", "Tokyo"),
    ("guard_poem", "Write two lines of a poem about the sea.", "sea"),
    ("guard_hammer", "List three uses of a hammer, one per line.", "nail"),
    ("guard_blue_sky", "Explain in two sentences why the sky appears blue.", "light"),
]

CONFIGS = [
    ("greedy", 0.0, 0.0),
    ("sampled", 0.6, 1.5),
]


def merge_adapter() -> Path:
    """Merge the LoRA into a standalone checkpoint so vLLM can load it."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(MERGED)
    if (out / "config.json").exists():
        return out
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype="auto", device_map="cpu"
    )
    merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
    merged.save_pretrained(out)
    AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True).save_pretrained(out)
    return out


def main() -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    merged = merge_adapter()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    llm = LLM(
        model=str(merged), max_model_len=4096, gpu_memory_utilization=0.85,
        enforce_eager=True, disable_log_stats=True,
    )

    rows = []
    for label, temp, pp in CONFIGS:
        params = SamplingParams(
            max_tokens=MAX_TOKENS, temperature=temp, presence_penalty=pp
        )
        for name, prompt, expected in PROBES:
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            o = llm.generate([rendered], params)[0]
            text = o.outputs[0].text
            n = len(o.outputs[0].token_ids)
            row = {
                "config": label, "probe": name, "tokens": n,
                "contains": expected.lower() in text.lower(),
                "hit_cap": n >= MAX_TOKENS, "text": text,
            }
            rows.append(row)
            print(f"  {label:8s} {name:16s} {n:5d} tok "
                  f"cap={str(row['hit_cap']):5s} ok={row['contains']}", flush=True)
        del params

    out = "/workspace/decode_4d.json"
    Path(out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}", flush=True)
    print("DECODE4D_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
