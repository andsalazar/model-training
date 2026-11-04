"""Measure generation throughput on one problem, to size the real run.

The paper reports ~11,000 tok/s on its hardware. If this pod is far below that,
a full reproduction is not affordable and the bottleneck must be fixed first.
"""

from __future__ import annotations

import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/src")

from wsp.eval.math500 import build_prompt

MODEL = "Qwen/Qwen3.5-0.8B"


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    row = load_dataset("HuggingFaceH4/MATH-500", split="test")[0]
    messages = [{"role": "user", "content": build_prompt(row["problem"])}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    for n in (128, 512):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=n,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        produced = out.shape[1] - inputs["input_ids"].shape[1]
        print(
            f"max_new={n:5d}  produced={produced:5d}  {elapsed:6.1f}s  "
            f"{produced / elapsed:8.1f} tok/s",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
