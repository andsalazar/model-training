"""Probe the model on the pod before committing to a full eval run.

Confirms the checkpoint loads, the chat template works, and generation
terminates, on a handful of problems.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/src")

from wsp.eval.math500 import build_prompt, extract_answer, is_correct

MODEL = "Qwen/Qwen3.5-0.8B"


def main() -> int:
    print(f"loading {MODEL}", flush=True)
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    print(f"loaded in {time.time() - started:.1f}s on {model.device}", flush=True)

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(5))

    for i, row in enumerate(dataset):
        messages = [{"role": "user", "content": build_prompt(row["problem"])}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        generated = output[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        prediction = extract_answer(response)
        ok = is_correct(prediction, row["answer"])

        print(f"\n--- problem {i} ({elapsed:.1f}s, {len(generated)} tokens) ---")
        print(f"gold:       {row['answer']}")
        print(f"prediction: {prediction}")
        print(f"correct:    {ok}")
        print(f"response:   {response[:300]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
