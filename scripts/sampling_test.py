"""Test whether the paper's 7.9% baseline comes from sampling parameters.

Qwen3.5-0.8B's model card warns that in thinking mode, with the *recommended*
sampling parameters, the model enters thinking loops. The paper reports 7.9% on
MATH-500 "with the same sampling settings".

If greedy decoding avoids the loop, the baseline should be far higher than 7.9%
and the paper's starting point is a sampling artifact rather than a weight
property. That distinction decides what the weight edit is actually fixing.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/src")

from wsp.eval.math500 import build_prompt, extract_answer, is_correct

MODEL = "Qwen/Qwen3.5-0.8B"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30

# From the model card, thinking mode for text tasks.
RECOMMENDED = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}

GREEDY = {"do_sample": False}


def run(model, tokenizer, dataset, gen_kwargs, label: str) -> dict:
    correct = 0
    truncated = 0
    samples = []
    started = time.time()

    for i, row in enumerate(dataset):
        messages = [{"role": "user", "content": build_prompt(row["problem"])}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=4096,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                **gen_kwargs,
            )

        generated = output[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        prediction = extract_answer(response)
        ok = is_correct(prediction, row["answer"])
        hit_cap = len(generated) >= 4096

        correct += int(ok)
        truncated += int(hit_cap)
        samples.append(
            {
                "index": i,
                "gold": row["answer"],
                "prediction": prediction,
                "correct": ok,
                "hit_token_cap": hit_cap,
                "tokens": int(len(generated)),
                "response": response[:1500],
            }
        )

        if (i + 1) % 5 == 0:
            print(
                f"  [{label}] {i + 1}/{len(dataset)} acc={correct / (i + 1):.3f} "
                f"capped={truncated}",
                flush=True,
            )

    elapsed = time.time() - started
    return {
        "label": label,
        "correct": correct,
        "total": len(dataset),
        "accuracy": correct / len(dataset),
        "hit_token_cap": truncated,
        "seconds": round(elapsed, 1),
        "samples": samples,
    }


def main() -> int:
    print(f"loading {MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(N))
    print(f"running {N} problems per condition\n", flush=True)

    results = []
    for label, kwargs in (("greedy", GREEDY), ("recommended", RECOMMENDED)):
        print(f"=== {label} ===", flush=True)
        results.append(run(model, tokenizer, dataset, kwargs, label))

    print("\n=== SUMMARY ===")
    for r in results:
        print(
            f"{r['label']:12s} acc={r['accuracy']:.3f} "
            f"({r['correct']}/{r['total']})  "
            f"hit_cap={r['hit_token_cap']}  {r['seconds']:.0f}s"
        )

    Path("/workspace/sampling_test.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("\nwritten: /workspace/sampling_test.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
