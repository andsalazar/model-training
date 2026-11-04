"""Run MATH-500 against a checkpoint. Executes on the GPU pod.

Usage:
    python run_math500.py --model Qwen/Qwen3.5-0.8B --limit 500 --out results.json

Writes incremental results so a long run can be inspected while it is going.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# The script may sit beside src/ (local) or inside it (pod), so search upward.
_here = Path(__file__).resolve().parent
for _candidate in (_here / "src", _here.parent / "src", _here):
    if (_candidate / "wsp").is_dir():
        sys.path.insert(0, str(_candidate))
        break

from wsp.eval.math500 import EvalResult, build_prompt, extract_answer, is_correct


def load_model(model_id: str, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    generated = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args()

    print(f"loading {args.model}", flush=True)
    model, tokenizer = load_model(args.model, torch.bfloat16)
    print(f"loaded on {model.device}", flush=True)

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"evaluating {len(dataset)} problems", flush=True)

    samples: list[dict] = []
    correct = 0
    started = time.time()

    for i, row in enumerate(dataset):
        question = row["problem"]
        gold = row["answer"]
        response = generate(
            model, tokenizer, build_prompt(question), args.max_new_tokens
        )
        prediction = extract_answer(response)
        ok = is_correct(prediction, gold)
        correct += int(ok)

        samples.append(
            {
                "index": i,
                "question": question[:200],
                "gold": gold,
                "prediction": prediction,
                "correct": ok,
                "response": response[:2000],
            }
        )

        if (i + 1) % 10 == 0:
            elapsed = time.time() - started
            rate = (i + 1) / elapsed
            eta = (len(dataset) - i - 1) / rate if rate else 0
            print(
                f"[{i + 1}/{len(dataset)}] acc={correct / (i + 1):.3f} "
                f"({rate:.2f}/s, eta {eta / 60:.1f}m)",
                flush=True,
            )

        if (i + 1) % args.save_every == 0:
            _write(args.out, EvalResult(correct, i + 1, samples), args.model)

    result = EvalResult(correct, len(dataset), samples)
    _write(args.out, result, args.model)
    print(f"\nFINAL: {result.as_dict()}", flush=True)
    return 0


def _write(path: str, result: EvalResult, model_id: str) -> None:
    payload = {
        "model": model_id,
        **result.as_dict(),
        "samples": result.samples,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
