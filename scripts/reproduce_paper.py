"""Reproduce the paper's Experiment 8 protocol exactly.

Protocol, taken from the released score files rather than inferred:

  sampling:  temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5
  max_tokens: 32768
  thinking:  on
  samples:   4 per problem
  problems:  the paper's own 25-problem practice subset, by MATH-500 id

The token budget is the load-bearing detail. At 32,768 tokens the base model
runs to the limit on 95% of answers; at a smaller cap it is cut off before the
loop can express itself, which is why a 4,096-token run scores far higher than
the paper's 7.9%.

Usage:
    python reproduce_paper.py --model Qwen/Qwen3.5-0.8B --out base.json
    python reproduce_paper.py --model RikiyaT/Qwen3.5-0.8B-weight-space-programming --out edited.json
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

sys.path.insert(0, "/workspace/src")

from wsp.eval.math500 import build_prompt, extract_answer, is_correct

# From runs/paper/scores_base_math500.json -> summary.sampling
SAMPLING = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "repetition_penalty": 1.0,
}
MAX_TOKENS = 32768
SAMPLES = 4
SEED = 0


def load_practice_ids() -> list[str]:
    path = Path("/workspace/practice_ids.json")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--out", default="repro.json")
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--limit", type=int, default=0, help="0 = all practice problems")
    args = parser.parse_args()

    torch.manual_seed(SEED)

    print(f"loading {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    print(f"loaded on {model.device}", flush=True)

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    by_id = {row["unique_id"] if "unique_id" in row else str(i): row
             for i, row in enumerate(dataset)}

    practice_ids = load_practice_ids()
    if args.limit:
        practice_ids = practice_ids[: args.limit]

    # MATH-500 rows carry a unique_id matching the paper's "test/<subject>/<n>.json".
    id_key = "unique_id" if "unique_id" in dataset.column_names else None
    if id_key:
        by_id = {row[id_key]: row for row in dataset}
    else:
        print("WARNING: no unique_id column; falling back to index order", flush=True)
        by_id = {str(i): row for i, row in enumerate(dataset)}

    missing = [p for p in practice_ids if p not in by_id]
    if missing:
        print(f"WARNING: {len(missing)} ids not found, e.g. {missing[:3]}", flush=True)
        practice_ids = [p for p in practice_ids if p in by_id]

    print(f"evaluating {len(practice_ids)} problems x {args.samples} samples", flush=True)

    per_problem = []
    total_correct = 0
    total_samples = 0
    total_tokens = 0
    cut = 0
    started = time.time()

    for pi, pid in enumerate(practice_ids):
        row = by_id[pid]
        messages = [{"role": "user", "content": build_prompt(row["problem"])}]
        # The template defaults to thinking OFF (it emits an immediately closed
        #  thinking block). The paper evaluates with thinking on, so this must be
        # set explicitly or the run measures a different configuration.
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        correct: list[int] = []
        for s in range(args.samples):
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    **SAMPLING,
                )
            generated = output[0][inputs["input_ids"].shape[1] :]
            n_tokens = int(len(generated))
            total_tokens += n_tokens
            if n_tokens >= args.max_tokens:
                cut += 1

            response = tokenizer.decode(generated, skip_special_tokens=True)
            prediction = extract_answer(response)
            ok = is_correct(prediction, row["answer"])
            correct.append(int(ok))
            total_correct += int(ok)
            total_samples += 1

        per_problem.append({"id": pid, "correct": correct})
        acc = total_correct / total_samples
        elapsed = time.time() - started
        print(
            f"[{pi + 1}/{len(practice_ids)}] {pid:40s} "
            f"acc={acc:.3f} cut={cut / total_samples:.2f} "
            f"({elapsed / 60:.1f}m)",
            flush=True,
        )

        Path(args.out).write_text(
            json.dumps(
                {
                    "summary": {
                        "acc": round(total_correct / total_samples, 4),
                        "cut_share": round(cut / total_samples, 4),
                        "mean_tokens": round(total_tokens / total_samples, 1),
                        "n_problems": len(per_problem),
                        "samples": args.samples,
                        "seconds": round(time.time() - started, 1),
                        "sampling": {**SAMPLING, "max_tokens": args.max_tokens, "seed": SEED},
                        "thinking": True,
                        "ckpt": args.model,
                    },
                    "per_problem": per_problem,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"\nFINAL acc={total_correct / total_samples:.4f} "
          f"cut={cut / total_samples:.4f} "
          f"mean_tokens={total_tokens / total_samples:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
