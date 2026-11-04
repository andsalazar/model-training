"""MATH-500 evaluation through vLLM.

vLLM ships its own attention and linear-attention kernels, which is the point:
the transformers path measured 36 tok/s because causal_conv1d fell back to
reference PyTorch and pinned a CPU core. The paper reports ~11,000 tok/s.

Protocol matches the paper exactly (see reproduce_paper.py): temperature 0.6,
top_p 0.95, top_k 20, presence_penalty 1.5, 32,768 max tokens, thinking on,
4 samples per problem, the paper's own 25-problem practice subset.

Usage:
    python run_vllm.py --model Qwen/Qwen3.5-0.8B --out base.json
    python run_vllm.py --model RikiyaT/Qwen3.5-0.8B-weight-space-programming --out edited.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/src")

from wsp.eval.math500 import build_prompt, extract_answer, is_correct

# From runs/paper/scores_base_math500.json -> summary.sampling
SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 1.5,
}
MAX_TOKENS = 32768
SAMPLES = 4
SEED = 0


def load_practice_ids(limit: int = 0) -> list[str]:
    ids = json.loads(Path("/workspace/practice_ids.json").read_text(encoding="utf-8"))
    return ids[:limit] if limit else ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--out", default="vllm_results.json")
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--limit", type=int, default=0, help="0 = all practice problems")
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    practice_ids = load_practice_ids(args.limit)
    print(f"model:    {args.model}")
    print(f"problems: {len(practice_ids)} x {args.samples} samples")
    print(f"max_tokens: {args.max_tokens}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    id_key = "unique_id" if "unique_id" in dataset.column_names else None
    if id_key:
        by_id = {row[id_key]: row for row in dataset}
    else:
        print("WARNING: no unique_id column; falling back to index order", flush=True)
        by_id = {str(i): row for i, row in enumerate(dataset)}

    missing = [p for p in practice_ids if p not in by_id]
    if missing:
        print(f"WARNING: {len(missing)} ids missing, e.g. {missing[:3]}", flush=True)
        practice_ids = [p for p in practice_ids if p in by_id]

    # Build every prompt up front so vLLM can batch them in one pass.
    prompts: list[str] = []
    owners: list[tuple[str, int]] = []
    for pid in practice_ids:
        row = by_id[pid]
        messages = [{"role": "user", "content": build_prompt(row["problem"])}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for s in range(args.samples):
            prompts.append(text)
            owners.append((pid, s))

    print(f"loading vLLM ({len(prompts)} prompts)", flush=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_tokens + 4096,
        trust_remote_code=True,
        seed=SEED,
    )

    sampling = SamplingParams(
        n=1,
        max_tokens=args.max_tokens,
        seed=SEED,
        **SAMPLING,
    )

    started = time.time()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.time() - started

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(
        f"generated {total_tokens:,} tokens in {elapsed:.1f}s "
        f"({total_tokens / elapsed:,.0f} tok/s)",
        flush=True,
    )

    per_problem: dict[str, list[int]] = {pid: [] for pid in practice_ids}
    details: list[dict] = []
    cut = 0
    correct_total = 0

    for (pid, s), out in zip(owners, outputs):
        row = by_id[pid]
        text = out.outputs[0].text
        n_tokens = len(out.outputs[0].token_ids)
        hit_cap = n_tokens >= args.max_tokens
        cut += int(hit_cap)

        prediction = extract_answer(text)
        ok = is_correct(prediction, row["answer"])
        correct_total += int(ok)
        per_problem[pid].append(int(ok))

        details.append(
            {
                "id": pid,
                "sample": s,
                "gold": row["answer"],
                "prediction": prediction,
                "correct": ok,
                "hit_token_cap": hit_cap,
                "tokens": n_tokens,
                "response_tail": text[-600:],
            }
        )

    n = len(outputs)
    summary = {
        "acc": round(correct_total / n, 4),
        "cut_share": round(cut / n, 4),
        "mean_tokens": round(total_tokens / n, 1),
        "n_problems": len(practice_ids),
        "samples": args.samples,
        "seconds": round(elapsed, 1),
        "tok_per_s": round(total_tokens / elapsed, 1),
        "sampling": {**SAMPLING, "max_tokens": args.max_tokens, "seed": SEED},
        "thinking": True,
        "ckpt": args.model,
    }

    Path(args.out).write_text(
        json.dumps(
            {
                "summary": summary,
                "per_problem": [
                    {"id": pid, "correct": v} for pid, v in per_problem.items()
                ],
                "details": details,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
