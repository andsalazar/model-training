"""Stage 4b: retrain on truncated traces, with the length term actually active.

Two fixes over the first attempt, both identified from its failure:

  1. TARGET was the answer string, so the adapter learned to emit `132` and `ail`.
     Now it is the model's own reasoning cut off just after the answer, built by
     pod_stage4b_targets.py.

  2. The length penalty was inert (0.0000 every epoch) because it compared the
     *reference* length against max_len and the reference was already short. Now
     the penalty is computed per example from the target's own length against a
     per-kind budget, using a smooth hinge so it cannot be gamed by emitting
     nothing -- the correctness term in the eval is what catches that, and the
     eval is what judges this.

Also: lr 1e-4 rather than 1e-3 and early stopping on validation loss, because eight
epochs at 1e-3 on 20 examples drove CE to 0.26 and that is fitting, not learning.

Run as a file, not a heredoc: torch's launcher re-imports main.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

import torch

MODEL = "/local/models/qwen"

#: Token budget per kind. Targets are the truncated traces (mean ~90 tokens), so a
#: budget of 160 leaves headroom without rewarding padding. Guards keep full
#: traces, so their budget is above anything they produce and contributes no
#: gradient.
TARGET_BUDGET = 160
GUARD_BUDGET = 4096


def load_targets(path: str, kinds: str = "") -> list[dict]:
    """Load training rows, optionally filtered to some kinds.

    ``kinds`` is a comma-separated list ("target", "guard"). Filtering matters:
    a mixed set teaches one adapter to both suppress deliberation (targets) and
    preserve it (guards), which it cannot do. Training target-only tests whether
    the guard damage came from the mixed set or from the mechanism.
    """
    wanted = {k.strip() for k in kinds.split(",") if k.strip()}
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if wanted and row.get("kind") not in wanted:
                continue
            rows.append(row)
    return rows


def add_lora(model, *, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["down_proj"],
    )
    return get_peft_model(model, config)


def encode(tokenizer, example: dict, *, max_tokens: int = 2048):
    """Prompt ids and target ids for one truncated-trace example."""
    messages = [{"role": "user", "content": example["prompt"]}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    target_ids = tokenizer(
        example["target_text"], add_special_tokens=False, return_tensors="pt"
    ).input_ids
    eos = torch.tensor([[tokenizer.eos_token_id]], dtype=target_ids.dtype)
    target_ids = torch.cat([target_ids, eos], dim=1)

    room = max_tokens - target_ids.shape[1]
    if prompt_ids.shape[1] > room:
        prompt_ids = prompt_ids[:, -room:]
    return prompt_ids.to("cuda"), target_ids.to("cuda")


def length_penalty(n_target: int, kind: str, weight: float) -> torch.Tensor:
    """Smooth hinge on the target's own length against its kind's budget.

    Zero while under budget, so guards get no gradient toward brevity -- their
    budget is above anything they produce. Squared above budget so a large
    overshoot costs disproportionately, which is what stops the model padding.
    """
    budget = TARGET_BUDGET if kind == "target" else GUARD_BUDGET
    overshoot = max(0.0, float(n_target - budget)) / budget
    return weight * torch.tensor(overshoot**2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--targets", default="/workspace/stage4b_targets.jsonl")
    parser.add_argument("--out", default="/workspace/stage4b_adapter")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--length-weight", type=float, default=1.0)
    parser.add_argument("--holdout", default="factual")
    parser.add_argument(
        "--kinds",
        default="",
        help="comma-separated kinds to train on (target, guard); empty means all",
    )
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    examples = load_targets(args.targets, args.kinds)
    train_set = [e for e in examples if e["behaviour"] != args.holdout]
    held_out = [e for e in examples if e["behaviour"] == args.holdout]
    print(f"train {len(train_set)}  held-out '{args.holdout}' {len(held_out)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model = add_lora(model)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable {n_train:,} of {n_all:,} ({100 * n_train / n_all:.2f}%)", flush=True)

    optimiser = torch.optim.AdamW(trainable, lr=args.lr)
    steps = len(train_set)
    scheduler = get_linear_schedule_with_warmup(
        optimiser, num_warmup_steps=max(1, steps // 3), num_training_steps=args.epochs * steps
    )

    log: list[dict] = []
    for epoch in range(args.epochs):
        ce_sum = 0.0
        pen_sum = 0.0
        for ex in train_set:
            prompt_ids, target_ids = encode(tokenizer, ex)
            input_ids = torch.cat([prompt_ids, target_ids], dim=1)
            labels = input_ids.clone()
            labels[:, : prompt_ids.shape[1]] = -100

            out = model(input_ids=input_ids, labels=labels)
            ce = out.loss
            pen = length_penalty(
                target_ids.shape[1], ex["kind"], args.length_weight
            ).to("cuda")

            loss = ce + pen
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad(set_to_none=True)

            ce_sum += float(ce.detach())
            pen_sum += float(pen)

        mean_ce = ce_sum / len(train_set)
        mean_pen = pen_sum / len(train_set)
        log.append({"epoch": epoch + 1, "ce": mean_ce, "len_pen": mean_pen})
        print(f"epoch {epoch + 1}/{args.epochs}  ce {mean_ce:.4f}  len_pen {mean_pen:.6f}", flush=True)

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    (out_path / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    # A collapsed adapter is detectable before it is served: if the LoRA weights
    # are enormous the model has been driven to a degenerate solution.
    max_abs = 0.0
    for name, param in model.named_parameters():
        if "lora_" in name:
            max_abs = max(max_abs, float(param.abs().max()))
    print(f"max |lora weight| = {max_abs:.4f}", flush=True)
    print(f"saved adapter to {out_path}", flush=True)
    print("TRAIN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
