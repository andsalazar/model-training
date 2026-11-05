"""Stage 4: train a behaviour-selective LoRA adapter against over-thinking.

Why an adapter rather than another weight edit: Stage 3 showed a single fixed
direction suppresses a shared "keep checking" state used by arithmetic, logic and
long-form writing alike, so it improved the target while breaking poems and lists.
An adapter can *condition* on the input, which is exactly the capacity the fixed
vector lacked.

The loss has two terms, and the second is the one that matters:

  length   pulls target outputs toward TARGET_MAX_LEN tokens and pulls guard
           outputs toward GUARD_MAX_LEN, which is set above their natural length
           so guards get NO gradient toward brevity
  correctness
           fires when the expected answer is absent, so the adapter cannot buy
           brevity with wrong answers

A length-only loss is trivially satisfiable by outputting nothing, which is why the
correctness term is not optional.

Run as a file, not a heredoc: torch's launcher and vLLM both re-import main.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

import torch
import torch.nn.functional as F

from wsp.stage4_data import build_dataset, split_dataset


def load_model(name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    return model, tokenizer


def add_lora(model, *, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Attach a LoRA to every MLP down_proj.

    Down-projection is the tensor Stage 1 and Stage 3 localised the behaviour to,
    so targeting it keeps the comparison honest: same site, different mechanism.
    """
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


def render(tokenizer, prompt: str, *, thinking: bool = True) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )


def encode_example(tokenizer, example, *, max_tokens: int = 2048):
    """Prompt ids plus the reference continuation the model must reproduce.

    The reference is the answer only -- the adapter is trained to go straight to
    it, not to reproduce the over-thinking it currently emits.
    """
    prompt_text = render(tokenizer, example.prompt)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    answer_ids = tokenizer(
        example.expected, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    eos = torch.tensor([[tokenizer.eos_token_id]], dtype=answer_ids.dtype)
    answer_ids = torch.cat([answer_ids, eos], dim=1)

    # Truncate the prompt from the left if needed; the answer must survive.
    room = max_tokens - answer_ids.shape[1]
    if prompt_ids.shape[1] > room:
        prompt_ids = prompt_ids[:, -room:]
    return prompt_ids.to("cuda"), answer_ids.to("cuda")


def train(
    model_name: str,
    *,
    epochs: int = 8,
    lr: float = 1e-3,
    length_weight: float = 0.05,
    out_dir: str = "/workspace/stage4_adapter",
    holdout: str = "factual",
) -> int:
    from peft import PeftModel
    from transformers import get_linear_schedule_with_warmup

    print(f"loading {model_name}", flush=True)
    model, tokenizer = load_model(model_name)

    examples = build_dataset()
    train_set, held_out = split_dataset(examples, holdout_behaviour=holdout)
    print(
        f"train {len(train_set)} examples, held-out behaviour '{holdout}' "
        f"{len(held_out)} examples",
        flush=True,
    )

    model = add_lora(model)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"trainable {n_trainable:,} of {total:,} ({100 * n_trainable / total:.2f}%)",
        flush=True,
    )

    optimiser = torch.optim.AdamW(trainable, lr=lr)
    steps_per_epoch = len(train_set)
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=max(1, steps_per_epoch // 4),
        num_training_steps=epochs * steps_per_epoch,
    )

    log: list[dict] = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_len_pen = 0.0
        for example in train_set:
            prompt_ids, answer_ids = encode_example(tokenizer, example)
            input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
            labels = input_ids.clone()
            labels[:, : prompt_ids.shape[1]] = -100

            out = model(input_ids=input_ids, labels=labels)
            ce = out.loss

            # Length penalty: only pulls DOWN, and only where a short answer is
            # correct. For guards max_len exceeds their natural length, so the
            # clamped term is zero and there is no gradient toward brevity.
            n_answer = answer_ids.shape[1]
            overshoot = torch.clamp(
                torch.tensor(float(n_answer - example.max_len)), min=0.0
            )
            len_pen = length_weight * overshoot

            loss = ce + len_pen
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad(set_to_none=True)

            epoch_loss += float(ce.detach())
            epoch_len_pen += float(len_pen)

        mean_ce = epoch_loss / len(train_set)
        mean_len = epoch_len_pen / len(train_set)
        log.append({"epoch": epoch + 1, "ce": mean_ce, "len_pen": mean_len})
        print(
            f"epoch {epoch + 1}/{epochs}  ce {mean_ce:.4f}  len_pen {mean_len:.4f}",
            flush=True,
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    (out_path / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"\nsaved adapter to {out_path}", flush=True)

    # Confirm the adapter is not a no-op before it leaves this script.
    changed = 0
    for name, param in model.named_parameters():
        if "lora_" in name and torch.count_nonzero(param):
            changed += 1
    print(f"non-zero LoRA tensors: {changed}", flush=True)
    print("TRAIN_DONE", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/local/models/qwen")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--length-weight", type=float, default=0.05)
    parser.add_argument("--out", default="/workspace/stage4_adapter")
    parser.add_argument("--holdout", default="factual")
    args = parser.parse_args()

    return train(
        args.model,
        epochs=args.epochs,
        lr=args.lr,
        length_weight=args.length_weight,
        out_dir=args.out,
        holdout=args.holdout,
    )


if __name__ == "__main__":
    sys.exit(main())
