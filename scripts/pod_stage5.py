"""Stage 5: generalise the adapter recipe to format compliance.

One script, three phases, because Stage 4 showed the phases are coupled: the
targets must come from the model's own traces, and the eval must be able to see
that the adapter did not just learn to emit the answer string.

  targets  generate the base model's own response per prompt, cut it off just
           after the expected answer appears, keep guards whole
  train    LoRA on down_proj against those truncated traces
  eval     broad probe set, harmful-before-validated, plus the ablation

The eval is deliberately adversarial to a false positive. Stage 4's first adapter
looked fine on a substring test and was actually emitting `ail`. So a target only
counts as compliant here if the answer appears AND the output is under a length
cap, and the guards must still contain real content.

Run as a file, not a heredoc: torch and vLLM both re-import main.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

MODEL = "/local/models/qwen"
MAX_TOKENS = 2048

#: A format-compliant answer should be short; anything past this is the model
#: arguing with itself about the format rather than producing it.
COMPLIANT_CAP = 120

#: Reasoning markers that mean the model is still deliberating rather than
#: answering. A compliant format answer should not contain any of them.
_REASONING_MARKERS = (
    "thinking",
    "analyze",
    "wait",
    "let me",
    "constraint",
    "**",
    "okay",
    "hmm",
)


def _compliant(text: str, expected: str, n_tokens: int) -> bool:
    """Whether an output actually satisfies a format instruction.

    A substring test is not enough here. The expected strings are single
    characters ("1", " ", "/", "-"), so "Thinking Process:\\n\\n1" contains "1"
    and would pass while being truncated deliberation. Compliance requires the
    answer to be present, the output to be short, and no reasoning markers.

    The marker list is deliberately broad. A false negative here costs one
    missed success; a false positive invents one, which is the failure this
    project has hit repeatedly.
    """
    if n_tokens > COMPLIANT_CAP:
        return False
    lowered = text.lower()
    if expected.lower() not in lowered:
        return False
    return not any(marker in lowered for marker in _REASONING_MARKERS)


def render(tok, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )


def phase_targets(model: str, out: str) -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from wsp.stage5_data import build

    rows = build()
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = LLM(
        model=model, max_model_len=4096, gpu_memory_utilization=0.85,
        enforce_eager=True, disable_log_stats=True,
    )

    built = []
    for ex in rows:
        o = llm.generate(
            [render(tok, ex.prompt)], SamplingParams(max_tokens=MAX_TOKENS, temperature=0)
        )[0]
        text = o.outputs[0].text
        n_full = len(o.outputs[0].token_ids)

        if ex.kind == "guard":
            target_text, ok = text, True
        else:
            idx = text.lower().find(ex.expected.lower())
            ok = idx >= 0
            target_text = text[: idx + len(ex.expected)] if ok else text

        if not ok:
            print(f"  SKIP (answer absent): {ex.prompt[:50]!r}", flush=True)
            continue

        n_cut = len(tok(target_text, add_special_tokens=False).input_ids)
        built.append({**ex.as_dict(), "target_text": target_text,
                      "full_tokens": n_full, "cut_tokens": n_cut})
        print(f"  {ex.kind:6s} {n_full:5d} -> {n_cut:5d}  {target_text[-50:]!r}",
              flush=True)

    Path(out).write_text(
        "\n".join(json.dumps(r) for r in built) + "\n", encoding="utf-8"
    )
    tg = [r for r in built if r["kind"] == "target"]
    if tg:
        mean = sum(r["cut_tokens"] for r in tg) / len(tg)
        print(f"\ntargets {len(tg)} mean cut {mean:.0f} tok", flush=True)
    print(f"written: {out}", flush=True)
    print("TARGETS_DONE", flush=True)
    return 0


def phase_train(model: str, targets: str, out: str, epochs: int, lr: float) -> int:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    rows = [json.loads(l) for l in Path(targets).read_text(encoding="utf-8").splitlines() if l.strip()]
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    m = get_peft_model(
        base,
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                   task_type="CAUSAL_LM", target_modules=["down_proj"]),
    )
    m.train()
    trainable = [p for p in m.parameters() if p.requires_grad]

    opt = torch.optim.AdamW(trainable, lr=lr)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=max(1, len(rows) // 3),
        num_training_steps=epochs * len(rows),
    )

    for epoch in range(epochs):
        total = 0.0
        for r in rows:
            p_ids = tok(render(tok, r["prompt"]), return_tensors="pt").input_ids
            t_ids = tok(r["target_text"], add_special_tokens=False,
                        return_tensors="pt").input_ids
            t_ids = torch.cat(
                [t_ids, torch.tensor([[tok.eos_token_id]], dtype=t_ids.dtype)], dim=1
            )
            room = 2048 - t_ids.shape[1]
            if p_ids.shape[1] > room:
                p_ids = p_ids[:, -room:]
            ids = torch.cat([p_ids, t_ids], dim=1).to("cuda")
            labels = ids.clone()
            labels[:, : p_ids.shape[1]] = -100

            loss = m(input_ids=ids, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            total += float(loss.detach())
        print(f"epoch {epoch + 1}/{epochs} loss {total / len(rows):.4f}", flush=True)

    p = Path(out)
    p.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(str(p))
    tok.save_pretrained(str(p))
    mx = max(float(x.abs().max()) for n, x in m.named_parameters() if "lora_" in n)
    print(f"max |lora| {mx:.4f}", flush=True)
    print("TRAIN_DONE", flush=True)
    return 0


def phase_eval(model: str, adapter: str, out: str) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    from wsp.stage5_data import build

    rows = build()
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    # Merge so vLLM can serve it, and so the number reflects a shippable model.
    base = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    merged_dir = Path("/local/models/qwen_stage5_merged")
    merged_dir.mkdir(parents=True, exist_ok=True)
    PeftModel.from_pretrained(base, adapter).merge_and_unload().save_pretrained(
        str(merged_dir), safe_serialization=True
    )
    tok.save_pretrained(str(merged_dir))
    del base

    params = SamplingParams(
        max_tokens=MAX_TOKENS, temperature=0.6, top_p=0.95, top_k=20,
        presence_penalty=1.5, seed=0,
    )

    def run(path: str, label: str) -> list[dict]:
        llm = LLM(model=path, max_model_len=4096, gpu_memory_utilization=0.85,
                  enforce_eager=True, disable_log_stats=True)
        got = []
        for ex in rows:
            o = llm.generate([render(tok, ex.prompt)], params)[0]
            text = o.outputs[0].text
            n = len(o.outputs[0].token_ids)
            row = {
                "probe": ex.prompt[:44],
                "kind": ex.kind,
                "tokens": n,
                "contains": ex.expected.lower() in text.lower(),
                "compliant": _compliant(text, ex.expected, n),
                "text": text[-200:],
            }
            got.append(row)
            print(f"  {label:8s} {ex.kind:6s} {n:5d} tok "
                  f"contains={str(row['contains']):5s} compliant={row['compliant']}",
                  flush=True)
        del llm
        return got

    base_rows = run(model, "base")
    merged_rows = run(str(merged_dir), "adapter")

    print("\n=== comparison ===", flush=True)
    verdicts = []
    for b, a in zip(base_rows, merged_rows):
        if b["contains"] and not a["contains"]:
            state = "broken"
        elif a["tokens"] >= MAX_TOKENS and b["tokens"] < MAX_TOKENS:
            state = "regressed"
        elif a["tokens"] / max(b["tokens"], 1) <= 0.75:
            state = "improved"
        elif a["tokens"] / max(b["tokens"], 1) >= 1.25:
            state = "regressed"
        else:
            state = "unchanged"
        verdicts.append({"probe": b["probe"], "kind": b["kind"],
                         "base": b["tokens"], "adapter": a["tokens"],
                         "base_compliant": b["compliant"],
                         "adapter_compliant": a["compliant"], "state": state})
        print(f"  {b['kind']:6s} {b['tokens']:5d} -> {a['tokens']:5d}  {state}  "
              f"compliant {b['compliant']} -> {a['compliant']}", flush=True)

    harmful = [v for v in verdicts if v["state"] in ("regressed", "broken")]
    guard_harm = [v for v in harmful if v["kind"] == "guard"]
    tgt_ok = [v for v in verdicts if v["kind"] == "target" and v["adapter_compliant"]
              and not v["base_compliant"]]
    tgt_broke = [v for v in verdicts if v["kind"] == "target"
                 and v["base_compliant"] and not v["adapter_compliant"]]

    status = "harmful" if harmful else ("validated" if tgt_ok else "no-effect")
    print(f"\nstatus: {status}", flush=True)
    if guard_harm:
        print(f"GUARD HARM: {len(guard_harm)}", flush=True)
    print(f"targets newly compliant: {len(tgt_ok)}", flush=True)
    print(f"targets broken: {len(tgt_broke)}", flush=True)

    Path(out).write_text(json.dumps(
        {"status": status, "verdicts": verdicts,
         "guard_harm": len(guard_harm), "newly_compliant": len(tgt_ok),
         "broken_targets": len(tgt_broke),
         "base": base_rows, "adapter": merged_rows}, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}", flush=True)
    print("EVAL_DONE", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["targets", "train", "eval"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--targets", default="/workspace/stage5_targets.jsonl")
    ap.add_argument("--out", default="/workspace/stage5_out")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()

    if a.phase == "targets":
        return phase_targets(a.model, a.targets)
    if a.phase == "train":
        return phase_train(a.model, a.targets, a.out, a.epochs, a.lr)
    return phase_eval(a.model, a.out, a.out + "_eval.json")


if __name__ == "__main__":
    sys.exit(main())
