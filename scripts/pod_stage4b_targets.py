"""Stage 4b: build training targets from the model's OWN truncated reasoning.

The first Stage 4 attempt trained on the answer string alone and the adapter
learned to emit just that string -- `132`, `391`, `sea`, and `ail` for a guard
whose expected answer was `nail`. It memorised the answer token and never saw a
behaviour, so it could not learn one.

This module fixes the target. For each prompt it:

  1. generates the base model's own full response (the over-thinking trace)
  2. finds where the answer appears
  3. keeps the trace up to the first few tokens AFTER the answer, and drops the
     rest

The reference continuation is then "the model's own reasoning, cut short right
after it reaches the answer". That is the closest textual encoding of "do what you
did, but stop sooner", which is the behaviour the weight-editing stages were
trying to produce.

Guards are handled differently on purpose: their trace is kept in full, so the
adapter has examples where long output is correct. Without them, any brevity
signal generalises to everything.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, "/workspace/src")

from wsp.stage4_data import build_dataset

MODEL = "/local/models/qwen"
MAX_TOKENS = 2048

#: How many tokens of the trace to keep after the answer first appears. A few is
#: enough to finish the sentence; much more and the self-verification is back.
TAIL_TOKENS = 8


@dataclass
class TraceTarget:
    prompt: str
    expected: str
    kind: str
    behaviour: str
    full_tokens: int
    cut_tokens: int
    target_text: str

    def as_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "expected": self.expected,
            "kind": self.kind,
            "behaviour": self.behaviour,
            "full_tokens": self.full_tokens,
            "cut_tokens": self.cut_tokens,
            "target_text": self.target_text,
            "reduction": self.full_tokens - self.cut_tokens,
        }


def cut_after_answer(text: str, answer: str) -> tuple[str, bool]:
    """Keep the trace up to just past the answer, dropping the rest.

    Returns ``(target, found)``. When the answer never appears the trace is
    returned unchanged and ``found`` is False, so the caller can exclude it rather
    than train on a target that does not contain the answer.
    """
    idx = text.lower().find(answer.lower())
    if idx < 0:
        return text, False
    end = idx + len(answer)
    return text[:end], True


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/stage4b_targets.jsonl"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    examples = build_dataset()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    llm = LLM(
        model=MODEL,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        disable_log_stats=True,
    )

    rows: list[TraceTarget] = []
    for ex in examples:
        messages = [{"role": "user", "content": ex.prompt}]
        rendered = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        o = llm.generate([rendered], SamplingParams(max_tokens=MAX_TOKENS, temperature=0))
        text = o[0].outputs[0].text
        n_full = len(o[0].outputs[0].token_ids)

        if ex.kind == "guard":
            # Guards keep their full trace: this is where long output is correct.
            target = text
            cut = n_full
            ok = True
        else:
            target, ok = cut_after_answer(text, ex.expected)
            cut = len(tok(target, add_special_tokens=False).input_ids)

        if not ok:
            print(f"  SKIP (answer absent): {ex.prompt[:50]!r}", flush=True)
            continue

        rows.append(
            TraceTarget(
                prompt=ex.prompt,
                expected=ex.expected,
                kind=ex.kind,
                behaviour=ex.behaviour,
                full_tokens=n_full,
                cut_tokens=cut,
                target_text=target,
            )
        )
        print(
            f"  {ex.kind:6s} {ex.behaviour:12s} {n_full:5d} -> {cut:5d} tok  "
            f"{target[-60:]!r}",
            flush=True,
        )

    path = Path(out)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.as_dict()) + "\n")

    targets = [r for r in rows if r.kind == "target"]
    guards = [r for r in rows if r.kind == "guard"]
    t_mean = (
        sum(r.cut_tokens for r in targets) / len(targets) if targets else 0.0
    )
    g_mean = sum(r.cut_tokens for r in guards) / len(guards) if guards else 0.0
    print(f"\ntargets: {len(targets)} mean target {t_mean:.0f} tok", flush=True)
    print(f"guards : {len(guards)} mean target {g_mean:.0f} tok", flush=True)
    if targets:
        reduction = sum(r.full_tokens - r.cut_tokens for r in targets) / len(targets)
        print(f"target reduction: {reduction:.0f} tok on average", flush=True)
    print(f"written: {path}", flush=True)
    print("TARGETS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
