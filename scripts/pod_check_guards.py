"""Do the guards still DO their task, or are they merely shorter?

The retrained adapter cut every guard's length (poem 2048->116, hammer 350->129,
blue_sky 2048->830) and the eval marked them all correct because the expected
substring appears. That is not the same as the guard being intact.

This prints the full guard outputs so the question is answered from text rather
than from a substring test. A two-line poem must be a two-line poem; three uses of
a hammer must be three uses.

Run as a file, not a heredoc: vLLM spawns a core process that re-imports main.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

MODEL = "/local/models/qwen"
ADAPTER = "/workspace/stage4b_adapter"
MAX_TOKENS = 2048

GUARDS = [
    ("guard_poem", "Write two lines of a poem about the sea."),
    ("guard_hammer", "List three uses of a hammer, one per line."),
    ("guard_blue_sky", "Explain in two sentences why the sky appears blue."),
    ("guard_capital_verbose", "Name the capital of France."),
]


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/guards.json"
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = []

    print("loading base + adapter...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
    merged.eval()

    def generate(model, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            out_ids = model.generate(
                ids,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out_ids[0][ids.shape[1] :], skip_special_tokens=True)

    for name, prompt in GUARDS:
        for label, model in (("base", None), ("adapter", merged)):
            if model is None:
                continue
            text = generate(model, prompt)
            rows.append({"model": label, "probe": name, "prompt": prompt, "text": text})
            n = len(tokenizer(text, add_special_tokens=False).input_ids)
            print(f"\n--- {label} / {name} ({n} tok) ---", flush=True)
            print(text, flush=True)

    Path(out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}", flush=True)
    print("GUARDS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
