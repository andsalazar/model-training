"""Check whether the chat template supports thinking mode, and how it is set.

The paper evaluates with thinking on. If the template defaults to thinking off,
every run silently evaluates a different model configuration.
"""

from __future__ import annotations

import sys

from transformers import AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-0.8B"


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    template = tok.chat_template or ""

    print(f"model: {MODEL}")
    print(f"has chat_template: {template is not None}")
    print(f"mentions enable_thinking: {'enable_thinking' in template}")
    print(f"occurrences of 'thinking': {template.count('thinking')}")

    messages = [{"role": "user", "content": "What is 2+2?"}]

    for label, kwargs in (
        ("default", {}),
        ("thinking=True", {"enable_thinking": True}),
        ("thinking=False", {"enable_thinking": False}),
    ):
        try:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
            tail = text[-160:].replace("\n", "\\n")
            print(f"\n--- {label} ---")
            print(f"  tail: ...{tail}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"\n--- {label} ---\n  ERROR: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
