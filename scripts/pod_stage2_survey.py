"""Stage 2 survey: what does this model actually do badly?

Before choosing a second behaviour to edit, measure the model on a spread of
tasks with a generous cap. The point is to find a behaviour that is real on this
checkpoint rather than one that sounds plausible in a task brief -- Stage 1 spent
several runs editing against a behaviour the model did not exhibit.

Run as a file, not a heredoc: vLLM spawns a core process that re-imports the main
module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

from wsp.runner import VLLMRunner, classify

MODEL = "/local/models/qwen"
MAX_TOKENS = 2048

PROBES = [
    # arithmetic, the Stage 1 behaviour
    ("arith_12x11", "What is 12 times 11? Reply with only the digits.", "132"),
    ("arith_9x8", "What is 9 times 8? Reply with only the digits.", "72"),
    # a harder arithmetic case, to see whether length tracks difficulty
    ("arith_17x23", "What is 17 times 23? Reply with only the digits.", "391"),
    ("arith_347x28", "What is 347 times 28? Reply with only the digits.", "9716"),
    # non-arithmetic reasoning: does it over-think these too?
    (
        "logic_syllogism",
        "All cats are mammals. All mammals are animals. Are all cats animals? "
        "Reply yes or no.",
        "yes",
    ),
    (
        "logic_order",
        "If A is taller than B, and B is taller than C, is A taller than C? "
        "Reply yes or no.",
        "yes",
    ),
    # factual recall: short, and a guard against a global verbosity edit
    ("fact_capital", "Name the capital of France. Reply with only the city.", "Paris"),
    ("fact_planet", "Which planet is closest to the Sun? Reply with only the name.", "Mercury"),
    # a task with a naturally long answer, where brevity is wrong
    (
        "long_legit",
        "List three uses of a hammer, one per line.",
        "nail",
    ),
    # instruction following: does it obey a format constraint?
    (
        "format_csv",
        "Output exactly three comma-separated colours, nothing else.",
        ",",
    ),
]


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/stage2_survey.json"
    runner = VLLMRunner(MODEL)
    rows = []

    for name, prompt, expected in PROBES:
        text = runner.generate(prompt, max_tokens=MAX_TOKENS)
        n_tokens = runner.count_tokens(text)
        signal = classify(text, n_tokens, MAX_TOKENS)
        row = {
            "probe": name,
            "prompt": prompt,
            "expected": expected,
            "correct": expected.lower() in text.lower(),
            **signal.as_dict(),
            "tail": text[-160:],
        }
        rows.append(row)
        print(
            f"{name:16s} tokens={signal.tokens:5d} cap={str(signal.hit_cap):5s} "
            f"rep={str(signal.repeated):5s} ok={row['correct']}",
            flush=True,
        )

    Path(out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
