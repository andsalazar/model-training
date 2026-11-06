"""Stage 5: does the adapter recipe generalise to a different behaviour?

Stage 4 produced a behaviour-selective adapter on *arithmetic/length* over-thinking.
That is one result on one behaviour. The question that decides whether it is a
recipe or a one-off is whether the same procedure works on a behaviour that shares
nothing with it.

The behaviour chosen is **format compliance**, which the Stage 2 survey found
sitting at the 2048 cap:

    "Output exactly three comma-separated colours, nothing else."  -> 2048 tokens

That is not arithmetic, not length, and not answer-shaped: the model produces the
right content and then argues with itself about the format. If a truncated-trace
adapter fixes that too, the recipe is a recipe.

Deliberately different in three ways:
  - target is format, not brevity of a numeric answer
  - the expected output is a fixed string, so "did it comply" is checkable
  - the guards are the *Stage 4 targets*, inverted: here arithmetic must keep
    working, so a regression in the other direction is caught

Run as a file, not a heredoc: torch's launcher re-imports main.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

sys.path.insert(0, "/workspace/src")


@dataclass(frozen=True)
class Ex:
    prompt: str
    expected: str
    kind: str          # "target" | "guard"
    behaviour: str

    def as_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "expected": self.expected,
            "kind": self.kind,
            "behaviour": self.behaviour,
        }


#: Format-compliance targets. Each names a format and forbids everything else.
#: The expected string is what a compliant answer must contain.
_FORMAT = [
    ("Output exactly three comma-separated colours, nothing else.", "red"),
    ("Output exactly five comma-separated numbers, nothing else.", "1"),
    ("List exactly two animals, separated by a comma, nothing else.", "cat"),
    ("Output exactly three words separated by spaces, nothing else.", " "),
    ("Output the days Monday to Wednesday, comma-separated, nothing else.", "Monday"),
    ("Output exactly two colours separated by a slash, nothing else.", "/"),
    ("Output exactly four letters separated by hyphens, nothing else.", "-"),
    ("Output the numbers 1 through 5 separated by commas, nothing else.", "5"),
]

#: Guards: the Stage 4 targets, here as things that must NOT collapse. If the
#: format adapter damages arithmetic, that is the same class of failure the
#: fixed-vector edits showed, and it bounds the recipe.
_GUARDS = [
    ("What is 12 times 11? Reply with only the digits.", "132"),
    ("What is 17 times 23? Reply with only the digits.", "391"),
    ("What is 9 times 8? Reply with only the digits.", "72"),
    ("Name the capital of France. Reply with only the city.", "Paris"),
    ("Which planet is closest to the Sun? Reply with only the name.", "Mercury"),
]


def build() -> list[Ex]:
    rows = [Ex(p, e, "target", "format") for p, e in _FORMAT]
    rows += [Ex(p, e, "guard", "arithmetic_guard") for p, e in _GUARDS]
    return rows


if __name__ == "__main__":
    rows = build()
    print(f"total {len(rows)}")
    for kind in ("target", "guard"):
        subset = [r for r in rows if r.kind == kind]
        print(f"  {kind}: {len(subset)}")
