"""Stage 4 training data: over-thinking targets paired with elaboration guards.

The design follows directly from Stage 3's failure. A directional edit suppressed
a shared "keep checking" state and therefore broke long-form tasks, so the adapter
has to learn *when* to suppress. That means the dataset must contain both kinds of
example, in comparable numbers:

  TARGET   a prompt where the model over-thinks, where a short answer is correct
  GUARD    a prompt where elaboration is correct, where suppressing it is wrong

Stage 3's `guard_poem` result specified the guard half: a two-line poem must not
collapse. Every guard example here is something the model should keep doing
verbosely.

The loss is implemented in the training script, not here; this module only
produces prompts with their expectations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    """One training prompt.

    ``expected`` is the string that must appear in the output for the answer to
    count. ``max_len`` is the length the adapter is pulled toward -- generous for
    guards, tight for targets, so brevity is only rewarded where it is right.
    """

    prompt: str
    expected: str
    max_len: int
    kind: str
    behaviour: str

    def as_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "expected": self.expected,
            "max_len": self.max_len,
            "kind": self.kind,
            "behaviour": self.behaviour,
        }


#: Targets: the model reaches the right answer and keeps re-verifying. Each asks
#: for only the answer, so any length beyond the answer is over-thinking.
_ARITHMETIC = [
    ("What is 12 times 11? Reply with only the digits.", "132"),
    ("What is 17 times 23? Reply with only the digits.", "391"),
    ("What is 13 times 14? Reply with only the digits.", "182"),
    ("What is 9 times 8? Reply with only the digits.", "72"),
    ("What is 24 times 17? Reply with only the digits.", "408"),
    ("What is 15 times 15? Reply with only the digits.", "225"),
    ("What is 7 times 13? Reply with only the digits.", "91"),
    ("What is 18 times 12? Reply with only the digits.", "216"),
]

#: Logic targets: same over-thinking, different behaviour.
_LOGIC = [
    (
        "All cats are mammals. All mammals are animals. Are all cats animals? "
        "Reply with only yes or no.",
        "yes",
    ),
    (
        "If A is taller than B, and B is taller than C, is A taller than C? "
        "Reply with only yes or no.",
        "yes",
    ),
    (
        "All roses are flowers. Some flowers fade quickly. Do all roses fade "
        "quickly? Reply with only yes or no.",
        "no",
    ),
    (
        "If it rains, the ground is wet. The ground is wet. Did it necessarily "
        "rain? Reply with only yes or no.",
        "no",
    ),
]

#: Factual targets: short answers the model currently wraps in verification.
_FACTUAL = [
    ("Name the capital of France. Reply with only the city.", "Paris"),
    ("Which planet is closest to the Sun? Reply with only the name.", "Mercury"),
    ("What is the chemical symbol for water? Reply with only the formula.", "H2O"),
    ("How many sides does a hexagon have? Reply with only the number.", "6"),
]

#: Guards: elaboration is CORRECT here. These are the examples Stage 3's edit
#: destroyed, and they are what makes the adapter behaviour-selective rather than
#: a global verbosity knob.
_GUARDS = [
    ("Write two lines of a poem about the sea.", "sea"),
    ("List three uses of a hammer, one per line.", "nail"),
    ("Explain in two sentences why the sky appears blue.", "light"),
    ("Describe the water cycle in three sentences.", "water"),
    ("Write a four-line rhyme about winter.", "winter"),
    ("Give three reasons to wear a seatbelt.", "safe"),
    ("Explain what a prime number is, in two sentences.", "divis"),
    ("Describe how a bicycle works, in three sentences.", "pedal"),
]

#: Length the adapter is pulled toward. Targets are pulled to a short answer;
#: guards are pulled to a *floor* (max_len high enough that the current behaviour
#: is not penalised) so there is no gradient toward brevity on them.
TARGET_MAX_LEN = 64
GUARD_MAX_LEN = 2048


def build_dataset() -> list[Example]:
    """Every example, targets and guards in comparable numbers.

    Counts: 16 targets (8 arithmetic, 4 logic, 4 factual) against 8 guards. The
    targets outweigh the guards 2:1 because there are three target behaviours and
    one guard type, but every guard class is represented, and the length term is a
    floor rather than a target so an under-weighted guard still gets no gradient
    toward brevity.
    """
    examples: list[Example] = []

    for prompt, answer in _ARITHMETIC:
        examples.append(
            Example(prompt, answer, TARGET_MAX_LEN, "target", "arithmetic")
        )
    for prompt, answer in _LOGIC:
        examples.append(Example(prompt, answer, TARGET_MAX_LEN, "target", "logic"))
    for prompt, answer in _FACTUAL:
        examples.append(Example(prompt, answer, TARGET_MAX_LEN, "target", "factual"))
    for prompt, expected in _GUARDS:
        examples.append(
            Example(prompt, expected, GUARD_MAX_LEN, "guard", "elaboration")
        )

    return examples


def split_dataset(
    examples: list[Example], *, holdout_behaviour: str = "factual"
) -> tuple[list[Example], list[Example]]:
    """Train/test split that holds out an entire behaviour.

    Splitting by behaviour rather than by example is the point: a held-out
    *example* tests memorisation, while a held-out *behaviour* tests whether the
    adapter generalises. ``factual`` is the default holdout because it is distinct
    from arithmetic and small enough that leakage would be obvious.
    """
    train = [e for e in examples if e.behaviour != holdout_behaviour]
    test = [e for e in examples if e.behaviour == holdout_behaviour]
    return train, test


def write_jsonl(examples: list[Example], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example.as_dict()) + "\n")
    return path


if __name__ == "__main__":
    all_examples = build_dataset()
    train, test = split_dataset(all_examples)
    write_jsonl(train, "/workspace/stage4_train.jsonl")
    write_jsonl(test, "/workspace/stage4_test.jsonl")
    print(f"total {len(all_examples)}  train {len(train)}  held-out behaviour {len(test)}")
    kinds: dict[str, int] = {}
    for e in all_examples:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    print("by kind:", kinds)
