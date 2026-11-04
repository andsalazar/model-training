"""MATH-500 evaluation.

The paper's headline metric. Scoring is exact-match on the final boxed answer,
which is what makes the 7.9% -> 48.1% jump meaningful: the model either reaches
the right answer or it does not.

Runs on GPU. Kept free of any WSP-specific logic so the same harness scores the
baseline and every edited checkpoint identically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Matches \boxed{...} including one level of nested braces.
_BOXED = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")

#: Fallback: the last number in the response.
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> str | None:
    """Pull the final answer out of a model response.

    Prefers the last ``\\boxed{...}``. Falls back to the last number only when
    the response looks finished, because a truncated response's last number is
    an intermediate step, not an answer.
    """
    boxed = _BOXED.findall(text)
    if boxed:
        return _normalize(boxed[-1])

    if not _looks_complete(text):
        return None

    numbers = _NUMBER.findall(text)
    if numbers:
        return _normalize(numbers[-1])
    return None


def _looks_complete(text: str) -> bool:
    """Whether a response reached a conclusion rather than being cut off.

    A truncated response ends mid-sentence or mid-formula; scoring its last
    number would credit an intermediate value as the answer.
    """
    stripped = text.rstrip()
    if not stripped:
        return False
    # Unbalanced braces or a trailing operator means generation was cut short.
    if stripped.count("{") != stripped.count("}"):
        return False
    if stripped.count("$") % 2 != 0:
        return False
    return stripped[-1] in ".!)$}0123456789"


def _normalize(value: str) -> str:
    """Strip formatting that differs between equivalent answers.

    Handles the LaTeX that MATH-500 answers are written in: spacing commands,
    ``\\left``/``\\right`` delimiters, and ``\\text{}`` wrappers all vary between
    an answer and its equivalent form.
    """
    value = value.strip().strip("$").strip()
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    value = value.replace("\\ ", "")
    value = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", value)
    value = value.replace(",", "").replace(" ", "")
    value = value.rstrip(".")
    # 5.0 and 5 are the same answer.
    if re.fullmatch(r"-?\d+\.0+", value):
        value = value.split(".")[0]
    return value


def is_correct(prediction: str | None, gold: str | None) -> bool:
    """Exact match after normalization."""
    if prediction is None or gold is None:
        return False
    return _normalize(prediction) == _normalize(gold)


@dataclass
class EvalResult:
    """Aggregate score for one checkpoint."""

    correct: int
    total: int
    samples: list[dict]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def as_dict(self) -> dict:
        return {
            "correct": self.correct,
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "accuracy_pct": f"{100 * self.accuracy:.1f}%",
        }


def build_prompt(question: str) -> str:
    """Prompt format used for every arm, so scores are comparable."""
    return (
        "Solve the following math problem. Show your reasoning, then give the "
        "final answer in \\boxed{}.\n\n"
        f"{question}"
    )
