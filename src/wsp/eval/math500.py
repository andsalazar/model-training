"""MATH-500 evaluation, matching the paper's harness exactly.

Ported from the released `exp8_math/code/mathqa.py` and `evaluate.py` so scores
are comparable rather than merely similar. Three details matter, and each was
wrong in an earlier version of this module:

1. **Prompt.** The user turn is the problem, a newline, then Qwen's math
   instruction. No wrapper, no "show your reasoning" preamble.
2. **Extraction.** The last ``\\boxed{}`` (or ``\\fbox{}``) in the text after the
   final ``</think>``. An unclosed or empty box yields ``None``. There is
   deliberately **no fallback to the last number**: a truncated response's last
   number is an intermediate step, and crediting it inflates the score.
3. **Grading.** ``math_verify`` equivalence, not string normalisation. ``5`` and
   ``\\frac{10}{2}`` are the same answer; ``0.5`` and ``\\frac{1}{2}`` are too.

The original MATH checker is kept as a secondary signal because the paper
reports both.
"""

from __future__ import annotations

import functools
import re
import signal
from dataclasses import dataclass

#: Qwen's math instruction, verbatim from the paper's harness.
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

_BOX_FORMS = re.compile(r"\\boxed|\\fbox")


def build_prompt(problem: str) -> str:
    """The user turn: the problem, a newline, Qwen's math instruction."""
    return problem + "\n" + INSTRUCTION


def answer_text(text: str) -> str:
    """The text extraction reads: anything before the last ``</think>`` dropped."""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def _box_at(text: str, idx: int) -> str | None:
    """Content of the box starting at ``text[idx]``, or None."""
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key) :]
    stripped = rest.lstrip()

    if stripped.startswith("{"):
        depth = 0
        for k, ch in enumerate(stripped):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out = stripped[1:k].strip()
                    return out or None
        return None  # unclosed, e.g. cut off inside the box

    # '\boxed 5' with no braces: read to the next '$' or end of line.
    if key == "\\boxed" and rest.startswith(" "):
        out = rest.strip().split("$")[0].split("\n")[0].strip()
        return out or None
    return None


def last_boxed(text: str) -> str | None:
    """Content of the last box in ``text``, or None."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _box_at(text, idx)


def extract_answer(text: str) -> str | None:
    """The final boxed answer, or None. No numeric fallback, by design."""
    return last_boxed(answer_text(text))


def count_boxes(text: str) -> int:
    """How many boxes a response contains, for diagnosing extraction failures."""
    return len(_BOX_FORMS.findall(answer_text(text)))


# ---------------------------------------------------------------- grading
_MV: list = []


def _math_verify():
    """The math_verify module, or None when it is not installed."""
    if not _MV:
        try:
            import math_verify  # noqa: PLC0415

            _MV.append(math_verify)
        except ImportError:
            _MV.append(None)
    return _MV[0]


def math_verify_available() -> bool:
    return _math_verify() is not None


def _timeout_arg() -> int | None:
    """math_verify's timeout uses ``signal.alarm``, which does not exist on
    Windows; passing one there makes every parse silently return ``[]``. Only
    request a timeout where the mechanism exists.
    """
    return 5 if hasattr(signal, "alarm") else None


@functools.lru_cache(maxsize=1 << 16)
def _parsed(value: str):
    mv = _math_verify()
    if mv is None:
        return None
    try:
        return mv.parse("\\boxed{" + value + "}", parsing_timeout=_timeout_arg())
    except Exception:  # noqa: BLE001 - unparseable is a grading outcome
        return None


def _normalize(value: str) -> str:
    """Fallback comparison used only when math_verify is unavailable."""
    value = value.strip().strip("$").strip()
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    value = value.replace("\\ ", "")
    value = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", value)
    value = value.replace(",", "").replace(" ", "")
    value = value.rstrip(".")
    if re.fullmatch(r"-?\d+\.0+", value):
        value = value.split(".")[0]
    return value


def is_correct(prediction: str | None, gold: str | None) -> bool:
    """Whether a prediction is equivalent to the gold answer.

    Uses math_verify when available, which is what the paper grades with.
    Falls back to normalised string equality so the harness still runs without
    the dependency, at the cost of comparability.
    """
    if prediction is None or gold is None:
        return False

    mv = _math_verify()
    if mv is not None:
        g, p = _parsed(gold), _parsed(prediction)
        if not g or not p:
            return False
        try:
            return bool(mv.verify(g, p, timeout_seconds=_timeout_arg()))
        except Exception:  # noqa: BLE001
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
