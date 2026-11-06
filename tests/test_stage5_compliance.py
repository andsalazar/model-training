"""Tests for Stage 5's format-compliance check.

The bug this guards against: the first version tested only "expected substring
present and short", and the expected strings are single characters. So
"Thinking Process:\n\n1" counted as a compliant answer to "output five
comma-separated numbers" because it contains "1". Six of eight targets were
reported newly compliant on that basis.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pod_stage5.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("pod_stage5", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pod_stage5"] = module
    spec.loader.exec_module(module)
    return module


def test_truncated_deliberation_is_not_compliant(mod) -> None:
    """The exact false positive the first eval reported as a success."""
    assert not mod._compliant("Thinking Process:\n\n1", "1", 6)


def test_bare_answer_is_compliant(mod) -> None:
    assert mod._compliant("1, 2, 3, 4, 5", "1", 12)


def test_answer_with_reasoning_is_not_compliant(mod) -> None:
    """Short but still deliberating: the length cap alone would pass this."""
    text = "Wait, I should check. 1, 2, 3, 4, 5"
    assert not mod._compliant(text, "1", 20)


def test_long_answer_is_not_compliant(mod) -> None:
    """Correct content but over the cap is still over-thinking."""
    assert not mod._compliant("1, 2, 3, 4, 5 " * 40, "1", 200)


def test_missing_answer_is_not_compliant(mod) -> None:
    assert not mod._compliant("red, green, blue", "1", 10)


def test_markdown_bold_counts_as_reasoning(mod) -> None:
    """The adapter's real outputs were full of **bold** deliberation headers."""
    assert not mod._compliant("**Analyze the Request:** red, green", "red", 12)


def test_guard_answers_are_judged_by_the_same_rule(mod) -> None:
    """A guard's expected answer is a real word, so a bare answer passes."""
    assert mod._compliant("132", "132", 1)
    assert not mod._compliant("Wait, 12 times 11 is 132", "132", 8)


def test_bare_thinking_prefix_is_not_compliant(mod) -> None:
    """The survivor the first correction still let through.

    "Thinking " is 3 tokens and contains a space, so a substring-plus-length
    test called it a compliant answer to "output three words separated by
    spaces". It is a truncated deliberation header.
    """
    assert not mod._compliant("Thinking ", " ", 3)
