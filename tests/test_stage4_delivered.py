"""Tests for Stage 4's "delivered" check.

The bug this guards against: the first Stage 4 eval used `expected in text` and
reported correct=True for outputs that end mid-deliberation. "132" appears in a
116-token trace that never closes its thinking block, so the substring test
called it a delivered answer. Re-judged strictly, most targets were unfinished.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pod_eval_adapter.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("pod_eval_adapter", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pod_eval_adapter"] = module
    spec.loader.exec_module(module)
    return module


def test_answer_buried_in_deliberation_is_not_delivered(mod) -> None:
    """The exact false positive: the answer is present but the block never closes."""
    text = (
        "Thinking Process:\n\n1. **Analyze the Request:** multiply 12 by 11.\n"
        "2. **Perform the Calculation:**\n    * 120 + 12 = 132"
    )
    assert not mod._delivered(text, "132", "target")


def test_bare_answer_is_delivered(mod) -> None:
    assert mod._delivered("132", "132", "target")


def test_answer_after_closing_the_block_is_delivered(mod) -> None:
    text = "Thinking Process:\n\n1. Compute.\n\n2. Done.\n</think>\n\n132"
    assert mod._delivered(text, "132", "target")


def test_missing_answer_is_not_delivered(mod) -> None:
    assert not mod._delivered("391", "132", "target")


def test_guards_are_judged_by_presence_only(mod) -> None:
    """Elaboration is the point for a guard, so long output is fine."""
    text = "Thinking Process:\n\n1. **Analyze:** the sea.\n\n2. Draft:\nThe waves roll in."
    assert mod._delivered(text, "sea", "guard")


def test_guard_without_its_subject_is_not_delivered(mod) -> None:
    assert not mod._delivered("Thinking Process:\n\n1. Analyze.", "sea", "guard")


def test_answer_only_in_the_trace_is_not_delivered(mod) -> None:
    """The answer computed inside the block but never emitted after it."""
    text = "Thinking Process:\n\n1. Compute 120 + 12 = 132\n2. Let me verify again."
    assert not mod._delivered(text, "132", "target")


def test_answer_after_close_but_still_deliberating_is_not_delivered(mod) -> None:
    """Closed the block, then kept arguing instead of answering."""
    text = "Thinking Process:\n\n1. Compute.\n</think>\n\nWait, let me check 132 again."
    assert not mod._delivered(text, "132", "target")
