"""Tests for MATH-500 extraction and grading, matching the paper's harness."""

from __future__ import annotations

import pytest

from wsp.eval.math500 import (
    INSTRUCTION,
    EvalResult,
    answer_text,
    build_prompt,
    count_boxes,
    extract_answer,
    is_correct,
    last_boxed,
)


# ---------------------------------------------------------------- prompt
def test_build_prompt_is_problem_newline_instruction() -> None:
    assert build_prompt("What is 2+2?") == "What is 2+2?\n" + INSTRUCTION


def test_instruction_asks_for_boxed_answer() -> None:
    assert "\\boxed{}" in INSTRUCTION
    assert "step by step" in INSTRUCTION


# ---------------------------------------------------------------- extraction
def test_extract_simple_box() -> None:
    assert extract_answer(r"The answer is \boxed{42}.") == "42"


def test_extract_last_box_wins() -> None:
    assert extract_answer(r"First \boxed{1}, but actually \boxed{7}.") == "7"


def test_extract_nested_braces() -> None:
    assert extract_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_extract_fbox() -> None:
    assert extract_answer(r"\fbox{9}") == "9"


def test_extract_boxed_with_space_no_braces() -> None:
    # The paper reads an unbraced '\boxed 5' to the end of the line, so the
    # trailing words are part of the extracted value.
    assert extract_answer(r"so \boxed 5 is the answer") == "5 is the answer"


def test_extract_boxed_with_space_stops_at_dollar() -> None:
    assert extract_answer(r"\boxed 5 $ and more") == "5"


def test_extract_unclosed_box_is_none() -> None:
    # Cut off inside the box: the paper treats this as no answer.
    assert extract_answer(r"so we get \boxed{\frac{1}{2") is None


def test_extract_empty_box_is_none() -> None:
    assert extract_answer(r"\boxed{}") is None


def test_extract_no_box_is_none() -> None:
    assert extract_answer("The answer is 42.") is None


def test_extract_does_not_fall_back_to_numbers() -> None:
    # The critical difference from the earlier implementation: a bare number
    # is not an answer, because a truncated response's last number is an
    # intermediate step.
    assert extract_answer("So the total is 128 apples.") is None


def test_extract_ignores_reasoning_before_last_think_close() -> None:
    text = r"thinking \boxed{wrong} more </think> so \boxed{right}"
    assert extract_answer(text) == "right"


def test_answer_text_strips_before_last_think_close() -> None:
    assert answer_text("a </think> b </think> c") == "c"


def test_answer_text_without_close_is_unchanged() -> None:
    assert answer_text("plain") == "plain"


def test_count_boxes() -> None:
    assert count_boxes(r"\boxed{1} and \boxed{2}") == 2
    assert count_boxes("none") == 0


def test_last_boxed_returns_none_for_empty_text() -> None:
    assert last_boxed("") is None


# ---------------------------------------------------------------- grading
def test_is_correct_rejects_none() -> None:
    assert not is_correct(None, "5")
    assert not is_correct("5", None)


def test_is_correct_exact_match() -> None:
    assert is_correct("5", "5")


def test_is_correct_rejects_wrong_answer() -> None:
    assert not is_correct("6", "5")


def test_is_correct_handles_equivalent_forms() -> None:
    # math_verify treats these as equal; string normalisation would not.
    assert is_correct(r"\frac{1}{2}", "0.5")


def test_is_correct_handles_latex_wrappers() -> None:
    assert is_correct(r"\text{Evelyn}", "Evelyn")


def test_is_correct_handles_left_right() -> None:
    assert is_correct(r"\left( 3, \frac{\pi}{2} \right)", r"(3,\frac{\pi}{2})")


# ---------------------------------------------------------------- aggregate
def test_accuracy_computation() -> None:
    result = EvalResult(correct=3, total=10, samples=[])
    assert result.accuracy == pytest.approx(0.3)
    assert result.as_dict()["accuracy_pct"] == "30.0%"


def test_accuracy_handles_empty() -> None:
    assert EvalResult(correct=0, total=0, samples=[]).accuracy == 0.0
