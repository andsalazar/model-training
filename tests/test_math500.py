"""Tests for MATH-500 answer extraction and scoring."""

from __future__ import annotations

import pytest

from wsp.eval.math500 import EvalResult, build_prompt, extract_answer, is_correct


def test_extract_boxed() -> None:
    assert extract_answer(r"The answer is \boxed{42}.") == "42"


def test_extract_last_boxed_wins() -> None:
    text = r"First \boxed{1}, but actually \boxed{7}."
    assert extract_answer(text) == "7"


def test_extract_boxed_with_nested_braces() -> None:
    assert extract_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_extract_falls_back_to_last_number() -> None:
    assert extract_answer("So the total is 128 apples.") == "128"


def test_extract_returns_none_without_numbers() -> None:
    assert extract_answer("I cannot solve this.") is None


def test_extract_ignores_truncated_response() -> None:
    # Cut off mid-formula: the trailing number is an intermediate step.
    truncated = "Step 2. The formula is $r = \\sqrt{x^2 + y^2"
    assert extract_answer(truncated) is None


def test_extract_ignores_unbalanced_braces() -> None:
    assert extract_answer("so we get \\frac{1}{2 and then 7") is None


def test_extract_accepts_complete_sentence() -> None:
    assert extract_answer("Therefore the answer is 42.") == "42"


def test_normalize_strips_left_right() -> None:
    assert is_correct(r"\left( 3, \frac{\pi}{2} \right)", r"(3,\frac{\pi}{2})")


def test_normalize_strips_text_wrapper() -> None:
    assert is_correct(r"\text{Evelyn}", "Evelyn")


def test_normalize_strips_spacing_commands() -> None:
    assert is_correct(r"3 \, 4", "34")


def test_extract_strips_commas() -> None:
    assert extract_answer(r"\boxed{1,234}") == "1234"


def test_normalize_trailing_zero() -> None:
    assert is_correct("5.0", "5")
    assert is_correct("5", "5.0")


def test_normalize_negative() -> None:
    assert is_correct("-3", "-3")


def test_is_correct_rejects_none() -> None:
    assert not is_correct(None, "5")
    assert not is_correct("5", None)


def test_is_correct_rejects_wrong_answer() -> None:
    assert not is_correct("6", "5")


def test_is_correct_handles_fractions() -> None:
    assert is_correct(r"\frac{1}{2}", r"\frac{1}{2}")
    assert not is_correct(r"\frac{1}{2}", r"\frac{1}{3}")


def test_accuracy_computation() -> None:
    result = EvalResult(correct=3, total=10, samples=[])
    assert result.accuracy == pytest.approx(0.3)
    assert result.as_dict()["accuracy_pct"] == "30.0%"


def test_accuracy_handles_empty() -> None:
    assert EvalResult(correct=0, total=0, samples=[]).accuracy == 0.0


def test_build_prompt_requests_boxed_answer() -> None:
    prompt = build_prompt("What is 2+2?")
    assert "What is 2+2?" in prompt
    assert "boxed" in prompt
