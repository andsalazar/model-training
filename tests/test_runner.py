"""Tests for loop detection and the runner's classification logic."""

from __future__ import annotations

import pytest

from wsp.runner import NullRunner, classify, detect_repetition


def test_detect_repetition_finds_repeated_tail() -> None:
    block = "the same reasoning step over and over again " * 8
    text = "start " + block + block + block
    assert detect_repetition(text)


def test_detect_repetition_ignores_short_text() -> None:
    assert not detect_repetition("short")


def test_detect_repetition_ignores_varied_text() -> None:
    text = " ".join(f"word{i}" for i in range(600))
    assert not detect_repetition(text)


def test_classify_hit_cap_is_looping() -> None:
    signal = classify("some text", n_tokens=2048, max_tokens=2048)
    assert signal.looped
    assert signal.hit_cap


def test_classify_clean_stop_is_not_looping() -> None:
    signal = classify("a short answer", n_tokens=12, max_tokens=2048)
    assert not signal.looped
    assert not signal.hit_cap


def test_classify_repetition_without_cap_is_looping() -> None:
    block = "circling back to the same point " * 10
    signal = classify(block * 3, n_tokens=100, max_tokens=2048)
    assert signal.looped
    assert signal.repeated
    assert not signal.hit_cap


def test_signal_serializes() -> None:
    d = classify("x", 1, 10).as_dict()
    assert set(d) == {"looped", "hit_cap", "repeated", "tokens", "text_chars"}


def test_null_runner_generate_raises() -> None:
    with pytest.raises(RuntimeError, match="no model attached"):
        NullRunner().generate("p")


def test_null_runner_measure_raises() -> None:
    with pytest.raises(RuntimeError, match="no model attached"):
        NullRunner().measure("p")


def test_null_runner_reload_raises() -> None:
    with pytest.raises(RuntimeError, match="no model attached"):
        NullRunner().reload("x")
