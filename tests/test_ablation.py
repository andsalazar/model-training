"""Tests for the blind-vs-weights ablation harness."""

from __future__ import annotations

from wsp.tasks.ablation import (
    AblationResult,
    Condition,
    build_prompts,
    detect_recall,
)


def test_build_prompts_returns_matched_pair() -> None:
    with_weights, blind = build_prompts("0.1 0.2\n0.3 0.4")

    assert with_weights.condition is Condition.WEIGHTS
    assert blind.condition is Condition.BLIND
    # Same system prompt, so the conditions differ only in the weight block.
    assert with_weights.system == blind.system


def test_weights_arm_contains_the_block() -> None:
    with_weights, blind = build_prompts("0.1 0.2\n0.3 0.4")

    assert "0.1 0.2" in with_weights.user
    assert "0.1 0.2" not in blind.user


def test_blind_arm_asks_for_recall_disclosure() -> None:
    _, blind = build_prompts("0.1 0.2")
    assert "recall" in blind.user.lower()


def test_build_prompts_handles_missing_weights() -> None:
    with_weights, blind = build_prompts(None)
    assert "unavailable" in with_weights.user
    assert "unavailable" not in blind.user


def test_detect_recall_finds_markers() -> None:
    text = "According to the model card, this is a known issue."
    markers = detect_recall(text)
    assert "model card" in markers
    assert "known issue" in markers


def test_detect_recall_returns_empty_for_neutral_text() -> None:
    assert detect_recall("Subtract 0.3 from columns 0-63 of layer 15.") == []


def _result(weights: str, blind: str) -> AblationResult:
    return AblationResult(
        editor="qwen",
        weights_response=weights,
        blind_response=blind,
        weights_recall_markers=detect_recall(weights),
        blind_recall_markers=detect_recall(blind),
    )


def test_verdict_recall_sufficient_when_both_succeed() -> None:
    r = _result(
        "Subtract the direction from layer 15 columns 0-63.",
        "Subtract the direction from layer 15 columns 0-63.",
    )
    assert r.weights_succeeded and r.blind_succeeded
    assert "contaminated" in r.verdict()


def test_verdict_weight_reading_when_blind_declines() -> None:
    r = _result(
        "Subtract 0.4 from layer 15 columns 0-63.",
        "I cannot determine this without seeing the weights.",
    )
    assert r.weights_succeeded
    assert not r.blind_succeeded
    assert "weight-reading supported" in r.verdict()


def test_verdict_no_signal_when_both_decline() -> None:
    r = _result(
        "I cannot determine a change from this slice.",
        "I cannot determine this without the weights.",
    )
    assert "no signal" in r.verdict()


def test_declining_response_is_not_a_success() -> None:
    # Mentions "layer" but explicitly declines, so it must not count.
    r = _result(
        "I cannot identify which layer to change.",
        "I cannot determine this.",
    )
    assert not r.weights_succeeded
