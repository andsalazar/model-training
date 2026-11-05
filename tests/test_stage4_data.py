"""Tests for the Stage 4 training data.

The dataset is where the Stage 3 failure gets encoded: half the examples must be
tasks where elaboration is CORRECT, or the adapter just learns to be terse
everywhere and repeats the fixed-vector mistake with a bigger parameter count.
"""

from __future__ import annotations

from wsp.stage4_data import (
    GUARD_MAX_LEN,
    TARGET_MAX_LEN,
    Example,
    build_dataset,
    split_dataset,
    write_jsonl,
)


def test_dataset_contains_both_targets_and_guards() -> None:
    examples = build_dataset()
    kinds = {e.kind for e in examples}
    assert kinds == {"target", "guard"}


def test_guards_are_a_substantial_fraction() -> None:
    """A token guard set would let the adapter learn global terseness.

    Stage 3's failure was exactly that: a mechanism that suppressed a shared state
    regardless of task. The guard fraction is the defence, so it gets asserted
    rather than left to a comment.
    """
    examples = build_dataset()
    guards = [e for e in examples if e.kind == "guard"]
    assert len(guards) >= len(examples) * 0.3, (
        f"guards are {len(guards)}/{len(examples)}, too few to constrain the adapter"
    )


def test_targets_are_pulled_short_and_guards_are_not() -> None:
    """The length term must be one-sided: no gradient toward brevity on guards."""
    examples = build_dataset()
    for e in examples:
        if e.kind == "target":
            assert e.max_len == TARGET_MAX_LEN
            assert e.max_len < 128, "targets should be pulled to a short answer"
        else:
            assert e.max_len == GUARD_MAX_LEN
            assert e.max_len >= 1024, "guards must not be penalised for length"


def test_every_example_has_a_non_empty_expectation() -> None:
    """The correctness term is what stops the adapter buying brevity with wrong
    answers, and it is vacuous if `expected` is empty."""
    for e in build_dataset():
        assert e.expected.strip(), f"no expectation for {e.prompt[:50]!r}"


def test_target_expectations_are_short_answers() -> None:
    for e in build_dataset():
        if e.kind == "target":
            assert len(e.expected) <= 10, (
                f"target expectation {e.expected!r} is not a short answer"
            )


def test_prompts_are_unique() -> None:
    prompts = [e.prompt for e in build_dataset()]
    assert len(prompts) == len(set(prompts)), "duplicate prompts would double-weight"


def test_holdout_removes_an_entire_behaviour() -> None:
    """Holding out by behaviour is what tests generalisation rather than memory."""
    examples = build_dataset()
    train, test = split_dataset(examples, holdout_behaviour="factual")
    assert test, "holdout is empty"
    assert all(e.behaviour == "factual" for e in test)
    assert all(e.behaviour != "factual" for e in train)
    assert len(train) + len(test) == len(examples)


def test_holdout_keeps_the_guard_class_in_training() -> None:
    """A holdout that removes all guards would reintroduce the Stage 3 failure."""
    train, _ = split_dataset(build_dataset(), holdout_behaviour="factual")
    assert any(e.kind == "guard" for e in train)


def test_holdout_of_elaboration_leaves_no_guards_and_is_a_stated_risk() -> None:
    """Holding out elaboration is possible but removes every guard.

    Pinned so the consequence is explicit rather than discovered mid-run: this
    split asks whether targets alone are enough, which is a different experiment.
    """
    train, test = split_dataset(build_dataset(), holdout_behaviour="elaboration")
    assert not any(e.kind == "guard" for e in train)
    assert all(e.behaviour == "elaboration" for e in test)


def test_write_jsonl_round_trips(tmp_path) -> None:
    import json

    examples = [
        Example("q1", "a1", 64, "target", "arithmetic"),
        Example("q2", "a2", 2048, "guard", "elaboration"),
    ]
    path = write_jsonl(examples, tmp_path / "out.jsonl")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["prompt"] == "q1"
    assert first["max_len"] == 64
    assert first["kind"] == "target"


def test_example_is_hashable_and_serialisable() -> None:
    e = Example("q", "a", 64, "target", "arithmetic")
    assert isinstance(hash(e), int)
    assert e.as_dict()["behaviour"] == "arithmetic"
