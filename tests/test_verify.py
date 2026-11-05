"""Tests for independent re-measurement.

The motivating failure: explore14's editor applied two edits and reported
post-edit token counts without calling measure_loop after either one. Auditing
the transcript catches that after the fact, but the reliable check is to measure
the checkpoint again outside the editor. These tests pin the rule that decides
whether an edit counts.
"""

from __future__ import annotations

import pytest

from wsp.verify import (
    MIN_DELTA,
    Probe,
    Reading,
    Role,
    classify_change,
    compare,
    measure_all,
    verify_edit,
)


class FakeRunner:
    """A runner whose output is scripted per prompt."""

    def __init__(self, script: dict[str, str]) -> None:
        self.script = script
        self.calls: list[str] = []

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        self.calls.append(prompt)
        return self.script[prompt]

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def _reading(
    name: str,
    tokens: int,
    *,
    role: Role = Role.TARGET,
    correct: bool = True,
    hit_cap: bool = False,
) -> Reading:
    return Reading(
        probe=name,
        role=role,
        tokens=tokens,
        hit_cap=hit_cap,
        repeated=False,
        correct=correct,
    )


def test_classify_change_flags_a_lost_answer_as_broken() -> None:
    verdict = classify_change(
        _reading("t", 200), _reading("t", 150, correct=False)
    )
    assert verdict.state == "broken"


def test_broken_outranks_any_improvement() -> None:
    """A shorter wrong answer is not an improvement."""
    verdict = classify_change(
        _reading("t", 2000), _reading("t", 100, correct=False)
    )
    assert verdict.state == "broken"
    assert "expected answer" in verdict.detail


def test_classify_change_flags_a_new_cap_hit_as_regressed() -> None:
    """The exact explore13 failure: held-out 17x23 went 830 -> 2048."""
    verdict = classify_change(
        _reading("h", 830, role=Role.HELDOUT),
        _reading("h", 2048, role=Role.HELDOUT, hit_cap=True),
    )
    assert verdict.state == "regressed"
    assert "cap" in verdict.detail


def test_classify_change_flags_large_growth_as_regressed() -> None:
    verdict = classify_change(
        _reading("h", 100, role=Role.HELDOUT), _reading("h", 400, role=Role.HELDOUT)
    )
    assert verdict.state == "regressed"


def test_classify_change_flags_a_large_fall_as_improved() -> None:
    verdict = classify_change(_reading("t", 2048), _reading("t", 164))
    assert verdict.state == "improved"


def test_small_change_is_unchanged_not_improved() -> None:
    """Short answers vary by a few tokens; that is not an effect.

    Without a noise floor every edit looks like it did something, which is how a
    null result gets reported as a win.
    """
    verdict = classify_change(
        _reading("g", 204, role=Role.GUARD), _reading("g", 204 + MIN_DELTA - 1, role=Role.GUARD)
    )
    assert verdict.state == "unchanged"
    assert str(MIN_DELTA) in verdict.detail


def test_baseline_with_no_tokens_is_unchanged() -> None:
    verdict = classify_change(_reading("t", 0), _reading("t", 50))
    assert verdict.state == "unchanged"
    assert "nothing to compare" in verdict.detail


def test_validated_requires_every_target_to_improve() -> None:
    verdicts = compare(
        [
            _reading("t1", 2000),
            _reading("t2", 2000),
            _reading("g", 204, role=Role.GUARD),
        ],
        [
            _reading("t1", 200),
            _reading("t2", 1700),  # 0.85x baseline: not enough to count as improved
            _reading("g", 204, role=Role.GUARD),
        ],
    )
    assert verdicts.status == "partial"


def test_validated_when_targets_improve_and_nothing_is_damaged() -> None:
    comparison = compare(
        [
            _reading("t", 2048),
            _reading("h", 830, role=Role.HELDOUT),
            _reading("g", 204, role=Role.GUARD),
        ],
        [
            _reading("t", 164),
            _reading("h", 556, role=Role.HELDOUT),
            _reading("g", 204, role=Role.GUARD),
        ],
    )
    assert comparison.status == "validated"
    assert comparison.validated
    assert "VALIDATED" in comparison.summary()


def test_harmful_outranks_a_helped_target() -> None:
    """The mistake to talk yourself out of: the target improved, but at a cost.

    explore13's contiguous-window edit cut the target from 2048 to 190 while
    pushing held-out 17x23 from 830 to 2048. That is not a validated edit.
    """
    comparison = compare(
        [
            _reading("t", 2048),
            _reading("h", 830, role=Role.HELDOUT),
        ],
        [
            _reading("t", 190),
            _reading("h", 2048, role=Role.HELDOUT),
        ],
    )
    assert comparison.status == "harmful"
    assert not comparison.validated
    assert "HARMFUL" in comparison.summary()
    assert "h (regressed)" in comparison.summary()


def test_guard_regression_is_harmful() -> None:
    """An edit that shortens unrelated behaviour is a verbosity knob, not a fix."""
    comparison = compare(
        [
            _reading("t", 2048),
            _reading("g", 204, role=Role.GUARD),
        ],
        [
            _reading("t", 164),
            _reading("g", 1200, role=Role.GUARD),
        ],
    )
    assert comparison.status == "harmful"


def test_no_effect_when_nothing_moves() -> None:
    comparison = compare(
        [_reading("t", 2048)], [_reading("t", 2010)]
    )
    assert comparison.status == "no-effect"
    assert "NO EFFECT" in comparison.summary()


def test_compare_matches_by_probe_name_not_position() -> None:
    """Re-measuring must line up with the baseline even if order differs."""
    comparison = compare(
        [_reading("a", 2000), _reading("b", 800)],
        [_reading("b", 100), _reading("a", 100)],
    )
    by_name = {v.probe: v for v in comparison.verdicts}
    assert by_name["a"].state == "improved"
    assert by_name["b"].state == "improved"


def test_compare_skips_probes_missing_from_the_edited_set() -> None:
    comparison = compare([_reading("a", 2000), _reading("b", 800)], [_reading("a", 100)])
    assert [v.probe for v in comparison.verdicts] == ["a"]


def test_measure_all_uses_the_expected_answer_for_correctness() -> None:
    probe = Probe(name="p", prompt="q", expected="132", role=Role.TARGET)
    runner = FakeRunner({"q": "the answer is 132"})
    readings = measure_all(runner, [probe])
    assert readings[0].correct


def test_measure_all_marks_a_missing_answer_incorrect() -> None:
    probe = Probe(name="p", prompt="q", expected="132", role=Role.TARGET)
    runner = FakeRunner({"q": "the answer is 131"})
    assert not measure_all(runner, [probe])[0].correct


def test_verify_edit_measures_after_the_baseline() -> None:
    probe = Probe(name="p", prompt="q", expected="132", role=Role.TARGET)
    runner = FakeRunner({"q": "132"})
    baseline = measure_all(runner, [probe])
    comparison = verify_edit(runner, [probe], baseline)
    # The fake returns the same text, so the edit is a no-op by construction.
    assert comparison.status == "no-effect"


def test_comparison_serialises_for_a_report() -> None:
    comparison = compare(
        [_reading("t", 2048), _reading("h", 830, role=Role.HELDOUT)],
        [_reading("t", 164), _reading("h", 556, role=Role.HELDOUT)],
    )
    payload = comparison.as_dict()
    assert payload["status"] == "validated"
    assert payload["validated"] is True
    assert {v["probe"] for v in payload["verdicts"]} == {"t", "h"}
    assert payload["baseline"][0]["tokens"] == 2048


def test_summary_states_the_reason_for_the_verdict() -> None:
    comparison = compare([_reading("h", 830, role=Role.HELDOUT)], [_reading("h", 2048, role=Role.HELDOUT)])
    summary = comparison.summary()
    assert "HARMFUL" in summary
    assert "h" in summary


@pytest.mark.parametrize(
    ("base", "edited", "expected"),
    [
        (2048, 164, "improved"),
        (830, 556, "improved"),
        (830, 2048, "regressed"),
        (204, 204, "unchanged"),
    ],
)
def test_the_ranked_column_result_is_validated(base: int, edited: int, expected: str) -> None:
    """The real numbers from the layer-12 ranked-column sweep."""
    verdict = classify_change(_reading("p", base), _reading("p", edited))
    assert verdict.state == expected
