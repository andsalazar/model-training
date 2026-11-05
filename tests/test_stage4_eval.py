"""Tests for Stage 4's evaluation judging.

The eval has to apply the same rule that refuted Stages 1-3, plus two checks the
earlier stages could not make: guard damage called out separately, and an ablation
that confirms the effect is the adapter rather than drift.

These test the pure judging logic, which is where a mistake would silently
validate a harmful adapter.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pod_eval_adapter.py"


def _load():
    spec = importlib.util.spec_from_file_location("pod_eval_adapter", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def _row(probe: str, kind: str, tokens: int, *, correct=True, hit_cap=False) -> dict:
    return {
        "probe": probe,
        "kind": kind,
        "tokens": tokens,
        "hit_cap": hit_cap,
        "correct": correct,
    }


def test_verdicts_flags_a_lost_answer_as_broken(mod) -> None:
    v = mod.verdicts(
        [_row("p", "target", 200)],
        [_row("p", "target", 100, correct=False)],
    )
    assert v[0]["state"] == "broken"


def test_verdicts_flags_a_new_cap_hit_as_regressed(mod) -> None:
    v = mod.verdicts(
        [_row("g", "guard", 1204)],
        [_row("g", "guard", 2048, hit_cap=True)],
    )
    assert v[0]["state"] == "regressed"


def test_verdicts_flags_a_large_fall_as_improved(mod) -> None:
    v = mod.verdicts(
        [_row("t", "target", 1869)],
        [_row("t", "target", 323)],
    )
    assert v[0]["state"] == "improved"


def test_verdicts_treat_a_small_change_as_unchanged(mod) -> None:
    """Short answers vary by a few tokens; that is not an effect."""
    v = mod.verdicts(
        [_row("t", "target", 232)],
        [_row("t", "target", 240)],
    )
    assert v[0]["state"] == "unchanged"


def test_summarise_reports_harmful_over_validated(mod) -> None:
    """The Stage 3 failure: the target improved and the edit is still a failure."""
    rows = [
        {
            "probe": "t",
            "kind": "target",
            "baseline": 1869,
            "edited": 323,
            "state": "improved",
        },
        {
            "probe": "g",
            "kind": "guard",
            "baseline": 1204,
            "edited": 2048,
            "state": "regressed",
        },
    ]
    result = mod.summarise(rows, "test")
    assert result["status"] == "harmful"
    assert "g" in result["harmful"]


def test_summarise_calls_out_guard_damage_separately(mod) -> None:
    """Guard damage is the specific failure mode Stage 4 exists to avoid."""
    rows = [
        {
            "probe": "t",
            "kind": "target",
            "baseline": 1869,
            "edited": 323,
            "state": "improved",
        },
        {
            "probe": "poem",
            "kind": "guard",
            "baseline": 1971,
            "edited": 2048,
            "state": "regressed",
        },
    ]
    result = mod.summarise(rows, "test")
    assert result["guard_harmful"] == ["poem"]


def test_summarise_validates_only_when_nothing_is_harmful(mod) -> None:
    rows = [
        {
            "probe": "t",
            "kind": "target",
            "baseline": 1869,
            "edited": 323,
            "state": "improved",
        },
        {
            "probe": "poem",
            "kind": "guard",
            "baseline": 1971,
            "edited": 1900,
            "state": "unchanged",
        },
    ]
    assert mod.summarise(rows, "test")["status"] == "validated"


def test_summarise_is_no_effect_when_targets_do_not_move(mod) -> None:
    rows = [
        {
            "probe": "t",
            "kind": "target",
            "baseline": 1869,
            "edited": 1860,
            "state": "unchanged",
        }
    ]
    assert mod.summarise(rows, "test")["status"] == "no-effect"


def test_summarise_does_not_validate_on_guard_improvement_alone(mod) -> None:
    """Shortening a guard is not a target improvement, and must not validate."""
    rows = [
        {
            "probe": "t",
            "kind": "target",
            "baseline": 1869,
            "edited": 1860,
            "state": "unchanged",
        },
        {
            "probe": "g",
            "kind": "guard",
            "baseline": 1971,
            "edited": 400,
            "state": "improved",
        },
    ]
    assert mod.summarise(rows, "test")["status"] == "no-effect"


def test_probe_set_covers_every_kind(mod) -> None:
    """A probe set missing guards would repeat Stage 1's mistake."""
    kinds = {p[3] for p in mod.PROBES}
    assert kinds == {"target", "heldout", "guard"}


def test_probe_set_includes_the_probes_that_broke_stage3(mod) -> None:
    """poem and hammer are the specification, not optional extras."""
    names = {p[0] for p in mod.PROBES}
    assert "guard_poem" in names
    assert "guard_hammer" in names
