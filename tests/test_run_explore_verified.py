"""Tests for the verified run script's checkpoint extraction.

The script re-measures whatever the editor edited, so it has to read the edit
calls correctly. Getting this wrong is silent: a missed edit means nothing is
verified, and a phantom edit means a reload failure is reported as a result.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_explore_verified.py"


def _load():
    """Import the script by path, since scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("run_explore_verified", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # The module imports run_explore, which needs /workspace/src on the path.
    sys.path.insert(0, str(_SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.fixture(scope="module")
def module():
    return _load()


def test_extracts_the_directory_an_edit_reports(module) -> None:
    calls = [
        {
            "name": "apply_directional_edit",
            "ok": True,
            "result": (
                "subtracted 1.0 x direction 'd' from (1024, 12) region "
                "(12288 weights changed) and reloaded the model from "
                "/tmp/edit_l12_a. Call measure_loop to test it."
            ),
        }
    ]
    assert module._edited_checkpoints(calls) == ["/tmp/edit_l12_a"]


def test_ignores_a_failed_edit(module) -> None:
    """A rejected edit changed nothing, so there is nothing to measure."""
    calls = [
        {
            "name": "apply_directional_edit",
            "ok": False,
            "result": "ERROR: no captured direction 'missing'",
        }
    ]
    assert module._edited_checkpoints(calls) == []


def test_ignores_a_failed_edit_that_still_mentions_a_path(module) -> None:
    """A failure message can name a path without having written it."""
    calls = [
        {
            "name": "apply_directional_edit",
            "ok": False,
            "result": "ERROR: could not write /tmp/edit_x",
        }
    ]
    assert module._edited_checkpoints(calls) == []


def test_ignores_other_tools(module) -> None:
    calls = [
        {"name": "measure_loop", "ok": True, "result": '{"tokens": 191}'},
        {"name": "capture_direction", "ok": True, "result": "norm 1.0000"},
    ]
    assert module._edited_checkpoints(calls) == []


def test_keeps_each_edit_once_in_order(module) -> None:
    """explore14 applied two edits; both must be measured, in order."""
    result = (
        "subtracted 1.0 x direction 'd' from (1024, 16) region "
        "(16384 weights changed) and reloaded the model from {path}. "
        "Call measure_loop to test it."
    )
    calls = [
        {"name": "apply_directional_edit", "ok": True, "result": result.format(path="/tmp/a")},
        {"name": "measure_loop", "ok": True, "result": "{}"},
        {"name": "apply_directional_edit", "ok": True, "result": result.format(path="/tmp/b")},
        {"name": "apply_directional_edit", "ok": True, "result": result.format(path="/tmp/a")},
    ]
    assert module._edited_checkpoints(calls) == ["/tmp/a", "/tmp/b"]


def test_returns_empty_when_no_edit_was_applied(module) -> None:
    assert module._edited_checkpoints([]) == []
