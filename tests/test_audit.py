"""Tests for the conclusion auditor.

The motivating case: an editor claimed every tensor was zero after calling only
list_tensors. That must be flagged.
"""

from __future__ import annotations

from wsp.audit import audit_conclusion, audit_run


def _calls(*names: str) -> list[dict]:
    return [{"name": n, "args": {}, "ok": True, "result": ""} for n in names]


def test_flags_tensor_claim_without_data_tool() -> None:
    result = audit_conclusion(
        "every tensor renders as all zeros", _calls("list_tensors")
    )
    assert not result.clean
    assert result.findings[0].kind == "tensor-data"


def test_accepts_tensor_claim_with_data_tool() -> None:
    result = audit_conclusion(
        "the column norms are all zero", _calls("list_tensors", "tensor_stats")
    )
    assert result.clean


def test_flags_behaviour_claim_without_behaviour_tool() -> None:
    result = audit_conclusion(
        "the model looped on every prompt", _calls("list_tensors")
    )
    assert not result.clean
    assert result.findings[0].kind == "behaviour"


def test_accepts_behaviour_claim_with_measure_loop() -> None:
    result = audit_conclusion(
        "the model looped on the first prompt", _calls("measure_loop")
    )
    assert result.clean


def test_flags_both_kinds_independently() -> None:
    result = audit_conclusion(
        "every tensor is zero and the model looped", _calls("list_tensors")
    )
    kinds = {f.kind for f in result.findings}
    assert kinds == {"tensor-data", "behaviour"}


def test_clean_conclusion_produces_no_findings() -> None:
    result = audit_conclusion(
        "I would subtract a direction from columns 0-63.", _calls("read_slice")
    )
    assert result.clean


def test_empty_conclusion_is_clean() -> None:
    assert audit_conclusion("", []).clean


def test_summary_names_the_missing_tool() -> None:
    result = audit_conclusion("all zeros", _calls("list_tensors"))
    text = result.summary()
    assert "tensor-data" in text
    assert "read_slice" in text


def test_audit_run_reads_serialized_dict() -> None:
    run = {
        "conclusion": "every column is zero",
        "tool_calls": _calls("list_tensors"),
    }
    assert not audit_run(run).clean


def test_audit_run_clean_when_tools_support_claim() -> None:
    run = {
        "conclusion": "the model looped",
        "tool_calls": _calls("measure_loop"),
    }
    assert audit_run(run).clean


def test_errored_data_tool_does_not_license_a_tensor_claim() -> None:
    # The editor called tensor_stats but it raised KeyError, so it saw nothing.
    calls = [
        {
            "name": "tensor_stats",
            "args": {},
            "ok": True,
            "result": "ERROR: KeyError: 'model.layers.0.mlp.down_proj.weight' not in checkpoint",
        }
    ]
    result = audit_conclusion("column norms peak at col 2011", calls)
    assert not result.clean
    assert result.findings[0].kind == "tensor-data"


def test_errored_behaviour_tool_does_not_license_a_behaviour_claim() -> None:
    calls = [
        {"name": "measure_loop", "args": {}, "ok": True, "result": "ERROR: no model attached"}
    ]
    result = audit_conclusion("the model looped", calls)
    assert not result.clean
    assert result.findings[0].kind == "behaviour"


def test_successful_call_alongside_a_failure_still_licenses() -> None:
    calls = [
        {"name": "tensor_stats", "args": {}, "ok": True, "result": "ERROR: KeyError"},
        {"name": "tensor_stats", "args": {}, "ok": True, "result": "col\tnorm\n3\t0.29"},
    ]
    assert audit_conclusion("column norms peak at col 3", calls).clean
