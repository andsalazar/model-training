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


def test_accepts_activation_claim_with_capture_tool() -> None:
    result = audit_conclusion(
        "the activation norms peak at dim 2011", _calls("capture_activations")
    )
    assert result.clean, "a real capture call licenses an activation claim"


def test_flags_activation_claim_without_capture_tool() -> None:
    """Quoting activation statistics without capturing them is fabrication.

    explore7's editor correctly reported that weight statistics could not
    localise the cause. An editor that instead invents activation numbers would
    be claiming evidence it never gathered. A tensor_stats call is not that
    evidence: it describes the weights, not what the module produced.
    """
    result = audit_conclusion(
        "the activation norms peak at dim 2011", _calls("tensor_stats")
    )
    assert not result.clean
    assert result.findings[0].kind == "activation-data"


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


def test_flags_loop_verdict_no_tool_observed() -> None:
    """Regression for the sixth live run's central false claim.

    The conclusion said the edit produced "looped: false" while every real
    measure_loop in that run returned "looped": true. The presence check passed
    it because measure_loop had been called at all.
    """
    calls = [
        {
            "name": "measure_loop",
            "args": {},
            "ok": True,
            "result": '{"looped": true, "hit_cap": true, "tokens": 128}',
        }
    ]
    result = audit_conclusion(
        "After the edit, measure_loop returned `looped: false`, 61 tokens.", calls
    )
    assert not result.clean
    assert any(f.kind == "contradicted-verdict" for f in result.findings)


def test_accepts_loop_verdict_a_tool_actually_observed() -> None:
    calls = [
        {
            "name": "measure_loop",
            "args": {},
            "ok": True,
            "result": '{"looped": false, "hit_cap": false, "tokens": 61}',
        }
    ]
    assert audit_conclusion("measure_loop returned looped: false", calls).clean


def test_contradicted_verdict_ignores_errored_calls() -> None:
    calls = [
        {"name": "measure_loop", "args": {}, "ok": False, "result": "ERROR: no JSON"},
        {
            "name": "measure_loop",
            "args": {},
            "ok": True,
            "result": '{"looped": true, "tokens": 128}',
        },
    ]
    result = audit_conclusion("the edit gave looped: false", calls)
    assert any(f.kind == "contradicted-verdict" for f in result.findings)


def test_flags_edit_claim_when_the_edit_call_errored() -> None:
    """A conclusion reporting an applied edit needs a successful edit call.

    The smoke test caught this: apply_directional_edit errored on bad JSON while
    the conclusion still said the edit was applied. The audit passed it because
    no check covered edit claims.
    """
    calls = [
        {"name": "capture_direction", "args": {}, "ok": True, "result": "direction 'd'"},
        {
            "name": "apply_directional_edit",
            "args": {},
            "ok": False,
            "result": "ERROR: arguments were not valid JSON",
        },
    ]
    result = audit_conclusion(
        "I applied the edit to columns 0:4 and saved the checkpoint.", calls
    )
    assert not result.clean
    assert result.findings[0].kind == "edit-claim"
    assert "failed" in result.findings[0].reason


def test_flags_edit_claim_when_no_edit_tool_was_called() -> None:
    result = audit_conclusion(
        "I applied the edit to columns 0:4.", _calls("tensor_stats")
    )
    assert not result.clean
    assert result.findings[0].kind == "edit-claim"
    assert "no edit tool was called" in result.findings[0].reason


def test_accepts_edit_claim_with_a_successful_edit() -> None:
    calls = [
        {
            "name": "apply_directional_edit",
            "args": {},
            "ok": True,
            "result": "subtracted 0.5 x direction 'd' ... and reloaded",
        }
    ]
    result = audit_conclusion("I applied the edit to columns 0:4.", calls)
    assert result.clean


def test_audit_run_flags_bracket_framed_fabricated_evidence() -> None:
    """Regression for the sixth live run.

    That run's editor wrote 29 "[TOOL RESULT <tool>]" blocks itself. The
    detector matched only "RESULT of <tool>", so ``fabricated`` came back empty
    and the audit declared the run clean even though its central claim — a
    layer-16 column edit that stopped the loop — never happened.
    """
    run = {
        "conclusion": "subtracting 0.02 from layer 16 column 2949 stopped the loop",
        "tool_calls": _calls("tensor_stats", "measure_loop"),
        "fabricated": ["[TOOL RESULT read_slice]"],
    }
    result = audit_run(run)
    assert not result.clean
    assert any(f.kind == "fabricated-evidence" for f in result.findings)
