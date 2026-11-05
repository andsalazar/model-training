"""Tests for the editor loop, using a scripted editor and fake tools."""

from __future__ import annotations

import numpy as np
import pytest

from wsp.agent import run_agent
from wsp.editors.base import Editor, EditorResponse, Usage
from wsp.tools import WeightTools


class ScriptedEditor(Editor):
    """Returns a fixed sequence of replies, one per turn."""

    model_id = "test/scripted"

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.prompts: list[str] = []

    async def complete(self, system, user, *, temperature=0.0, max_tokens=8192):
        self.prompts.append(user)
        text = self._replies.pop(0) if self._replies else "DONE\nout of replies"
        return EditorResponse(text=text, usage=Usage(), model=self.model_id)


class FakeStore:
    keys = ["w"]

    def get(self, name):
        return np.ones((4, 8), dtype=np.float32)

    def info(self, name):
        class I:
            shape = (4, 8)
            dtype = "BF16"

        return I()


class FakeRunner:
    def generate(self, prompt, max_tokens=512):
        return "generated"

    def measure(self, prompt, max_tokens=2048):
        return {"looped": True, "tokens": 100, "hit_cap": False}


def _tools() -> WeightTools:
    return WeightTools(FakeStore(), FakeRunner())


class FakeCapturer:
    """Returns a canned activation array per prompt."""

    def __init__(self, by_prompt: dict[str, np.ndarray]):
        self.by_prompt = by_prompt
        self.calls: list[tuple[str, str, int]] = []

    def capture(self, prompt: str, module_path: str, max_tokens: int = 64):
        self.calls.append((prompt, module_path, max_tokens))
        return self.by_prompt[prompt]

    def reload(self, model_path: str) -> None:
        pass


@pytest.mark.asyncio
async def test_agent_stops_on_done() -> None:
    editor = ScriptedEditor(["DONE\nThe edit is to zero column 3."])
    run = await run_agent(editor, _tools(), "fix the loop")

    assert "zero column 3" in run.conclusion
    assert run.turns == 1


@pytest.mark.asyncio
async def test_agent_executes_a_tool_call() -> None:
    editor = ScriptedEditor(
        [
            'CALL list_tensors\n{"pattern": ""}',
            "DONE\nconcluded",
        ]
    )
    run = await run_agent(editor, _tools(), "fix the loop")

    names = [c["name"] for c in run.tool_log.summary()]
    assert names == ["list_tensors"]
    assert run.turns == 2


@pytest.mark.asyncio
async def test_agent_feeds_tool_result_back() -> None:
    editor = ScriptedEditor(
        [
            'CALL run_model\n{"prompt": "2+2?"}',
            "DONE\nconcluded",
        ]
    )
    await run_agent(editor, _tools(), "fix the loop")

    # The second prompt must contain the tool's output, framed so the editor
    # cannot mistake its own prose for a result.
    assert "[TOOL RESULT run_model]" in editor.prompts[1]
    assert "generated" in editor.prompts[1]


@pytest.mark.asyncio
async def test_agent_reports_unknown_tool_without_crashing() -> None:
    editor = ScriptedEditor(
        [
            'CALL no_such_tool\n{}',
            "DONE\nconcluded",
        ]
    )
    run = await run_agent(editor, _tools(), "fix the loop")

    tool_turns = [t for t in run.transcript if t["role"] == "tool"]
    assert "no such tool" in tool_turns[0]["result"]


@pytest.mark.asyncio
async def test_agent_reports_bad_json_without_crashing() -> None:
    editor = ScriptedEditor(
        [
            "CALL list_tensors\n{not json}",
            "DONE\nconcluded",
        ]
    )
    run = await run_agent(editor, _tools(), "fix the loop")

    tool_turns = [t for t in run.transcript if t["role"] == "tool"]
    assert "not valid JSON" in tool_turns[0]["result"]


@pytest.mark.asyncio
async def test_agent_nudges_when_reply_has_no_call() -> None:
    editor = ScriptedEditor(
        [
            "I am thinking about it.",
            "DONE\nconcluded",
        ]
    )
    run = await run_agent(editor, _tools(), "fix the loop")

    assert "no tool call" in editor.prompts[1]
    assert run.turns == 2


@pytest.mark.asyncio
async def test_agent_stops_at_turn_budget() -> None:
    editor = ScriptedEditor(['CALL list_tensors\n{"pattern": ""}'] * 10)
    run = await run_agent(editor, _tools(), "fix the loop", max_turns=3)

    assert run.turns == 3
    assert "no conclusion" in run.conclusion


@pytest.mark.asyncio
async def test_agent_records_forward_passes() -> None:
    editor = ScriptedEditor(
        [
            'CALL run_model\n{"prompt": "a"}',
            'CALL run_model\n{"prompt": "b"}',
            "DONE\nconcluded",
        ]
    )
    run = await run_agent(editor, _tools(), "fix the loop")
    assert run.forward_passes == 2


@pytest.mark.asyncio
async def test_agent_ignores_done_quoted_mid_line() -> None:
    # An editor echoing the instructions must not be read as having concluded.
    editor = ScriptedEditor(
        [
            'The format is "DONE\\n<paragraph>". Let me look first.\n'
            'CALL list_tensors\n{"pattern": ""}',
            "DONE\nreal conclusion",
        ]
    )
    run = await run_agent(editor, _tools(), "fix the loop")

    assert run.conclusion == "real conclusion"
    assert run.turns == 2

@pytest.mark.asyncio
async def test_agent_reports_turn_budget_in_prompt() -> None:
    editor = ScriptedEditor(["DONE\nconcluded"])
    await run_agent(editor, _tools(), "fix the loop", max_turns=7)

    assert "turn 1 of 7" in editor.prompts[0]
    assert "6 turns" in editor.prompts[0]


@pytest.mark.asyncio
async def test_agent_nudges_toward_testing_late_in_run() -> None:
    # Three turns of exploration, then the budget is nearly gone.
    editor = ScriptedEditor(
        [
            'CALL list_tensors\n{"pattern": ""}',
            'CALL list_tensors\n{"pattern": ""}',
            'CALL list_tensors\n{"pattern": ""}',
            "DONE\nconcluded",
        ]
    )
    await run_agent(editor, _tools(), "fix the loop", max_turns=4)

    # The third prompt should carry the commit-to-a-test nudge.
    assert any("Stop exploring" in p for p in editor.prompts)


@pytest.mark.asyncio
async def test_agent_serializes_to_dict() -> None:
    editor = ScriptedEditor(["DONE\nconcluded"])
    run = await run_agent(editor, _tools(), "fix the loop")
    d = run.as_dict()

    assert d["editor"] == "test/scripted"
    assert d["conclusion"] == "concluded"
    assert isinstance(d["tool_calls"], list)


@pytest.mark.asyncio
async def test_agent_runs_a_call_written_mid_line() -> None:
    # Regression: the fifth live run's editor wrote "CALL tensor_stats" inline,
    # immediately after preceding prose with no newline. A line-anchored regex
    # dropped every one of those calls, and the editor then invented the
    # results. The call must actually execute.
    editor = ScriptedEditor(
        [
            'Let me check layer 23 and 12CALL tensor_stats\n{"tensor": "w"}',
            "DONE\nconcluded",
        ]
    )
    run = await run_agent(editor, _tools(), "fix the loop")

    names = [c["name"] for c in run.tool_log.summary()]
    assert names == ["tensor_stats"]


@pytest.mark.asyncio
async def test_agent_runs_several_calls_in_one_turn() -> None:
    editor = ScriptedEditor(
        [
            'CALL list_tensors\n{"pattern": ""}\n\nCALL run_model\n{"prompt": "x"}',
            "DONE\nconcluded",
        ]
    )
    run = await run_agent(editor, _tools(), "fix the loop")

    names = [c["name"] for c in run.tool_log.summary()]
    assert names == ["list_tensors", "run_model"]
    assert run.forward_passes == 1


@pytest.mark.asyncio
async def test_agent_parses_arguments_with_nested_braces() -> None:
    # A non-greedy .*? would stop at the first inner closing brace.
    editor = ScriptedEditor(
        [
            'CALL run_model\n{"prompt": "value is {a: 1} ok"}',
            "DONE\nconcluded",
        ]
    )
    run = await run_agent(editor, _tools(), "fix the loop")

    call = run.tool_log.summary()[0]
    assert call["name"] == "run_model"
    assert call["args"]["prompt"] == "value is {a: 1} ok"


@pytest.mark.asyncio
async def test_agent_flags_an_editor_that_writes_its_own_tool_output() -> None:
    editor = ScriptedEditor(
        [
            'CALL list_tensors\n{"pattern": ""}\n\n'
            "RESULT of tensor_stats:\n" '{"top_norms": [{"col": 1180, "norm": 12.4}]}',
            "DONE\ncolumns 1180 and 1204 have norm 12.4",
        ]
    )
    run = await run_agent(editor, _tools(), "fix the loop")

    assert run.fabricated
    assert any("tensor_stats" in f for f in run.fabricated)


@pytest.mark.asyncio
async def test_agent_flags_editor_using_the_real_tool_result_framing() -> None:
    """The sixth live run impersonated the harness's own "[TOOL RESULT x]" block.

    The detector originally matched only "RESULT of <tool>", so 29 fabricated
    blocks in that run went undetected and the audit reported it clean. The
    bracket form is the *real* framing, which makes it the more dangerous one
    to miss.
    """
    editor = ScriptedEditor(
        [
            'CALL read_slice\n{"tensor": "w", "rows": "0:8", "cols": "2949:2950"}\n\n'
            "[TOOL RESULT read_slice]\n"
            "row\t2949\n0\t+0.0201\n1\t+0.0199\n",
            "DONE\ncolumn 2949 is a near-constant positive column at +0.0200",
        ]
    )
    run = await run_agent(editor, _tools(), "fix the loop")

    assert run.fabricated, "bracket-framed fabricated result was not detected"
    assert any("read_slice" in f for f in run.fabricated)


@pytest.mark.asyncio
async def test_agent_does_not_flag_a_genuine_tool_result() -> None:
    """A real tool result is appended by the harness, never written by the editor."""
    editor = ScriptedEditor(
        [
            'CALL read_slice\n{"tensor": "w", "rows": "0:2", "cols": "0:2"}',
            "DONE\nread the slice",
        ]
    )
    run = await run_agent(editor, _tools(), "fix the loop")

    assert not run.fabricated
    # The harness did append a real result to the transcript.
    assert any(t.get("role") == "tool" for t in run.transcript)


# -- activation tools through dispatch -------------------------------------


def _capture_tools(by_prompt: dict[str, np.ndarray]) -> WeightTools:
    return WeightTools(FakeStore(), FakeRunner(), capturer=FakeCapturer(by_prompt))


@pytest.mark.asyncio
async def test_agent_dispatches_capture_activations() -> None:
    """The editor must be able to reach the capture tools through dispatch."""
    acts = np.ones((4, 8), dtype=np.float32)
    acts[:, 2] = 5.0
    editor = ScriptedEditor(
        [
            'CALL capture_activations\n{"tensor": "w", "prompt": "loop"}',
            "DONE\ncaptured",
        ]
    )
    run = await run_agent(editor, _capture_tools({"loop": acts}), "fix the loop")

    names = [c["name"] for c in run.tool_log.summary()]
    assert names == ["capture_activations"]
    tool_turns = [t for t in run.transcript if t.get("role") == "tool"]
    assert "activations" in tool_turns[0]["result"]
    assert "ERROR" not in tool_turns[0]["result"]


@pytest.mark.asyncio
async def test_agent_dispatches_capture_direction_then_aligned_columns() -> None:
    """A two-call sequence: capture a direction, then rank columns by it."""
    looping = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    control = np.zeros((1, 4), dtype=np.float32)
    editor = ScriptedEditor(
        [
            'CALL capture_direction\n{"name": "d", "tensor": "w", '
            '"looping_prompt": "loop", "control_prompt": "calm"}',
            'CALL aligned_columns\n{"tensor": "w", "direction": "d", "k": 3}',
            "DONE\nranked",
        ]
    )
    run = await run_agent(
        editor, _capture_tools({"loop": looping, "calm": control}), "fix the loop"
    )

    names = [c["name"] for c in run.tool_log.summary()]
    assert names == ["capture_direction", "aligned_columns"]
    tool_turns = [t for t in run.transcript if t.get("role") == "tool"]
    assert "direction 'd'" in tool_turns[0]["result"]
    assert "alignment with 'd'" in tool_turns[1]["result"]


@pytest.mark.asyncio
async def test_agent_reports_capture_without_a_capturer() -> None:
    """Without a capturer the tool errors rather than pretending to measure."""
    editor = ScriptedEditor(
        [
            'CALL capture_activations\n{"tensor": "w", "prompt": "loop"}',
            "DONE\ncould not capture",
        ]
    )
    run = await run_agent(editor, _tools(), "fix the loop")

    tool_turns = [t for t in run.transcript if t.get("role") == "tool"]
    assert "no activation capturer" in tool_turns[0]["result"]
    assert run.tool_log.calls[0].ok is False


@pytest.mark.asyncio
async def test_agent_prompt_mentions_activation_capture() -> None:
    """The editor must be told capture exists, or it will never use it.

    explore7's editor correctly concluded weight statistics could not localise
    the cause and stopped. It had no alternative because nothing told it one
    existed.
    """
    editor = ScriptedEditor(["DONE\ndone"])
    await run_agent(editor, _tools(), "fix the loop")

    prompt = editor.prompts[0]
    assert "capture_direction" in prompt
    assert "aligned_columns" in prompt
    assert "capture_activations" in prompt
