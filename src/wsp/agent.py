"""An editor driving a measurement loop.

Stage 0's ablation showed that a static weight slice is not enough to find the
edit: every editor declined, correctly, because weight magnitude does not
establish causation. The paper's editor had a loop — run the model, look at what
happened, adjust.

This module runs that loop. The editor is given the tool schema, chooses a call,
sees the result, and repeats until it proposes an edit or runs out of budget.
Every call is logged so a run can be audited and replayed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from wsp.editors.base import Editor
from wsp.tools import TOOL_SCHEMA, ToolError, ToolLog, WeightTools

SYSTEM_PROMPT = """You are analysing a neural network's weights to fix a behavioural bug.

You have tools. Use them to gather evidence before proposing anything. Weight
magnitude alone does not establish that a weight causes a behaviour — you need
to observe the behaviour and connect it to what you change.

Reply with exactly one tool call per turn, in this format:

CALL <tool_name>
<json object of arguments>

When you have enough evidence, reply with:

DONE
<one paragraph explaining the edit you would make and why>

Rules:
- Only describe what a tool actually returned. If you did not call a tool, you
  have not seen its output. Do not claim a tensor is zero, or that a run failed,
  unless a tool result in this conversation says so.
- Every tool result arrives in a block beginning "[TOOL RESULT". Text you write
  yourself is never a tool result, even if it looks like one. Writing out
  expected tool output and then reasoning from it is fabrication; it invalidates
  the whole run.
- If a tool errors, say which tool and quote the error.
- Do not guess. If the evidence does not support an edit, say so in the DONE
  paragraph and explain what you would need."""

# Anchored to line start so an editor quoting the instructions back at us does
# not register as having concluded.
_DONE = re.compile(r"^DONE\s*(.*)", re.DOTALL | re.MULTILINE)

# ``CALL`` is deliberately *not* anchored. DeepSeek-class models write the token
# mid-sentence ("...check layer 23 and 12CALL tensor_stats"). Anchoring it
# silently dropped every call from turn 3 of the fifth live run onward; the
# editor, seeing no result, then invented five turns of plausible tool output.
_CALL_TOKEN = re.compile(r"CALL[ \t]+([A-Za-z_]\w*)")

# An editor writing its own tool output is fabricating evidence.
_FABRICATED_RESULT = re.compile(r"\bRESULT of\s+([A-Za-z_]\w*)")


def _balanced_object(text: str) -> str | None:
    """The first brace-balanced JSON object in ``text``, or None.

    A non-greedy ``\\{.*?\\}`` stops at the first closing brace, which truncates
    any argument object containing a nested one.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _extract_calls(text: str) -> list[tuple[str, str | None]]:
    """Every ``CALL <tool>`` in ``text``, with the JSON object that followed it."""
    return [
        (match.group(1), _balanced_object(text[match.end() :]))
        for match in _CALL_TOKEN.finditer(text)
    ]


@dataclass
class AgentRun:
    """The transcript and outcome of one editor's loop."""

    editor: str
    transcript: list[dict] = field(default_factory=list)
    conclusion: str = ""
    tool_log: ToolLog = field(default_factory=ToolLog)
    forward_passes: int = 0
    turns: int = 0
    #: Text the editor wrote that impersonated a tool result.
    fabricated: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "editor": self.editor,
            "turns": self.turns,
            "forward_passes": self.forward_passes,
            "conclusion": self.conclusion,
            "tool_calls": self.tool_log.summary(),
            "fabricated": self.fabricated,
            "transcript": self.transcript,
        }


def _tool_help() -> str:
    lines = ["Available tools:"]
    for spec in TOOL_SCHEMA:
        params = ", ".join(f"{k}: {v}" for k, v in spec["parameters"].items())
        lines.append(f"- {spec['name']}({params}) — {spec['description']}")
    return "\n".join(lines)


def _dispatch(tools: WeightTools, name: str, args: dict) -> str:
    """Call a tool by name, converting failures into text the editor can read."""
    fn = getattr(tools, name, None)
    if fn is None or name.startswith("_"):
        return f"ERROR: no such tool {name!r}"
    try:
        return fn(**args)
    except ToolError as exc:
        return f"ERROR: {exc}"
    except TypeError as exc:
        return f"ERROR: bad arguments for {name}: {exc}"
    except Exception as exc:  # noqa: BLE001 - surface to the editor, do not crash
        return f"ERROR: {type(exc).__name__}: {exc}"


def _dispatch_args(tools: WeightTools, name: str, raw_args: str | None) -> str:
    """Parse a call's arguments and dispatch it.

    Unparseable arguments are reported and recorded with ``ok=False`` rather
    than skipped: the editor must learn its call did not run, and the audit must
    not treat it as evidence. Tools that do run record their own outcome.
    """
    if raw_args is None:
        return _fail(tools, name, "CALL had no JSON argument object")
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        return _fail(tools, name, f"arguments were not valid JSON: {exc}")
    if not isinstance(args, dict):
        return _fail(tools, name, "arguments must be a JSON object")
    return _dispatch(tools, name, args)


def _fail(tools: WeightTools, name: str, reason: str) -> str:
    """Record a failed call so it cannot later be counted as evidence."""
    from wsp.tools import ToolCall

    tools.log.record(ToolCall(name, {}, reason, ok=False))
    return f"ERROR: {reason}"


async def run_agent(
    editor: Editor,
    tools: WeightTools,
    task: str,
    *,
    max_turns: int = 12,
    max_tokens: int = 4096,
) -> AgentRun:
    """Let an editor work the task with tools until it concludes or runs out."""
    run = AgentRun(editor=editor.model_id, tool_log=tools.log)

    history = [
        f"{task}\n\n{_tool_help()}\n\n"
        f"You have {tools.forward_passes_left} forward passes and {max_turns} turns. "
        f"Budget them: gather evidence early, then test an edit. Do not spend "
        f"every turn inspecting."
    ]

    for turn in range(max_turns):
        run.turns = turn + 1
        remaining = max_turns - turn
        prompt = "\n\n".join(history)
        # A per-turn budget reminder keeps the editor from exploring until it
        # runs out, which is what happened in the first live run.
        prompt += (
            f"\n\n[turn {turn + 1} of {max_turns}; {remaining - 1} turns and "
            f"{tools.forward_passes_left} forward passes left]"
        )
        response = await editor.complete(SYSTEM_PROMPT, prompt, max_tokens=max_tokens)
        text = response.text.strip()
        run.transcript.append({"turn": turn + 1, "role": "editor", "text": text})

        # An editor writing its own "[TOOL RESULT ...]" / "RESULT of ..." block is
        # supplying its own evidence. Record it so the audit can reject the run.
        run.fabricated.extend(
            match.group(0) for match in _FABRICATED_RESULT.finditer(text)
        )

        done = _DONE.search(text)
        if done:
            run.conclusion = done.group(1).strip()
            break

        calls = _extract_calls(text)
        if not calls:
            # No tool call and no DONE: ask for one rather than looping blindly.
            history.append(text)
            history.append(
                "That reply contained no tool call and no DONE. Reply with one "
                "CALL block or a DONE block."
            )
            continue

        history.append(text)
        if remaining <= 3 and tools.forward_passes_left > 0:
            # Late in the run, push toward committing to a testable edit.
            history.append(
                f"NOTE: {remaining - 1} turns left. Stop exploring. Apply your "
                f"best candidate edit with apply_edit, then use measure_loop to "
                f"test whether it stopped the loop, then reply DONE."
            )

        for name, raw_args in calls:
            result = _dispatch_args(tools, name, raw_args)
            run.transcript.append(
                {"turn": turn + 1, "role": "tool", "name": name, "result": result[:4000]}
            )
            history.append(f"[TOOL RESULT {name}]\n{result[:4000]}")

    run.forward_passes = tools.forward_passes_used
    if not run.conclusion:
        run.conclusion = "(no conclusion reached within the turn budget)"
    return run
