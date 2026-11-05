"""Check an editor's conclusion against what its tools actually returned.

A smoke test caught an editor claiming "every tensor renders as all zeros" when
it had only ever called ``list_tensors``. That kind of fabrication would
silently corrupt a research result, so conclusions are audited rather than
trusted.

The audit is deliberately conservative: it flags claims it cannot corroborate,
and it does not try to judge whether a *correct* claim was well reasoned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Tools whose results contain numeric tensor data.
DATA_TOOLS = frozenset({"read_slice", "tensor_stats"})

#: Tools that observe behaviour.
BEHAVIOUR_TOOLS = frozenset({"run_model", "measure_loop"})

#: Phrases asserting something about tensor contents. Covers both qualitative
#: claims ("all zeros") and specific numeric ones ("column norms peak at col
#: 2011", "std 0.0314"), because an editor can fabricate either.
_TENSOR_CLAIM = re.compile(
    r"\b(all zeros?|all zero|every (?:tensor|column|value)|norm\s*=\s*0|"
    r"renders? as (?:all )?zero|degenerate|no non-?zero)\b"
    r"|\b(?:column|col|norm|std|mean)s?\b[^.]{0,40}?\b\d+(?:\.\d+)?\b",
    re.IGNORECASE,
)

#: Phrases asserting something about model behaviour.
_BEHAVIOUR_CLAIM = re.compile(
    r"\b(looped|looping|did not loop|failed to (?:run|execute)|"
    r"no baseline generation|never observed)\b",
    re.IGNORECASE,
)


@dataclass
class AuditFinding:
    """One unsupported claim."""

    kind: str
    claim: str
    reason: str


@dataclass
class AuditResult:
    """The outcome of auditing one run."""

    findings: list[AuditFinding] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.findings

    def summary(self) -> str:
        if self.clean:
            return f"no unsupported claims ({len(self.tools_called)} tool calls)"
        lines = [f"{len(self.findings)} unsupported claim(s):"]
        for f in self.findings:
            lines.append(f"  [{f.kind}] {f.claim!r} — {f.reason}")
        return "\n".join(lines)


def _succeeded(tool_calls: list[dict], names: frozenset[str]) -> bool:
    """Whether any of ``names`` was called *and* returned data.

    A call that errored produced no evidence, so it must not license a claim.
    An editor that called tensor_stats and got a KeyError has seen nothing.
    """
    for call in tool_calls:
        if call.get("name") not in names:
            continue
        if not call.get("ok", True):
            continue
        result = str(call.get("result", ""))
        if result.startswith("ERROR"):
            continue
        return True
    return False


def audit_conclusion(conclusion: str, tool_calls: list[dict]) -> AuditResult:
    """Check a conclusion against the tools that actually returned data."""
    called = [c["name"] for c in tool_calls]
    result = AuditResult(tools_called=called)

    if not conclusion:
        return result

    if _TENSOR_CLAIM.search(conclusion) and not _succeeded(tool_calls, DATA_TOOLS):
        match = _TENSOR_CLAIM.search(conclusion)
        result.findings.append(
            AuditFinding(
                kind="tensor-data",
                claim=match.group(0),
                reason=(
                    "claims about tensor contents require a successful "
                    f"read_slice or tensor_stats; called: {sorted(set(called)) or 'none'}"
                ),
            )
        )

    if _BEHAVIOUR_CLAIM.search(conclusion) and not _succeeded(tool_calls, BEHAVIOUR_TOOLS):
        match = _BEHAVIOUR_CLAIM.search(conclusion)
        result.findings.append(
            AuditFinding(
                kind="behaviour",
                claim=match.group(0),
                reason=(
                    "claims about behaviour require a successful run_model or "
                    f"measure_loop; called: {sorted(set(called)) or 'none'}"
                ),
            )
        )

    return result


def audit_run(run_dict: dict) -> AuditResult:
    """Audit a serialized AgentRun."""
    result = audit_conclusion(
        run_dict.get("conclusion", ""), run_dict.get("tool_calls", [])
    )

    # An editor that wrote its own tool output supplied its own evidence. The
    # sixth live run produced five turns of invented tensor_stats output
    # (complete with a JSON schema that exists in no commit) after a parser bug
    # swallowed its real calls, so this is a real failure mode, not a
    # hypothetical.
    fabricated = run_dict.get("fabricated") or []
    if fabricated:
        result.findings.append(
            AuditFinding(
                kind="fabricated-evidence",
                claim=fabricated[0],
                reason=(
                    f"{len(fabricated)} tool result(s) written by the editor "
                    "itself rather than returned by a tool"
                ),
            )
        )

    return result
