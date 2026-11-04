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

#: Phrases asserting something about tensor contents.
_TENSOR_CLAIM = re.compile(
    r"\b(all zeros?|all zero|every (?:tensor|column|value)|norm\s*=\s*0|"
    r"renders? as (?:all )?zero|degenerate|no non-?zero)\b",
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


def audit_conclusion(conclusion: str, tool_calls: list[dict]) -> AuditResult:
    """Check a conclusion against the tools that were actually called."""
    called = [c["name"] for c in tool_calls]
    result = AuditResult(tools_called=called)

    if not conclusion:
        return result

    if _TENSOR_CLAIM.search(conclusion) and not (set(called) & DATA_TOOLS):
        match = _TENSOR_CLAIM.search(conclusion)
        result.findings.append(
            AuditFinding(
                kind="tensor-data",
                claim=match.group(0),
                reason=(
                    "claims about tensor contents require read_slice or "
                    f"tensor_stats; called: {sorted(set(called)) or 'none'}"
                ),
            )
        )

    if _BEHAVIOUR_CLAIM.search(conclusion) and not (set(called) & BEHAVIOUR_TOOLS):
        match = _BEHAVIOUR_CLAIM.search(conclusion)
        result.findings.append(
            AuditFinding(
                kind="behaviour",
                claim=match.group(0),
                reason=(
                    "claims about behaviour require run_model or measure_loop; "
                    f"called: {sorted(set(called)) or 'none'}"
                ),
            )
        )

    return result


def audit_run(run_dict: dict) -> AuditResult:
    """Audit a serialized AgentRun."""
    return audit_conclusion(
        run_dict.get("conclusion", ""), run_dict.get("tool_calls", [])
    )
