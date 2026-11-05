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

#: Tools whose results contain numeric *weight* data.
DATA_TOOLS = frozenset({"read_slice", "tensor_stats"})

#: Tools whose results contain numeric *activation* data. Kept separate from
#: DATA_TOOLS because the two are different evidence: a tensor_stats call says
#: nothing about what a module produces on a prompt, so it must not license a
#: conclusion that quotes activation statistics.
ACTIVATION_TOOLS = frozenset(
    {
        "capture_activations",
        "capture_direction",
        "aligned_columns",
        "list_directions",
        "direction_similarity",
    }
)

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

#: Tools that mutate the checkpoint.
EDIT_TOOLS = frozenset({"apply_edit", "apply_directional_edit"})

#: Phrases asserting that an edit was actually applied. A conclusion that says
#: it applied an edit needs a successful edit call behind it: the smoke test
#: caught a run whose apply_directional_edit had errored on bad JSON while the
#: conclusion still reported the edit as applied.
_EDIT_CLAIM = re.compile(
    r"\b(applied|made|performed|wrote|saved|committed)\b[^.]{0,40}?"
    r"\b(edit|change|modification|subtraction)\b"
    r"|\bedit (?:was )?(?:applied|made|written|saved)\b",
    re.IGNORECASE,
)

#: Phrases asserting a *quantitative* claim about captured activations.
#:
#: Deliberately narrow, because a bare mention of "activations" is not a claim.
#: explore2 correctly reported that its tools "exposed no activations"; explore3
#: proposed a "residual-space direction" as a future edit; explore7 noted that a
#: direction would need "activations rather than weight magnitude". None of those
#: asserts a measured activation value, so none should be flagged. What must be
#: caught is a conclusion quoting activation statistics it never captured, so the
#: activation word has to be followed by a statistical term or a number.
#:
#: "mean" and "magnitude" are deliberately absent: both are ordinary English
#: words ("a mean move", "weight magnitude") and matching them produced false
#: positives on real runs.
_ACTIVATION_CLAIM = re.compile(
    r"\b(?:activations?|hidden states?|residual stream|contrast_cosine|"
    r"aligned columns?)\b[^.]{0,60}?"
    r"\b(?:norms?|std|variance|cosine|peaks?|\d+(?:\.\d+)?)\b",
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

    # A numeric claim is licensed by any real observation: weight statistics,
    # activation statistics, or an edit call (which reports how many weights it
    # changed). The activation check below is an *additional* requirement for
    # claims specifically about activations, not a replacement for this one.
    numeric_tools = DATA_TOOLS | ACTIVATION_TOOLS | EDIT_TOOLS
    if _TENSOR_CLAIM.search(conclusion) and not _succeeded(tool_calls, numeric_tools):
        match = _TENSOR_CLAIM.search(conclusion)
        result.findings.append(
            AuditFinding(
                kind="tensor-data",
                claim=match.group(0),
                reason=(
                    "claims about tensor contents require a successful "
                    "read_slice, tensor_stats or capture call; called: "
                    f"{sorted(set(called)) or 'none'}"
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

    # Activation statistics are a different observation from weight statistics.
    # A tensor_stats call describes the weights and says nothing about what the
    # module produced on a prompt, so it cannot license a claim about activations.
    if _ACTIVATION_CLAIM.search(conclusion) and not _succeeded(
        tool_calls, ACTIVATION_TOOLS
    ):
        match = _ACTIVATION_CLAIM.search(conclusion)
        result.findings.append(
            AuditFinding(
                kind="activation-data",
                claim=match.group(0),
                reason=(
                    "claims about activations require a successful "
                    "capture_activations or capture_direction; called: "
                    f"{sorted(set(called)) or 'none'}"
                ),
            )
        )

    # A conclusion can cite a tool that *was* called and still misreport what it
    # returned. The sixth live run claimed "looped: false" after its edit while
    # every successful measure_loop in the same run returned "looped": true; the
    # presence check above passed it because measure_loop had been called. Check
    # the claimed verdict against the verdicts actually observed.
    for finding in _contradicted_verdicts(conclusion, tool_calls):
        result.findings.append(finding)

    # A conclusion can also cite a tool's *number* while misquoting it. The
    # checks above only ask whether a tool was called; explore13's conclusion
    # reported a direction_similarity of 0.9995 when the tool returned -0.0485,
    # and a held-out "24 tokens" that no measurement produced.
    for finding in _misquoted_numbers(conclusion, tool_calls):
        result.findings.append(finding)

    # Claiming an edit was applied when the edit call errored is fabrication of
    # a different kind: the conclusion reports a mutation that never happened.
    if _EDIT_CLAIM.search(conclusion) and not _succeeded(tool_calls, EDIT_TOOLS):
        match = _EDIT_CLAIM.search(conclusion)
        attempted = [c for c in tool_calls if c.get("name") in EDIT_TOOLS]
        if attempted:
            reason = (
                "conclusion reports an edit as applied but every edit call "
                f"failed; called: {sorted({c['name'] for c in attempted})}"
            )
        else:
            reason = (
                "conclusion reports an edit as applied but no edit tool was "
                f"called; called: {sorted(set(called)) or 'none'}"
            )
        result.findings.append(
            AuditFinding(kind="edit-claim", claim=match.group(0), reason=reason)
        )

    return result


#: A claimed loop verdict in the conclusion, e.g. "looped: false" or
#: "`looped: false`". Matched loosely so markdown backticks and spacing do not
#: hide a misreport.
_CLAIMED_LOOP_VERDICT = re.compile(
    r"looped\W{0,3}(true|false)", re.IGNORECASE
)

#: Metric names a tool reports, paired with the tools whose results carry them.
#: Ordered longest-first so "contrast_cosine" is matched before "cosine".
_METRIC_TOOLS: tuple[tuple[str, frozenset[str]], ...] = (
    ("contrast_cosine", ACTIVATION_TOOLS),
    ("direction_similarity", ACTIVATION_TOOLS),
    ("cosine", ACTIVATION_TOOLS),
)

#: A decimal number quoted next to a metric name, e.g. "cosine 0.9995".
_QUOTED_DECIMAL = re.compile(r"-?\d+\.\d+")

#: A token count quoted next to the word "tokens", e.g. "191 tokens".
_QUOTED_TOKEN_COUNT = re.compile(r"\b(\d+)\s*tokens\b", re.IGNORECASE)


def _real_results(tool_calls: list[dict], names: frozenset[str]) -> str:
    """Concatenated results of the successful calls to ``names``."""
    parts: list[str] = []
    for call in tool_calls:
        if call.get("name") not in names:
            continue
        if not call.get("ok", True):
            continue
        result = str(call.get("result", ""))
        if result.startswith("ERROR"):
            continue
        parts.append(result)
    return "\n".join(parts).replace(",", "")


def _misquoted_numbers(conclusion: str, tool_calls: list[dict]) -> list[AuditFinding]:
    """Findings for numbers the conclusion attributes to a tool but misquotes.

    explore13's conclusion reported "direction_similarity between them was
    0.9995" while the tool returned -0.0485, and "a held-out question (17x23)
    still answers correctly in 24 tokens" when no such measurement was ever
    made. Both are the same failure: the conclusion cites a tool's number
    without that number appearing in the tool's output. The presence checks
    above cannot see it, because the tool really was called and really did
    succeed — only the value is invented.
    """
    findings: list[AuditFinding] = []
    seen: set[str] = set()

    for metric, names in _METRIC_TOOLS:
        results = _real_results(tool_calls, names)
        for match in re.finditer(re.escape(metric), conclusion, re.IGNORECASE):
            window = conclusion[match.end() : match.end() + 80]
            for number in _QUOTED_DECIMAL.findall(window):
                if number in seen:
                    continue
                if results and number.lstrip("-") in results:
                    continue
                seen.add(number)
                reason = (
                    "conclusion quotes a value the tool did not return; "
                    f"real results: {results[:200]!r}"
                    if results
                    else "conclusion quotes a value from a tool that never "
                    "returned a result"
                )
                findings.append(
                    AuditFinding(
                        kind="misquoted-number",
                        claim=f"{metric} {number}",
                        reason=reason,
                    )
                )

    behaviour = _real_results(tool_calls, BEHAVIOUR_TOOLS)
    if behaviour:
        for match in _QUOTED_TOKEN_COUNT.finditer(conclusion):
            number = match.group(1)
            if number in seen:
                continue
            if re.search(rf"\b{number}\b", behaviour):
                continue
            seen.add(number)
            findings.append(
                AuditFinding(
                    kind="misquoted-number",
                    claim=f"{number} tokens",
                    reason=(
                        "conclusion quotes a token count no measure_loop or "
                        f"run_model returned; real results: {behaviour[:200]!r}"
                    ),
                )
            )

    return findings

#: A loop verdict inside a real measure_loop result.
_OBSERVED_LOOP_VERDICT = re.compile(
    r'"looped"\s*:\s*(true|false)', re.IGNORECASE
)


def _contradicted_verdicts(
    conclusion: str, tool_calls: list[dict]
) -> list[AuditFinding]:
    """Findings for loop verdicts the conclusion asserts but no tool observed.

    Only ``measure_loop`` results count as observations. A claim is contradicted
    when it asserts a verdict that appears in no successful result — including
    the case where the opposite verdict is what every result showed.
    """
    claimed = {m.group(1).lower() for m in _CLAIMED_LOOP_VERDICT.finditer(conclusion)}
    if not claimed:
        return []

    observed: set[str] = set()
    for call in tool_calls:
        if call.get("name") != "measure_loop":
            continue
        if not call.get("ok", True):
            continue
        result = str(call.get("result", ""))
        if result.startswith("ERROR"):
            continue
        observed.update(m.group(1).lower() for m in _OBSERVED_LOOP_VERDICT.finditer(result))

    if not observed:
        # No observation at all is already covered by the behaviour check.
        return []

    unsupported = claimed - observed
    if not unsupported:
        return []

    return [
        AuditFinding(
            kind="contradicted-verdict",
            claim=f"looped: {verdict}",
            reason=(
                "conclusion asserts a loop verdict that no measure_loop result "
                f"showed; observed verdicts: {sorted(observed)}"
            ),
        )
        for verdict in sorted(unsupported)
    ]


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
