"""Independent re-measurement, so a result does not rest on the editor's honesty.

Every editor run in Stage 1 needed checking, and both of the runs that got far
enough to matter fabricated numbers: explore13 reported a ``direction_similarity``
of 0.9995 when the tool returned -0.0485, and explore14 reported post-edit token
counts without ever calling ``measure_loop`` after its edits. Auditing the
transcript catches some of that after the fact, but the reliable check is to
measure the checkpoint again outside the editor and compare.

This module is that check. It is deliberately free of model plumbing: a *runner*
is anything with ``generate`` and ``count_tokens``, so the same code runs against
vLLM on a pod and against a fake in a unit test.

The comparison is a fixed rule, not a judgement call:

- a probe whose correct answer disappears is **broken**
- a probe that now hits the token cap, or grows past a factor of the baseline,
  **regressed**
- a probe that falls below a fraction of the baseline **improved**
- anything inside the noise floor is **unchanged**

An edit is **validated** only when at least one target improved and nothing
regressed or broke. An edit that improves the target while pushing a held-out
question to the cap is **harmful**, not validated -- that is exactly the failure
mode the first ranked-column attempt avoided and the contiguous-window attempt
did not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from wsp.runner import classify


class Role(str, Enum):
    """What a probe is for.

    The role decides what the comparison demands of it. ``TARGET`` is the
    behaviour the edit is supposed to change. ``HELDOUT`` is a different instance
    of the same behaviour, which must not be damaged. ``GUARD`` is unrelated
    behaviour that must be left alone, so an edit that simply makes the model
    terser everywhere is caught.
    """

    TARGET = "target"
    HELDOUT = "heldout"
    GUARD = "guard"


#: Fraction of baseline tokens at or below which a probe counts as improved.
IMPROVE_RATIO = 0.75

#: Fraction of baseline tokens at or above which a probe counts as regressed.
REGRESS_RATIO = 1.25

#: Absolute token change below which a difference is noise rather than signal.
#: Short answers vary by a few tokens between runs; treating that as an effect
#: would make every edit look like it did something.
MIN_DELTA = 25


@dataclass(frozen=True)
class Probe:
    """One prompt to measure, and the answer that shows it still works."""

    name: str
    prompt: str
    expected: str
    role: Role = Role.TARGET
    max_tokens: int = 2048


@dataclass(frozen=True)
class Reading:
    """What one probe did on one checkpoint."""

    probe: str
    role: Role
    tokens: int
    hit_cap: bool
    repeated: bool
    correct: bool
    text: str = ""

    @property
    def text_chars(self) -> int:
        return len(self.text)

    def as_dict(self) -> dict:
        return {
            "probe": self.probe,
            "role": self.role.value,
            "tokens": self.tokens,
            "hit_cap": self.hit_cap,
            "repeated": self.repeated,
            "correct": self.correct,
            "text_chars": self.text_chars,
            "text": self.text,
        }


@dataclass(frozen=True)
class Verdict:
    """How one probe changed between the baseline and the edited checkpoint."""

    probe: str
    role: Role
    state: str
    baseline_tokens: int
    edited_tokens: int
    detail: str

    def as_dict(self) -> dict:
        return {
            "probe": self.probe,
            "role": self.role.value,
            "state": self.state,
            "baseline_tokens": self.baseline_tokens,
            "edited_tokens": self.edited_tokens,
            "detail": self.detail,
        }


def measure_probe(runner, probe: Probe) -> Reading:
    """Run one probe against whatever checkpoint ``runner`` is holding."""
    text = runner.generate(probe.prompt, max_tokens=probe.max_tokens)
    n_tokens = runner.count_tokens(text)
    signal = classify(text, n_tokens, probe.max_tokens)
    return Reading(
        probe=probe.name,
        role=probe.role,
        tokens=signal.tokens,
        hit_cap=signal.hit_cap,
        repeated=signal.repeated,
        correct=probe.expected.lower() in text.lower(),
        text=text,
    )


def measure_all(runner, probes: list[Probe]) -> list[Reading]:
    """Run every probe in order, so two checkpoints are measured identically."""
    return [measure_probe(runner, probe) for probe in probes]


def classify_change(baseline: Reading, edited: Reading) -> Verdict:
    """Compare one probe's two readings under the fixed rule above."""
    if baseline.correct and not edited.correct:
        state = "broken"
        detail = (
            "the expected answer was present before the edit and is gone after it"
        )
    elif edited.hit_cap and not baseline.hit_cap:
        state = "regressed"
        detail = (
            f"hit the {edited.tokens}-token cap after the edit; "
            f"baseline stopped at {baseline.tokens}"
        )
    elif baseline.tokens == 0:
        state = "unchanged"
        detail = "baseline produced no tokens, so there is nothing to compare"
    else:
        delta = edited.tokens - baseline.tokens
        ratio = edited.tokens / baseline.tokens
        if abs(delta) < MIN_DELTA:
            state = "unchanged"
            detail = (
                f"changed by {delta:+d} tokens, inside the {MIN_DELTA}-token "
                "noise floor"
            )
        elif ratio >= REGRESS_RATIO:
            state = "regressed"
            detail = (
                f"grew from {baseline.tokens} to {edited.tokens} tokens "
                f"({ratio:.2f}x baseline)"
            )
        elif ratio <= IMPROVE_RATIO:
            state = "improved"
            detail = (
                f"fell from {baseline.tokens} to {edited.tokens} tokens "
                f"({ratio:.2f}x baseline)"
            )
        else:
            state = "unchanged"
            detail = (
                f"moved from {baseline.tokens} to {edited.tokens} tokens "
                f"({ratio:.2f}x baseline), which is neither a meaningful "
                "improvement nor a regression"
            )

    return Verdict(
        probe=baseline.probe,
        role=baseline.role,
        state=state,
        baseline_tokens=baseline.tokens,
        edited_tokens=edited.tokens,
        detail=detail,
    )


@dataclass
class Comparison:
    """The verdict on one edit, across every probe."""

    verdicts: list[Verdict] = field(default_factory=list)
    baseline: list[Reading] = field(default_factory=list)
    edited: list[Reading] = field(default_factory=list)

    def _of_role(self, role: Role) -> list[Verdict]:
        return [v for v in self.verdicts if v.role is role]

    @property
    def status(self) -> str:
        """``validated``, ``no-effect`` or ``harmful``.

        ``harmful`` is checked first: an edit that helped the target is still a
        failure if it damaged something else, and that is the easy mistake to
        talk yourself out of.
        """
        harmful = [
            v for v in self.verdicts if v.state in {"regressed", "broken"}
        ]
        if harmful:
            return "harmful"

        targets = self._of_role(Role.TARGET)
        if targets and all(v.state == "improved" for v in targets):
            return "validated"
        if any(v.state == "improved" for v in targets):
            return "partial"
        return "no-effect"

    @property
    def validated(self) -> bool:
        return self.status == "validated"

    def summary(self) -> str:
        """A table plus the reason for the verdict, for a log or a report."""
        if not self.verdicts:
            return "no probes measured"

        width = max(len(v.probe) for v in self.verdicts)
        lines = [
            f"{'probe':{width}s} {'role':8s} {'base':>6s} {'edited':>7s}  state",
            "-" * (width + 34),
        ]
        for v in self.verdicts:
            lines.append(
                f"{v.probe:{width}s} {v.role.value:8s} "
                f"{v.baseline_tokens:6d} {v.edited_tokens:7d}  {v.state}"
            )

        lines.append("")
        harmful = [v for v in self.verdicts if v.state in {"regressed", "broken"}]
        if harmful:
            lines.append(
                f"HARMFUL: {len(harmful)} probe(s) regressed or broke — "
                + ", ".join(f"{v.probe} ({v.state})" for v in harmful)
            )
        elif self.status == "validated":
            lines.append(
                "VALIDATED: every target improved and no held-out or guard "
                "probe was damaged"
            )
        elif self.status == "partial":
            lines.append(
                "PARTIAL: some targets improved but not all; check the per-probe "
                "detail before calling this a result"
            )
        else:
            lines.append(
                "NO EFFECT: no target improved by more than the noise floor"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "validated": self.validated,
            "verdicts": [v.as_dict() for v in self.verdicts],
            "baseline": [r.as_dict() for r in self.baseline],
            "edited": [r.as_dict() for r in self.edited],
        }


def compare(baseline: list[Reading], edited: list[Reading]) -> Comparison:
    """Compare two sets of readings, matched by probe name."""
    by_name = {r.probe: r for r in edited}
    verdicts: list[Verdict] = []
    for base in baseline:
        other = by_name.get(base.probe)
        if other is None:
            continue
        verdicts.append(classify_change(base, other))
    return Comparison(verdicts=verdicts, baseline=list(baseline), edited=list(edited))


def verify_edit(runner, probes: list[Probe], baseline: list[Reading]) -> Comparison:
    """Re-measure ``probes`` on the runner's current checkpoint and compare.

    ``baseline`` must have been captured *before* the edit, on the untouched
    checkpoint. That ordering is the whole point: a reading taken before an edit
    says nothing about the edit's effect, which is the failure explore14's
    conclusion was built on.
    """
    edited = measure_all(runner, probes)
    return compare(baseline, edited)
