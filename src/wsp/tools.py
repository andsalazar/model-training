"""Tools an editor can call while working on a weight edit.

The Stage 0 ablation established that a static weight slice is not enough: every
editor, including the one that solved the task in the paper, declined because
weight magnitude alone does not establish a causal link to behaviour. What the
paper's editor actually had was a measurement loop.

This module gives an editor that loop. Each tool is a narrow, auditable
operation on a live model, and every call is recorded so a run can be replayed
and audited afterwards.

The tools are deliberately read-mostly: the only mutating operation is
``apply_edit``, and it returns a new checkpoint rather than modifying the
original.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from wsp import verify
from wsp.activations import (
    ActivationRegistry,
    contrast_cosine,
    direction_from_contrast,
    subtract_direction,
    summarize_direction,
    top_aligned_columns,
)
from wsp.verify import Probe, Role
from wsp.weights.slices import TensorSlice, apply_edit, to_text


@dataclass
class ToolCall:
    """One recorded tool invocation."""

    name: str
    arguments: dict[str, Any]
    result_summary: str
    ok: bool = True


@dataclass
class ToolLog:
    """Every call an editor made, in order."""

    calls: list[ToolCall] = field(default_factory=list)

    def record(self, call: ToolCall) -> None:
        self.calls.append(call)

    def summary(self) -> list[dict]:
        return [
            {"name": c.name, "args": c.arguments, "ok": c.ok, "result": c.result_summary}
            for c in self.calls
        ]

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return path


class ToolError(RuntimeError):
    """Raised when a tool cannot complete; reported back to the editor."""


class WeightTools:
    """The tool surface available to an editor.

    ``runner`` is the object that can actually execute the model. It is injected
    so this class stays testable without a GPU: tests pass a fake runner and
    assert on the calls made.
    """

    def __init__(
        self,
        store,
        runner,
        *,
        log: ToolLog | None = None,
        max_forward_passes: int = 50,
        capturer=None,
    ) -> None:
        self._store = store
        self._runner = runner
        self.log = log or ToolLog()
        self._forward_passes = 0
        self._max_forward_passes = max_forward_passes
        #: Optional activation capturer. When absent, the capture tools report
        #: that no model is attached rather than pretending to have measured.
        self._capturer = capturer
        self._directions = ActivationRegistry()
        #: Baseline readings captured before an edit, and the probes they came
        #: from. Both are needed for verify_edit to re-measure exactly the same
        #: prompts rather than whatever the editor remembers.
        self._baseline: list[verify.Reading] | None = None
        self._probes: list[Probe] = _default_probes()

    # -- introspection -----------------------------------------------------

    def list_tensors(self, pattern: str = "", limit: int = 200) -> str:
        """Names and shapes of tensors, optionally filtered by substring.

        Truncation is stated explicitly rather than silently cutting the list:
        an editor that sees a partial list without being told will reasonably
        conclude the naming is inconsistent.
        """
        keys = [k for k in self._store.keys if pattern in k]
        if not keys:
            self.log.record(ToolCall("list_tensors", {"pattern": pattern}, "no matches", ok=False))
            raise ToolError(f"no tensors match {pattern!r}")

        shown = keys[:limit]
        lines = []
        for key in shown:
            info = self._store.info(key)
            lines.append(f"{key}  {list(info.shape)} {info.dtype}")

        if len(keys) > len(shown):
            lines.append(
                f"... showing {len(shown)} of {len(keys)} matches. "
                f"Narrow the pattern or raise limit to see the rest."
            )

        out = "\n".join(lines)
        self.log.record(
            ToolCall(
                "list_tensors",
                {"pattern": pattern, "limit": limit},
                f"{len(shown)}/{len(keys)} tensors",
            )
        )
        return out

    def read_slice(
        self,
        tensor: str,
        rows: str | None = None,
        cols: str | None = None,
        max_chars: int = 20000,
    ) -> str:
        """Render a rectangular region of a tensor as text."""
        array = self._store.get(tensor)
        target = TensorSlice(
            name=tensor, rows=_parse_range(rows), cols=_parse_range(cols)
        )
        region = target.apply(array)
        text = to_text(region)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... truncated at {max_chars} chars"
        self.log.record(
            ToolCall(
                "read_slice",
                {"tensor": tensor, "rows": rows, "cols": cols},
                f"{region.shape}",
            )
        )
        return text

    def tensor_stats(self, tensor: str, cols: str | None = None) -> str:
        """Per-column summary statistics, which is what a slice cannot show.

        A column's norm and variance are the quantities that make an outlier
        visible without reading every value.
        """
        array = self._store.get(tensor)
        target = TensorSlice(name=tensor, cols=_parse_range(cols))
        region = target.apply(array)

        norms = np.linalg.norm(region, axis=0)
        means = region.mean(axis=0)
        stds = region.std(axis=0)
        order = np.argsort(norms)[::-1]

        lines = [f"{tensor}  region {region.shape}", "col\tnorm\tmean\tstd"]
        for idx in order[:32]:
            lines.append(
                f"{idx}\t{norms[idx]:.4f}\t{means[idx]:+.5f}\t{stds[idx]:.5f}"
            )
        out = "\n".join(lines)
        self.log.record(
            ToolCall("tensor_stats", {"tensor": tensor, "cols": cols}, f"{region.shape}")
        )
        return out

    # -- measurement -------------------------------------------------------

    def run_model(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate from the current checkpoint and return the text.

        This is the tool the ablation showed was missing: behaviour, not
        weights, is what identifies the bug.
        """
        if self._forward_passes >= self._max_forward_passes:
            raise ToolError(
                f"forward-pass budget exhausted ({self._max_forward_passes})"
            )
        self._forward_passes += 1
        text = self._runner.generate(prompt, max_tokens=max_tokens)
        self.log.record(
            ToolCall(
                "run_model",
                {"prompt": prompt[:120], "max_tokens": max_tokens},
                f"{len(text)} chars",
            )
        )
        return text

    def measure_loop(self, prompt: str, max_tokens: int = 2048) -> str:
        """Run the model and report whether it looped, plus token counts.

        Returns a compact verdict rather than raw text, because the signal the
        editor needs is termination, not prose.
        """
        if self._forward_passes >= self._max_forward_passes:
            raise ToolError(
                f"forward-pass budget exhausted ({self._max_forward_passes})"
            )
        self._forward_passes += 1
        result = self._runner.measure(prompt, max_tokens=max_tokens)
        self.log.record(
            ToolCall(
                "measure_loop",
                {"prompt": prompt[:120], "max_tokens": max_tokens},
                json.dumps(result),
            )
        )
        return json.dumps(result, indent=2)

    # -- activation capture ------------------------------------------------

    def _capture(self, prompt: str, tensor: str, max_tokens: int):
        """Capture activations at the module a tensor belongs to."""
        if self._capturer is None:
            raise ToolError(
                "no activation capturer attached: capture needs a pod, or pass "
                "a capturer that can execute"
            )
        if self._forward_passes >= self._max_forward_passes:
            raise ToolError(
                f"forward-pass budget exhausted ({self._max_forward_passes})"
            )
        from wsp.runner import module_path_for_tensor

        try:
            module_path = module_path_for_tensor(tensor)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

        self._forward_passes += 1
        return self._capturer.capture(prompt, module_path, max_tokens=max_tokens)

    def capture_activations(
        self, tensor: str, prompt: str, max_tokens: int = 64
    ) -> str:
        """Capture the module output for ``tensor`` and summarise its statistics.

        This is the observation ``tensor_stats`` cannot provide. A weight column's
        norm says how large it is, not what it does; the activation it produces on
        a looping prompt is what shows the loop's direction in hidden space.
        """
        array = self._capture(prompt, tensor, max_tokens)
        norms = np.linalg.norm(array, axis=0)
        means = array.mean(axis=0)
        order = np.argsort(norms)[::-1]

        lines = [
            f"{tensor}  activations {array.shape}  (tokens x hidden)",
            "dim\tnorm\tmean",
        ]
        for idx in order[:32]:
            lines.append(f"{idx}\t{norms[idx]:.4f}\t{means[idx]:+.5f}")
        out = "\n".join(lines)
        self.log.record(
            ToolCall(
                "capture_activations",
                {"tensor": tensor, "prompt": prompt[:120], "max_tokens": max_tokens},
                f"{array.shape}",
            )
        )
        return out

    def capture_direction(
        self,
        name: str,
        tensor: str,
        looping_prompt: str,
        control_prompt: str,
        max_tokens: int = 64,
        token_window: str | None = None,
    ) -> str:
        """Derive a hidden-space direction from a looping vs control prompt.

        The direction points from the control state toward the looping state and
        is stored under ``name`` so ``apply_directional_edit`` can use it without
        the vector being pasted through the transcript. ``contrast_cosine`` is
        reported so the editor can see whether the contrast found anything: a
        value near 1.0 means the two states were nearly identical and the
        direction is noise.

        ``token_window`` (``"start:stop"``) restricts the contrast to a token
        range. Averaging over all tokens averages mostly over the prefix the two
        prompts share, which is why a looping run and its control can come back
        with a cosine near 1.0 even when their behaviour differs sharply.
        """
        looping = self._capture(looping_prompt, tensor, max_tokens)
        control = self._capture(control_prompt, tensor, max_tokens)

        try:
            vector = direction_from_contrast(looping, control, token_window)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

        layer = _layer_from_tensor(tensor)
        summary = summarize_direction(
            name,
            layer,
            vector,
            n_tokens=int(np.asarray(looping).shape[0]),
            contrast=contrast_cosine(looping, control, token_window),
        )
        self._directions.put(name, vector, summary)
        self.log.record(
            ToolCall(
                "capture_direction",
                {
                    "name": name,
                    "tensor": tensor,
                    "looping_prompt": looping_prompt[:120],
                    "control_prompt": control_prompt[:120],
                    "token_window": token_window,
                },
                f"norm {summary.norm:.4f}, contrast_cosine {summary.contrast_cosine:+.4f}",
            )
        )
        return summary.as_text()

    def list_directions(self) -> str:
        """Names of captured directions, with their norms."""
        names = self._directions.names
        if not names:
            self.log.record(ToolCall("list_directions", {}, "none", ok=False))
            return "no captured directions"
        lines = ["name\tlayer\tnorm\tcontrast_cosine"]
        for name in names:
            s = self._directions.summary(name)
            lines.append(
                f"{name}\t{s.layer}\t{s.norm:.4f}\t{s.contrast_cosine:+.4f}"
            )
        out = "\n".join(lines)
        self.log.record(ToolCall("list_directions", {}, f"{len(names)} directions"))
        return out

    def direction_similarity(self, a: str, b: str) -> str:
        """Cosine between two captured directions.

        The check that settles whether a direction is signal or noise. A high
        ``contrast_cosine`` only says the two states looked alike; if the same
        direction comes back from a different layer or token window, the
        component is reproducible rather than an artefact of one comparison.
        """
        try:
            score = self._directions.similarity(a, b)
        except (KeyError, ValueError) as exc:
            raise ToolError(str(exc)) from exc

        lines = [f"cosine({a!r}, {b!r}) = {score:+.4f}"]
        if score > 0.9:
            lines.append(
                "The two directions agree closely: the component is reproducible, "
                "so it is signal rather than noise from one comparison."
            )
        elif score < 0.3:
            lines.append(
                "The two directions disagree: at least one is dominated by noise "
                "from its own comparison. Prefer a capture with lower "
                "contrast_cosine, or a token_window past the shared prefix."
            )
        out = "\n".join(lines)
        self.log.record(
            ToolCall("direction_similarity", {"a": a, "b": b}, f"{score:+.4f}")
        )
        return out

    # -- verification ------------------------------------------------------

    def baseline(self, probes: str = "", max_tokens: int = 2048) -> str:
        """Measure named probes on the current checkpoint and remember them.

        Call this *before* editing. The readings are kept so ``verify_edit`` can
        compare against them afterwards; an edit is only a result if something
        changed, and something only changed if there is a before.
        """
        if self._forward_passes + len(_parse_probes(probes)) > self._max_forward_passes:
            raise ToolError(
                "forward-pass budget too small for this baseline; "
                f"{self.forward_passes_left} left"
            )
        chosen = _parse_probes(probes)
        readings = verify.measure_all(self._runner, chosen)
        self._forward_passes += len(readings)
        self._baseline = readings

        lines = [f"baseline captured on {len(readings)} probe(s)", ""]
        for r in readings:
            flag = "" if r.correct else "  WRONG"
            cap = "  hit_cap" if r.hit_cap else ""
            lines.append(
                f"{r.probe:16s} {r.role.value:8s} {r.tokens:5d} tokens{cap}{flag}"
            )
        out = "\n".join(lines)
        self.log.record(
            ToolCall(
                "baseline",
                {"probes": probes or "default"},
                f"{len(readings)} probes",
            )
        )
        return out

    def verify_edit(self, max_tokens: int = 2048) -> str:
        """Re-measure the baseline probes on the current checkpoint and compare.

        This is the check that a single improved prompt cannot fake: it measures
        the target, held-out instances of the same behaviour, and a guard that
        must not move. A target that improves while a held-out probe regresses is
        reported as HARMFUL rather than validated.
        """
        if self._baseline is None:
            raise ToolError(
                "no baseline captured: call baseline before editing, so there is "
                "something to compare against"
            )
        if self._forward_passes + len(self._baseline) > self._max_forward_passes:
            raise ToolError(
                "forward-pass budget too small for this verification; "
                f"{self.forward_passes_left} left"
            )
        comparison = verify.verify_edit(self._runner, self._probes, self._baseline)
        self._forward_passes += len(comparison.edited)

        out = comparison.summary()
        self.log.record(
            ToolCall(
                "verify_edit",
                {},
                f"{comparison.status}: "
                + ", ".join(f"{v.probe}={v.state}" for v in comparison.verdicts),
            )
        )
        return out

    def aligned_columns(
        self, tensor: str, direction: str, k: int = 16
    ) -> str:
        """Columns of ``tensor`` most aligned with a captured direction.

        A column that writes along the loop direction is a candidate for the
        edit; a column orthogonal to it cannot move the state that way however
        large its norm is. This is the localisation ``tensor_stats`` cannot do.
        """
        try:
            vector = self._directions.get(direction)
        except KeyError as exc:
            raise ToolError(str(exc)) from exc

        array = self._store.get(tensor)
        if array.ndim != 2:
            raise ToolError(f"{tensor} is not 2-D (shape {array.shape})")

        top = top_aligned_columns(array, vector, k=k)
        lines = [f"{tensor}  alignment with {direction!r}", "col\tcosine"]
        for col, score in top:
            lines.append(f"{col}\t{score:+.5f}")
        out = "\n".join(lines)
        self.log.record(
            ToolCall(
                "aligned_columns",
                {"tensor": tensor, "direction": direction, "k": k},
                f"{len(top)} columns",
            )
        )
        return out

    # -- mutation ----------------------------------------------------------

    def apply_directional_edit(
        self,
        tensor: str,
        direction: str,
        cols: str | None,
        scale: float,
        out_path: str,
    ) -> str:
        """Subtract a captured direction from columns and load the result.

        Unlike ``apply_edit``, which takes one scalar per column, this removes a
        hidden-space vector from each selected column. That is the operation a
        per-column scalar cannot express: it changes *which direction* a neuron
        writes along, not just how large its output is.
        """
        try:
            vector = self._directions.get(direction)
        except KeyError as exc:
            raise ToolError(str(exc)) from exc

        # Resolve once: the editor may address the tensor by a shorter name than
        # the checkpoint key, and the write path needs the real key.
        tensor = self._store.resolve(tensor)
        array = self._store.get(tensor)
        if array.ndim != 2:
            raise ToolError(f"{tensor} is not 2-D (shape {array.shape})")

        selection = _parse_cols(cols)
        target = TensorSlice(name=tensor, cols=selection if isinstance(selection, tuple) else None)
        region = target.apply(array) if isinstance(selection, tuple) else array[:, selection]
        if region.shape[0] != vector.shape[0]:
            raise ToolError(
                f"direction has {vector.shape[0]} dims but {tensor} region has "
                f"{region.shape[0]} rows"
            )

        edited = subtract_direction(array, selection, vector, scale)

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        from wsp.runner import write_edited_checkpoint

        ckpt_dir = write_edited_checkpoint(
            self._store.path, tensor, edited, out.with_suffix("")
        )
        self._runner.reload(str(ckpt_dir))

        written = next(ckpt_dir.glob("*.safetensors"))
        self._store.reopen(written)
        reloaded = self._store.get(tensor)
        changed = int(np.count_nonzero(~np.isclose(reloaded, array, rtol=0.0, atol=0.0)))
        if changed == 0:
            raise ToolError(
                "edit did not change the tensor; refusing to report success"
            )

        self.log.record(
            ToolCall(
                "apply_directional_edit",
                {
                    "tensor": tensor,
                    "direction": direction,
                    "cols": cols,
                    "scale": scale,
                },
                f"{changed} weights changed, reloaded from {ckpt_dir}",
            )
        )
        return (
            f"subtracted {scale} x direction {direction!r} from {region.shape} region "
            f"({changed} weights changed) and reloaded the model from {ckpt_dir}. "
            f"Call measure_loop to test it."
        )

    def apply_edit(
        self,
        tensor: str,
        cols: str,
        direction: str,
        out_path: str,
    ) -> str:
        """Subtract a direction from the given columns and load the result.

        ``direction`` is a comma-separated list of numbers, one per column. The
        edit is additive and reversible; the source checkpoint is untouched.

        The edited tensor is written to a full checkpoint directory and the
        runner is reloaded from it, so a subsequent ``run_model`` or
        ``measure_loop`` actually observes the edit. An earlier version saved a
        bare ``.npy`` that nothing loaded, which made every edit a silent no-op.
        """
        try:
            values = np.array(
                [float(v) for v in direction.replace(" ", "").split(",")],
                dtype=np.float32,
            )
        except ValueError as exc:
            raise ToolError(f"direction must be comma-separated numbers: {exc}") from exc

        # Resolve once: the editor may address the tensor by a shorter name than
        # the checkpoint key, and the write path needs the real key.
        tensor = self._store.resolve(tensor)
        array = self._store.get(tensor)
        target = TensorSlice(name=tensor, cols=_parse_range(cols))
        region = target.apply(array)
        if values.shape != (region.shape[1],):
            raise ToolError(
                f"direction has {values.shape[0]} values, region has {region.shape[1]} columns"
            )

        delta = -np.broadcast_to(values, region.shape).astype(np.float32)
        edited = apply_edit(array, target, delta)

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Materialise a loadable checkpoint and point the runner at it.
        from wsp.runner import write_edited_checkpoint

        ckpt_dir = write_edited_checkpoint(
            self._store.path, tensor, edited, out.with_suffix("")
        )
        self._runner.reload(str(ckpt_dir))

        # Verify against the written file, not the in-memory array, so a silent
        # no-op cannot recur.
        written = next(ckpt_dir.glob("*.safetensors"))
        self._store.reopen(written)
        reloaded = self._store.get(tensor)
        changed = int(np.count_nonzero(~np.isclose(reloaded, array, rtol=0.0, atol=0.0)))
        if changed == 0:
            raise ToolError(
                "edit did not change the tensor; refusing to report success"
            )

        self.log.record(
            ToolCall(
                "apply_edit",
                {"tensor": tensor, "cols": cols, "n_values": int(values.size)},
                f"{changed} weights changed, reloaded from {ckpt_dir}",
            )
        )
        return (
            f"edited {region.shape} region ({changed} weights changed) and reloaded "
            f"the model from {ckpt_dir}. Call measure_loop to test it."
        )

    # -- budget ------------------------------------------------------------

    @property
    def forward_passes_used(self) -> int:
        return self._forward_passes

    @property
    def forward_passes_left(self) -> int:
        return self._max_forward_passes - self._forward_passes


def _parse_range(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    start, stop = value.split(":")
    return int(start), int(stop)


#: The probe set used when the editor does not name one. Targets are the
#: over-thinking behaviour; held-out probes are other instances of it; the guard
#: is unrelated behaviour that a merely-terser model would also shorten.
_PROBE_LIBRARY: dict[str, Probe] = {
    "target_12x11": Probe(
        name="target_12x11",
        prompt="What is 12 times 11? Reply with only the digits.",
        expected="132",
        role=Role.TARGET,
    ),
    "heldout_17x23": Probe(
        name="heldout_17x23",
        prompt="What is 17 times 23? Reply with only the digits.",
        expected="391",
        role=Role.HELDOUT,
    ),
    "heldout_13x14": Probe(
        name="heldout_13x14",
        prompt="What is 13 times 14? Reply with only the digits.",
        expected="182",
        role=Role.HELDOUT,
    ),
    "heldout_9x8": Probe(
        name="heldout_9x8",
        prompt="What is 9 times 8? Reply with only the digits.",
        expected="72",
        role=Role.HELDOUT,
    ),
    "guard_capital": Probe(
        name="guard_capital",
        prompt="Name the capital of France. Reply with only the city.",
        expected="Paris",
        role=Role.GUARD,
    ),
}

#: The set ``baseline`` measures when the editor passes no probe names. Kept
#: small on purpose: every probe costs a forward pass, and a target with no
#: held-out instance cannot tell a real suppressor from a prompt-specific fluke.
_DEFAULT_PROBE_NAMES: tuple[str, ...] = (
    "target_12x11",
    "heldout_17x23",
    "heldout_9x8",
    "guard_capital",
)


def _default_probes() -> list[Probe]:
    return [_PROBE_LIBRARY[name] for name in _DEFAULT_PROBE_NAMES]


def _parse_probes(value: str | None) -> list[Probe]:
    """Resolve a comma-separated probe list, or the default set when empty."""
    if not value:
        return _default_probes()
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [n for n in names if n not in _PROBE_LIBRARY]
    if unknown:
        known = ", ".join(sorted(_PROBE_LIBRARY))
        raise ToolError(f"unknown probe(s) {unknown}; known: {known}")
    return [_PROBE_LIBRARY[n] for n in names]


def _parse_cols(value: str | None) -> tuple[int, int] | list[int] | None:
    """Parse a column selection: ``"start:stop"`` or ``"12,987,914"``.

    ``aligned_columns`` ranks columns by alignment and the top-ranked ones are
    scattered, so a contiguous range cannot express the edit the evidence points
    at. explore13's editor used ``900:1000`` because that was the only form
    available, and 97 of those 100 columns were not ranked as aligned at all.
    """
    if not value:
        return None
    text = value.strip()
    if ":" in text:
        return _parse_range(text)
    try:
        indices = [int(part) for part in text.replace(" ", "").split(",") if part]
    except ValueError as exc:
        raise ToolError(
            f"cols must be 'start:stop' or comma-separated indices: {exc}"
        ) from exc
    if not indices:
        raise ToolError("cols was empty")
    return indices


def _layer_from_tensor(tensor: str) -> int:
    """The decoder layer index in a tensor name, or -1 when there is none."""
    import re

    match = re.search(r"layers\.(\d+)\.", tensor)
    return int(match.group(1)) if match else -1


#: Schema shown to the editor, so it knows what it can call.
TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "list_tensors",
        "description": (
            "List tensor names and shapes, filtered by substring. Reports when "
            "the list is truncated."
        ),
        "parameters": {
            "pattern": "substring to match, or empty for all",
            "limit": "max entries to show (default 200)",
        },
    },
    {
        "name": "read_slice",
        "description": "Render a rectangular region of a tensor as text.",
        "parameters": {"tensor": "name", "rows": "start:stop or null", "cols": "start:stop or null"},
    },
    {
        "name": "tensor_stats",
        "description": "Per-column norm, mean and std, sorted by norm descending.",
        "parameters": {"tensor": "name", "cols": "start:stop or null"},
    },
    {
        "name": "run_model",
        "description": "Generate from the current checkpoint and return the text.",
        "parameters": {"prompt": "text", "max_tokens": "int"},
    },
    {
        "name": "measure_loop",
        "description": "Run the model and report whether it looped, with token counts.",
        "parameters": {"prompt": "text", "max_tokens": "int"},
    },
    {
        "name": "apply_edit",
        "description": "Subtract a direction from columns and save a new checkpoint.",
        "parameters": {
            "tensor": "name",
            "cols": "start:stop",
            "direction": "comma-separated numbers, one per column",
            "out_path": "where to save",
        },
    },
    {
        "name": "capture_activations",
        "description": (
            "Run one forward pass and report the hooked module's output statistics "
            "(tokens x hidden). Shows what a weight column actually produces, which "
            "tensor_stats cannot."
        ),
        "parameters": {
            "tensor": "name of the weight whose module to hook",
            "prompt": "text",
            "max_tokens": "integer, default 64",
        },
    },
    {
        "name": "capture_direction",
        "description": (
            "Derive a hidden-space direction from a looping prompt vs a control "
            "prompt and store it under a name. Reports contrast_cosine: near 1.0 "
            "means the contrast found nothing and the direction is noise."
        ),
        "parameters": {
            "name": "short label to refer to it later",
            "tensor": "name of the weight whose module to hook",
            "looping_prompt": "text that makes the model loop",
            "control_prompt": "text that does not",
            "max_tokens": "integer, default 64",
            "token_window": (
                "optional 'start:stop' token range to contrast; use the region "
                "where the two prompts diverge, not the prefix they share"
            ),
        },
    },
    {
        "name": "list_directions",
        "description": "Names of captured directions with their norms.",
        "parameters": {},
    },
    {
        "name": "direction_similarity",
        "description": (
            "Cosine between two captured directions. Use this to test whether a "
            "direction is signal: a high contrast_cosine only means the two states "
            "looked alike, but agreement between captures at different layers or "
            "token windows means the component is reproducible."
        ),
        "parameters": {
            "a": "first captured direction name",
            "b": "second captured direction name",
        },
    },
    {
        "name": "aligned_columns",
        "description": (
            "Columns of a tensor most aligned with a captured direction. A column "
            "orthogonal to the loop direction cannot move the state that way, "
            "however large its norm."
        ),
        "parameters": {
            "tensor": "name",
            "direction": "captured direction name",
            "k": "how many columns to show",
        },
    },
    {
        "name": "baseline",
        "description": (
            "Measure the probe set on the CURRENT checkpoint and remember the "
            "readings. Call this before editing: an edit is only a result if "
            "something changed, and something only changed if there is a before. "
            "Probes cover the target, held-out instances of the same behaviour, "
            "and a guard that must not move."
        ),
        "parameters": {
            "probes": (
                "comma-separated probe names, or empty for the default set: "
                + ", ".join(sorted(_PROBE_LIBRARY))
            ),
        },
    },
    {
        "name": "verify_edit",
        "description": (
            "Re-measure the baseline probes on the current checkpoint and compare. "
            "This is the check a single improved prompt cannot fake: it reports "
            "VALIDATED only when every target improved and no held-out or guard "
            "probe was damaged, and HARMFUL when an edit that helped the target "
            "pushed something else to the token cap."
        ),
        "parameters": {},
    },
    {
        "name": "apply_directional_edit",
        "description": (
            "Subtract a captured direction from columns and load the result. "
            "Changes which direction a neuron writes along, not just its magnitude."
        ),
        "parameters": {
            "tensor": "name",
            "direction": "captured direction name",
            "cols": (
                "start:stop, or comma-separated indices like '12,987,914' to "
                "edit exactly the columns aligned_columns ranked, or null for all"
            ),
            "scale": "how far to move along the direction",
            "out_path": "where to save",
        },
    },
]
