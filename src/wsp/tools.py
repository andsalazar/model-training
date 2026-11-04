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
    ) -> None:
        self._store = store
        self._runner = runner
        self.log = log or ToolLog()
        self._forward_passes = 0
        self._max_forward_passes = max_forward_passes

    # -- introspection -----------------------------------------------------

    def list_tensors(self, pattern: str = "") -> str:
        """Names and shapes of tensors, optionally filtered by substring."""
        keys = [k for k in self._store.keys if pattern in k]
        if not keys:
            self.log.record(ToolCall("list_tensors", {"pattern": pattern}, "no matches", ok=False))
            raise ToolError(f"no tensors match {pattern!r}")

        lines = []
        for key in keys[:200]:
            info = self._store.info(key)
            lines.append(f"{key}  {list(info.shape)} {info.dtype}")
        out = "\n".join(lines)
        self.log.record(
            ToolCall("list_tensors", {"pattern": pattern}, f"{len(keys)} tensors")
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

    # -- mutation ----------------------------------------------------------

    def apply_edit(
        self,
        tensor: str,
        cols: str,
        direction: str,
        out_path: str,
    ) -> str:
        """Subtract a direction from the given columns and save a new checkpoint.

        ``direction`` is a comma-separated list of numbers, one per column. The
        edit is additive and reversible; the source checkpoint is untouched.
        """
        try:
            values = np.array(
                [float(v) for v in direction.replace(" ", "").split(",")],
                dtype=np.float32,
            )
        except ValueError as exc:
            raise ToolError(f"direction must be comma-separated numbers: {exc}") from exc

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
        np.save(out.with_suffix(".npy"), edited)

        self.log.record(
            ToolCall(
                "apply_edit",
                {"tensor": tensor, "cols": cols, "n_values": int(values.size)},
                f"wrote {out.with_suffix('.npy')}",
            )
        )
        return f"edited {region.shape} region, wrote {out.with_suffix('.npy')}"

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


#: Schema shown to the editor, so it knows what it can call.
TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "list_tensors",
        "description": "List tensor names and shapes, filtered by substring.",
        "parameters": {"pattern": "substring to match, or empty for all"},
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
]
