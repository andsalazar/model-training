"""Weight slicing and serialization.

No context window holds a full model, so Weight-Space Programming operates on
*slices*: a named tensor, or a sub-block of one, rendered as text the editor can
reason about. This module owns that rendering and the inverse operation.

The paper's Qwen edit targeted "64 output columns in each of two MLP layers",
which is a column slice of a 2-D weight matrix. That shape is the primary case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TensorSlice:
    """A rectangular region of a 2-D weight tensor.

    ``rows`` and ``cols`` are half-open ``(start, stop)`` ranges. ``None`` means
    the full extent along that axis.
    """

    name: str
    rows: tuple[int, int] | None = None
    cols: tuple[int, int] | None = None

    def apply(self, tensor: np.ndarray) -> np.ndarray:
        if tensor.ndim != 2:
            raise ValueError(f"{self.name}: expected a 2-D tensor, got shape {tensor.shape}")
        r = slice(*self.rows) if self.rows else slice(None)
        c = slice(*self.cols) if self.cols else slice(None)
        return tensor[r, c]

    def describe(self, shape: tuple[int, int]) -> str:
        view = self.apply(np.zeros(shape, dtype=np.float32))
        return f"{self.name}[{view.shape[0]}x{view.shape[1]}]"


def to_text(array: np.ndarray, *, precision: int = 5) -> str:
    """Render an array as a compact, line-per-row text block.

    Row-per-line with space-separated values keeps the structure legible to the
    editor: it can count columns and reason about which output unit a value
    belongs to.
    """
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D array, got shape {array.shape}")

    fmt = f"{{:.{precision}g}}"
    lines = [" ".join(fmt.format(float(v)) for v in row) for row in array]
    return "\n".join(lines)


def from_text(text: str, *, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Parse a text block produced by :func:`to_text` back into an array.

    Tolerates surrounding prose and code fences, because editors sometimes wrap
    their output. Raises if the result is ragged or does not match ``shape``.
    """
    rows: list[list[float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("[](),")
        if not line or line.startswith(("#", "//", "```")):
            continue
        parts = line.replace(",", " ").split()
        try:
            values = [float(p) for p in parts]
        except ValueError:
            # Not a numeric row; skip prose the editor added.
            continue
        if values:
            rows.append(values)

    if not rows:
        raise ValueError("no numeric rows found in editor output")

    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"ragged output: row widths {sorted(widths)}")

    array = np.asarray(rows, dtype=np.float32)
    if shape is not None and array.shape != shape:
        raise ValueError(f"shape mismatch: expected {shape}, parsed {array.shape}")
    return array


def apply_edit(
    tensor: np.ndarray,
    target: TensorSlice,
    delta: np.ndarray,
) -> np.ndarray:
    """Return a copy of ``tensor`` with ``delta`` added at ``target``.

    The paper's edit is a directional subtraction; expressing it as an additive
    delta keeps the operation reversible and auditable.
    """
    region = target.apply(tensor)
    if region.shape != delta.shape:
        raise ValueError(
            f"{target.name}: delta shape {delta.shape} does not match region {region.shape}"
        )

    out = tensor.copy()
    r = slice(*target.rows) if target.rows else slice(None)
    c = slice(*target.cols) if target.cols else slice(None)
    out[r, c] = region + delta
    return out


def count_changed(original: np.ndarray, edited: np.ndarray, *, atol: float = 0.0) -> int:
    """Number of elements that differ, for reporting the edit's footprint."""
    if original.shape != edited.shape:
        raise ValueError(f"shape mismatch: {original.shape} vs {edited.shape}")
    return int(np.count_nonzero(~np.isclose(original, edited, atol=atol, rtol=0.0)))
