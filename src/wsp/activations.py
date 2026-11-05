"""Activation capture and directional edits.

Stage 1's blocker was localisation. Per-column weight statistics describe how
large a column is, not what it does: explore7 measured layer 23 column 1404 as a
high-variance column and correctly concluded that subtracting a scalar shifts the
mean and therefore cannot address a spread anomaly. The direction that drives a
repetition loop has to come from the *activations* the loop produces, not from
the weight magnitudes.

This module holds the pure math for that: turning a looping state and a control
state into a direction, scoring which weight columns write along it, and keeping
captured directions addressable by name so an editor can refer to one in a later
call. The model-side hooking lives in :mod:`wsp.runner`; everything here is
array math and is testable without a GPU.

Orientation matters and is easy to get wrong. A decoder MLP's ``down_proj`` has
shape ``[hidden, intermediate]``, so ``down_proj[:, j]`` is a *hidden-space*
vector: the direction intermediate neuron ``j`` writes into the residual stream.
A captured residual-stream direction is also hidden-space, so subtracting it from
a column is dimensionally valid and means "stop this neuron writing along the
loop direction".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Above this cosine the two states are so similar that a direction derived from
#: them is mostly noise. Reported as a warning rather than silently accepted.
_WEAK_CONTRAST = 0.99


@dataclass(frozen=True)
class DirectionSummary:
    """A captured direction, summarised for an editor to reason about."""

    name: str
    layer: int
    dim: int
    norm: float
    #: The largest-magnitude components, as ``(dim, value)`` pairs.
    top_dims: tuple[tuple[int, float], ...]
    #: How many tokens the direction was derived from.
    n_tokens: int
    #: Cosine similarity between the two states it was derived from. Near 1.0
    #: means the contrast found almost no difference and the direction is noise.
    contrast_cosine: float

    def as_text(self) -> str:
        lines = [
            f"direction {self.name!r}  layer {self.layer}  dim {self.dim}",
            f"norm {self.norm:.4f}  tokens {self.n_tokens}  "
            f"contrast_cosine {self.contrast_cosine:+.4f}",
        ]
        # A direction is normalised, so a contrast that found almost nothing is
        # still reported at unit norm. Say so plainly: an editor reading only the
        # norm would think it had a strong signal when it has amplified noise.
        #
        # The cosine is a caution, not a verdict. It compares the two *means*, and
        # a prefix the prompts share appears in both and cancels when they are
        # subtracted — so a high cosine does not by itself mean the direction is
        # noise. explore12's editor read it as fatal and abandoned a usable
        # direction. Point at the check that actually settles it instead.
        if abs(self.contrast_cosine) > _WEAK_CONTRAST:
            lines.append(
                f"CAUTION: contrast_cosine {self.contrast_cosine:+.4f} is close to "
                f"1.0, so the two states look similar. That is not proof the "
                f"direction is noise: a prefix both prompts share cancels in the "
                f"subtraction. Test it instead — capture the same direction at "
                f"another layer or with a token_window past the shared prefix and "
                f"compare the two vectors. Agreement means the signal is real."
            )
        lines.append("dim\tvalue")
        for dim, value in self.top_dims:
            lines.append(f"{dim}\t{value:+.5f}")
        return "\n".join(lines)


@dataclass
class ActivationRegistry:
    """Named directions captured during a run.

    An editor captures a direction in one call and applies it in another, so the
    vector has to survive between calls without being pasted through the
    transcript. Names are short and editor-chosen.
    """

    _directions: dict[str, np.ndarray] = field(default_factory=dict)
    _meta: dict[str, DirectionSummary] = field(default_factory=dict)

    def put(self, name: str, vector: np.ndarray, summary: DirectionSummary) -> None:
        self._directions[name] = np.asarray(vector, dtype=np.float32)
        self._meta[name] = summary

    def get(self, name: str) -> np.ndarray:
        if name not in self._directions:
            known = ", ".join(sorted(self._directions)) or "none"
            raise KeyError(f"no captured direction {name!r} (known: {known})")
        return self._directions[name]

    def summary(self, name: str) -> DirectionSummary:
        if name not in self._meta:
            known = ", ".join(sorted(self._meta)) or "none"
            raise KeyError(f"no captured direction {name!r} (known: {known})")
        return self._meta[name]

    @property
    def names(self) -> list[str]:
        return sorted(self._directions)

    def similarity(self, a: str, b: str) -> float:
        """Cosine between two captured directions.

        This is the check that settles whether a direction is signal or noise. A
        high ``contrast_cosine`` only says the two states looked alike; if the
        same direction comes back from a different layer or a different token
        window, the component is real and reproducible rather than an artefact of
        one comparison.
        """
        va = self.get(a)
        vb = self.get(b)
        if va.shape != vb.shape:
            raise ValueError(
                f"directions have different dims: {a} {va.shape} vs {b} {vb.shape}"
            )
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)


def _as_2d(activations: np.ndarray) -> np.ndarray:
    """Coerce captured activations to ``[n_tokens, hidden]``."""
    array = np.asarray(activations, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D activations, got shape {array.shape}")
    return array


def mean_state(activations: np.ndarray) -> np.ndarray:
    """The mean activation vector over tokens."""
    return _as_2d(activations).mean(axis=0)


def _window(array: np.ndarray, token_window: str | None) -> np.ndarray:
    """Restrict an activation array to a token range.

    ``token_window`` is ``"start:stop"`` in token indices, or ``None`` for all
    tokens. A contrast taken over all tokens averages mostly over the prefix the
    two prompts share, which is why a looping run and its control can come back
    with a cosine near 1.0 even when their behaviour differs sharply. Restricting
    to the region where they diverge is what makes the contrast informative.
    """
    matrix = np.asarray(array, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D activation array, got {matrix.shape}")
    if not token_window:
        return matrix
    try:
        start_text, _, stop_text = token_window.partition(":")
        start = int(start_text) if start_text.strip() else 0
        stop = int(stop_text) if stop_text.strip() else matrix.shape[0]
    except ValueError as exc:
        raise ValueError(f"token_window must be 'start:stop': {exc}") from exc
    if start < 0 or stop > matrix.shape[0] or start >= stop:
        raise ValueError(
            f"token_window {token_window!r} is outside 0:{matrix.shape[0]}"
        )
    return matrix[start:stop]


def direction_from_contrast(
    looping: np.ndarray, control: np.ndarray, token_window: str | None = None
) -> np.ndarray:
    """A unit direction pointing from the control state toward the looping state.

    ``looping`` and ``control`` are activation arrays of shape
    ``[n_tokens, hidden]``; each is reduced to its mean over tokens first. The
    result is normalised, so its scale carries no meaning and only its direction
    does — the caller chooses how far to move along it.

    ``token_window`` restricts both arrays to a token range before averaging. Use
    it to contrast the region where the two prompts diverge rather than the
    prefix they share.
    """
    loop_mean = mean_state(_window(looping, token_window))
    control_mean = mean_state(_window(control, token_window))
    if loop_mean.shape != control_mean.shape:
        raise ValueError(
            f"shape mismatch: looping {loop_mean.shape} vs control {control_mean.shape}"
        )
    delta = loop_mean - control_mean
    norm = float(np.linalg.norm(delta))
    if norm == 0.0:
        raise ValueError(
            "looping and control states are identical; no direction to derive"
        )
    return (delta / norm).astype(np.float32)


def contrast_cosine(
    looping: np.ndarray, control: np.ndarray, token_window: str | None = None
) -> float:
    """Cosine similarity between the two mean states.

    Near 1.0 means the contrast found almost no difference, so a direction
    derived from it is mostly noise. Reported alongside the direction so an
    editor can judge whether the capture is worth acting on.

    ``token_window`` must match the one passed to :func:`direction_from_contrast`,
    or the reported cosine describes a different region than the direction.
    """
    loop_mean = mean_state(_window(looping, token_window))
    control_mean = mean_state(_window(control, token_window))
    denom = float(np.linalg.norm(loop_mean) * np.linalg.norm(control_mean))
    if denom == 0.0:
        return 0.0
    return float(np.dot(loop_mean, control_mean) / denom)


def column_alignment(weight: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Cosine alignment of every column of ``weight`` with ``direction``.

    ``weight`` is ``[hidden, intermediate]`` and ``direction`` is ``[hidden]``,
    so the result has one score per intermediate column. A high positive score
    means that column writes along the direction; a score near zero means it is
    orthogonal and editing it cannot move the state along that direction.
    """
    matrix = np.asarray(weight, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got shape {matrix.shape}")
    vec = np.asarray(direction, dtype=np.float32).reshape(-1)
    if vec.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"direction has {vec.shape[0]} dims but weight has {matrix.shape[0]} rows"
        )
    col_norms = np.linalg.norm(matrix, axis=0)
    denom = np.where(col_norms == 0.0, 1.0, col_norms)
    return (matrix.T @ vec / denom).astype(np.float32)


def top_aligned_columns(
    weight: np.ndarray, direction: np.ndarray, k: int = 16
) -> list[tuple[int, float]]:
    """The ``k`` columns most aligned with ``direction``, strongest first."""
    scores = column_alignment(weight, direction)
    order = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in order]


def alignment_spectrum(
    weight: np.ndarray, direction: np.ndarray, depth: int = 40
) -> list[tuple[int, float]]:
    """The top ``depth`` columns with their alignment scores, unclipped.

    Exists so a caller can *see* the shape of the ranking before choosing a cut.
    Stage 2's failure was caused by taking the top 12 of a distribution that ties
    to within 0.0012 at that point: the cut was arbitrary and the verdict turned
    on it. A spectrum makes the plateau visible instead of hiding it behind a k.
    """
    scores = column_alignment(weight, direction)
    order = np.argsort(scores)[::-1][:depth]
    return [(int(i), float(scores[i])) for i in order]


@dataclass(frozen=True)
class StabilitySelection:
    """Columns that survive a threshold, and the reason for the cut.

    ``threshold`` is a *fraction of the top score*, not a rank. That is the whole
    point: a rank is arbitrary inside a tie, while "columns within 90% of the best
    alignment" is a statement about the score distribution and moves smoothly as
    the distribution moves.
    """

    columns: tuple[int, ...]
    scores: tuple[float, ...]
    threshold: float
    ratio: float
    candidate_depth: int

    @property
    def k(self) -> int:
        return len(self.columns)

    def as_dict(self) -> dict:
        return {
            "columns": list(self.columns),
            "scores": list(self.scores),
            "threshold": self.threshold,
            "ratio": self.ratio,
            "candidate_depth": self.candidate_depth,
            "k": self.k,
        }

    def describe(self) -> str:
        return (
            f"{self.k} columns within {self.ratio:.0%} of the top score "
            f"({self.threshold:.6f}) out of the best {self.candidate_depth}"
        )


def select_stable_columns(
    weight: np.ndarray,
    direction: np.ndarray,
    *,
    ratio: float = 0.9,
    candidate_depth: int = 200,
    min_columns: int = 4,
) -> StabilitySelection:
    """Select columns by score threshold rather than by rank.

    The rule: take every column among the best ``candidate_depth`` whose alignment
    score is at least ``ratio`` times the best score. This is *stable* in the sense
    Stage 2 needs — a column's membership depends on its own score relative to the
    maximum, not on how many other columns happen to be slightly above it, so
    small numerical differences cannot swap columns across a boundary.

    It is also honest about magnitude: when the ranking is a plateau the threshold
    admits many columns and the selection says so, instead of silently taking an
    arbitrary 12.

    ``min_columns`` guards the degenerate case where the top score is far ahead of
    everything else and the rule would select one or two columns, which is too
    small an edit to move behaviour reliably.
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    if candidate_depth < 1:
        raise ValueError(f"candidate_depth must be positive, got {candidate_depth}")
    if min_columns < 1:
        raise ValueError(f"min_columns must be positive, got {min_columns}")

    spectrum = alignment_spectrum(weight, direction, depth=candidate_depth)
    if not spectrum:
        raise ValueError("no columns to select from")

    best = spectrum[0][1]
    if best <= 0.0:
        # Every cosine is non-positive: the direction is unrelated to every column
        # in the candidate set, so there is nothing to align with.
        return StabilitySelection(
            columns=(),
            scores=(),
            threshold=0.0,
            ratio=ratio,
            candidate_depth=candidate_depth,
        )

    threshold = best * ratio
    chosen = [(col, score) for col, score in spectrum if score >= threshold]

    # Backfill from the spectrum when the threshold is too strict to yield an edit
    # of a usable size. Deterministic: the spectrum is already score-ordered.
    if len(chosen) < min_columns:
        chosen = spectrum[:min_columns]

    return StabilitySelection(
        columns=tuple(col for col, _ in chosen),
        scores=tuple(score for _, score in chosen),
        threshold=threshold,
        ratio=ratio,
        candidate_depth=candidate_depth,
    )


def selection_overlap(a: StabilitySelection, b: StabilitySelection) -> float:
    """Jaccard overlap of two selections, for reporting stability directly.

    Two runs that select the same columns by threshold should overlap almost
    completely even when the *ranks* inside the selection differ. This is the
    number to quote when claiming a selection is reproducible.
    """
    sa, sb = set(a.columns), set(b.columns)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def summarize_direction(
    name: str,
    layer: int,
    vector: np.ndarray,
    *,
    n_tokens: int,
    contrast: float,
    k: int = 12,
) -> DirectionSummary:
    """Build the editor-facing summary of a captured direction."""
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    order = np.argsort(np.abs(vec))[::-1][:k]
    return DirectionSummary(
        name=name,
        layer=layer,
        dim=int(vec.shape[0]),
        norm=float(np.linalg.norm(vec)),
        top_dims=tuple((int(i), float(vec[i])) for i in order),
        n_tokens=n_tokens,
        contrast_cosine=contrast,
    )


def subtract_direction(
    weight: np.ndarray,
    cols: tuple[int, int] | list[int] | None,
    direction: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Return a copy of ``weight`` with ``scale * direction`` removed from ``cols``.

    ``direction`` is hidden-space and is broadcast down each selected column, so
    the edit removes the loop direction from those neurons' write vectors. This
    is the operation a scalar per-column subtraction cannot express.

    ``cols`` is either a half-open ``(start, stop)`` range or an explicit list of
    column indices. The list form matters: ``aligned_columns`` ranks columns by
    alignment and the top-ranked ones are scattered across the matrix, so a
    contiguous range forces the edit to include columns the evidence did not
    select. explore13's window ``900:1000`` contained only 3 of the 12 columns its
    own ranking identified.
    """
    matrix = np.asarray(weight, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got shape {matrix.shape}")
    vec = np.asarray(direction, dtype=np.float32).reshape(-1)
    if vec.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"direction has {vec.shape[0]} dims but weight has {matrix.shape[0]} rows"
        )

    out = matrix.copy()
    if cols is None:
        selection: slice | list[int] = slice(None)
    elif isinstance(cols, list):
        selection = cols
    else:
        selection = slice(*cols)
    out[:, selection] = out[:, selection] - np.float32(scale) * vec[:, None]
    return out
