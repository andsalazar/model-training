"""Turn a WSP direction into a specific, minimal set of Ovi weights to edit.

Where the residual stream is written
------------------------------------
Only two places per block add to the residual stream directly, and both are linear
layers whose output is the added term:

- ``self_attn.o`` — ``x = x + y * gate`` where ``y = self_attn(...)`` ends in ``.o``
- ``ffn[2]`` — ``src_seq = src_seq + y * gate`` where ``y = ffn(...)`` ends in ``.2``

Everything else in the block writes *into* those two paths. So a direction measured
in the residual stream maps onto exactly these matrices: a column of ``o`` (or of
``ffn[2]``) is one feature direction the layer can write, and the cosine between that
column and the captured direction says how much editing it can move the state the way
we want.

Why a threshold and not a top-k
-------------------------------
This is the lesson from Stage 2, recorded in :func:`wsp.activations.select_stable_columns`
and kept here: a rank cut is arbitrary inside a tie, and Stage 2's verdict turned on
exactly such a cut from a distribution tied to within 0.0012. Selection is by fraction
of the best alignment score, so it describes the score distribution rather than an
arbitrary position in it.

What this module deliberately does not do
-----------------------------------------
It does not edit weights. It produces a plan — which matrix, which columns, what
scale — and :func:`apply_plan` exists only so a caller has one obvious place to do
the edit and one place to undo it. The plan is serialisable, so a target chosen from
one capture can be inspected before anything is changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wsp.activations import (
    StabilitySelection,
    alignment_spectrum,
    select_stable_columns,
)

#: ``(stream, block attribute, weight path within the attribute)`` for the two sites
#: that write to the residual stream. Verified against Ovi's fusion block: the
#: self-attention output is scaled by ``e[2]`` and added, and the FFN output by ``e[5]``.
RESIDUAL_WRITERS: tuple[tuple[str, str], ...] = (
    ("self_attn", "o"),
    ("ffn", "2"),
)


@dataclass(frozen=True)
class TargetSite:
    """One weight matrix, the columns to edit, and how strongly."""

    stream: str
    layer: int
    attribute: str
    weight_path: str
    columns: tuple[int, ...]
    scores: tuple[float, ...]
    #: Fraction of the matrix frobenius norm each column carries, before damping.
    column_norms: tuple[float, ...]

    @property
    def name(self) -> str:
        return f"{self.stream}.blocks[{self.layer}].{self.attribute}.{self.weight_path}"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "stream": self.stream,
            "layer": self.layer,
            "attribute": self.attribute,
            "weight_path": self.weight_path,
            "n_columns": len(self.columns),
            "mean_score": float(np.mean(self.scores)) if self.scores else 0.0,
            "mean_column_norm": (
                float(np.mean(self.column_norms)) if self.column_norms else 0.0
            ),
        }


@dataclass
class TargetPlan:
    """A set of sites to edit, with the reasoning that produced them."""

    sites: list[TargetSite] = field(default_factory=list)
    ratio: float = 0.9
    #: Damping applied to the edit. Below 1.0 leaves most of the behaviour intact,
    #: which is the point: the claim under test is that a *small* targeted change
    #: moves the outcome, so a large one would not distinguish the hypothesis from
    #: "any perturbation changes something".
    strength: float = 0.02
    notes: list[str] = field(default_factory=list)

    @property
    def n_columns(self) -> int:
        return sum(len(site.columns) for site in self.sites)

    def as_dict(self) -> dict:
        return {
            "ratio": self.ratio,
            "strength": self.strength,
            "n_sites": len(self.sites),
            "n_columns": self.n_columns,
            "sites": [site.as_dict() for site in self.sites],
            "notes": self.notes,
        }


def _matrix_at(weight: np.ndarray, path: str) -> np.ndarray:
    """Resolve ``"o"`` or ``"2"`` to the actual array.

    ``ffn`` is a ``nn.Sequential`` whose second entry is the projection back to the
    residual width; the ``[hidden, hidden]`` square matrix is the one that writes.
    A separate helper keeps that shape check in one place instead of scattering
    index-into-Sequential assumptions through the caller.
    """
    matrix = np.asarray(weight, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D weight at {path}, got shape {matrix.shape}")
    return matrix


def find_targets(
    direction: np.ndarray,
    weights: dict[str, np.ndarray],
    *,
    stream: str,
    layer: int,
    ratio: float = 0.9,
    candidate_depth: int = 200,
    min_columns: int = 4,
) -> list[TargetSite]:
    """Locate the columns best aligned with ``direction`` in one block.

    ``weights`` is keyed ``"<attribute>.<path>"`` — the caller supplies the arrays,
    so this function needs neither torch nor a model and the targeting maths is
    testable on its own.
    """
    vec = np.asarray(direction, dtype=np.float32).reshape(-1)
    sites: list[TargetSite] = []
    for attribute, path in RESIDUAL_WRITERS:
        key = f"{attribute}.{path}"
        if key not in weights:
            continue
        matrix = _matrix_at(weights[key], key)
        if matrix.shape[0] != vec.shape[0]:
            # A direction captured at one width cannot be projected onto another.
            raise ValueError(
                f"{key} has {matrix.shape[0]} rows but the direction has "
                f"{vec.shape[0]} dims"
            )
        selection: StabilitySelection = select_stable_columns(
            matrix,
            vec,
            ratio=ratio,
            candidate_depth=min(candidate_depth, matrix.shape[1]),
            min_columns=min(min_columns, matrix.shape[1]),
        )
        if not selection.columns:
            continue
        norms = np.linalg.norm(matrix[:, list(selection.columns)], axis=0)
        sites.append(
            TargetSite(
                stream=stream,
                layer=layer,
                attribute=attribute,
                weight_path=path,
                columns=selection.columns,
                scores=selection.scores,
                column_norms=tuple(float(n) for n in norms),
            )
        )
    return sites


def build_plan(
    directions: dict[tuple[str, int], np.ndarray],
    weight_source,
    *,
    ratio: float = 0.9,
    strength: float = 0.02,
) -> TargetPlan:
    """Build a plan over every (stream, layer) for which a direction was captured.

    ``weight_source`` is a callable ``(stream, layer, key) -> ndarray | None``. It is
    injected rather than passed a model so the plan can be built from a saved capture
    and a safetensors file, without loading 23GB of weights onto a GPU.
    """
    plan = TargetPlan(ratio=ratio, strength=strength)
    for (stream, layer), direction in sorted(directions.items()):
        weights: dict[str, np.ndarray] = {}
        for attribute, path in RESIDUAL_WRITERS:
            key = f"{attribute}.{path}"
            matrix = weight_source(stream, layer, key)
            if matrix is not None:
                weights[key] = matrix
        if not weights:
            plan.notes.append(
                f"{stream} layer {layer}: no residual-writer weights found; skipped"
            )
            continue
        sites = find_targets(
            direction, weights, stream=stream, layer=layer, ratio=ratio
        )
        if not sites:
            plan.notes.append(
                f"{stream} layer {layer}: direction aligned with no column "
                "(all cosines non-positive); no edit proposed"
            )
            continue
        plan.sites.extend(sites)
    if not plan.sites:
        plan.notes.append(
            "no sites selected; the plan edits nothing, which is not evidence that "
            "the direction is wrong -- check that weight_source resolved any keys"
        )
    return plan


def damping_vector(site: TargetSite, strength: float) -> np.ndarray:
    """Per-column scale to apply to the selected columns.

    A *negative* scale shrinks a column: the column writes a feature, and damping it
    reduces how much of that feature the layer contributes. This is the mechanical
    form of the hypothesised TBC behaviour — quiet the directions that fire on
    something already present — expressed as an edit WSP can aim.

    Normalised by the column's own norm so a single ``strength`` means the same
    fraction of each column regardless of how large it happens to be.
    """
    if not 0.0 <= strength < 1.0:
        raise ValueError(f"strength must be in [0, 1), got {strength}")
    scales = np.full(len(site.columns), -float(strength), dtype=np.float32)
    return scales


@dataclass
class ApplyResult:
    """What an apply actually did, so a partial edit cannot report success."""

    changed: dict[str, dict] = field(default_factory=dict)
    skipped: tuple[str, ...] = ()
    requested: int = 0

    @property
    def complete(self) -> bool:
        return not self.skipped

    def as_dict(self) -> dict:
        return {
            "requested": self.requested,
            "applied": len(self.changed),
            "skipped": list(self.skipped),
            "complete": self.complete,
            "sites": dict(self.changed),
        }


def apply_plan(plan: TargetPlan, weight_source, weight_sink) -> ApplyResult:
    """Apply a plan through injected accessors.

    ``weight_source(stream, layer, key)`` returns the current matrix and
    ``weight_sink(stream, layer, key, matrix)`` writes it back. Both are injected so
    this function never has to know whether it is editing a torch module in place or
    a safetensors file on disk — and so a dry run can pass a sink that only records.

    A site whose weight cannot be found is **skipped and reported**, not silently
    dropped: a plan that claimed 8 sites and edited 6 must not report success.
    """
    result = ApplyResult(requested=len(plan.sites))
    for site in plan.sites:
        key = f"{site.attribute}.{site.weight_path}"
        matrix = weight_source(site.stream, site.layer, key)
        if matrix is None:
            result.skipped += (site.name,)
            continue
        updated = np.array(matrix, dtype=np.float32, copy=True)
        scales = damping_vector(site, plan.strength)
        cols = list(site.columns)
        # Linear stores weight as [out, in]: row = output feature, column = input
        # feature. Column j is therefore "how every output reads input feature j",
        # so scaling column j damps that input direction at every output — which is
        # what a residual-stream direction identifies.
        updated[:, cols] *= (1.0 + scales)
        weight_sink(site.stream, site.layer, key, updated)
        delta = float(
            np.linalg.norm(updated - matrix) / max(np.linalg.norm(matrix), 1e-12)
        )
        result.changed[site.name] = {
            "columns": len(cols),
            "relative_change": round(delta, 6),
        }
    if result.skipped:
        plan.notes.append(
            f"{len(result.skipped)} of {len(plan.sites)} sites were not found and "
            f"were left unedited: {', '.join(result.skipped)}"
        )
    return result
