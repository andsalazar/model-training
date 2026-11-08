"""Attention-map contrast: the measurement the residual result motivates.

Why this module exists
----------------------
Stage 6c measured a drifting-versus-stable direction in Ovi's *residual stream* and
projected it onto the two matrices that write to that stream (``self_attn.o`` and
``ffn.2``). The best column alignment was 0.129 against a 0.018 chance level — above
chance, but far too weak to justify an edit whose effect would be 0.1-0.3% of the
state norm.

The most likely explanation is structural, not a failure of the instrument. A
residual-stream state is a **sum** of every component's contribution: the attention
output, the FFN output, the timestep embedding, the text conditioning. A direction
derived from that sum is dominated by whichever component contributes most, and there
is no reason for that to be the component that distinguishes drifting from stable.

What distinguishes a stable rollout from a drifting one may not be *what* is written
to the residual stream but *where each token reads from* — the attention pattern. That
is not representable as a column of ``o`` or ``ffn.2``: ``o`` maps the attention output
back to the residual width, but the pattern itself lives in the ``[heads, query, key]``
tensor before that projection. A damping edit on ``o`` cannot reach it.

So this module measures the attention pattern directly and asks the question the
residual projection could not answer: **does the drift-to-stable difference show up in
attention, and if so, in which heads and which query/key blocks?**

What it measures
----------------
``AttentionMap``
    A reduced summary of one ``[heads, queries, keys]`` attention tensor. Full maps are
    enormous — 24 heads x 27,900 queries x 27,900 keys is 18.7 billion entries — so the
    summary keeps what a contrast needs: per-head row entropy (how spread each query's
    attention is), per-head column mass (which keys are read), and the diagonal mass
    (how much each token attends to itself).

``attention_contrast``
    The difference between two summaries, per head. Reports which heads separate and by
    how much, so a targeted edit has somewhere to aim.

``head_ranking``
    Orders heads by how much they separate the two conditions, with a stability check
    so a head that separates by noise is not ranked first. This is the attention
    analogue of the column ranking in :mod:`wsp.targeting`.

Why entropy and mass rather than the raw map
--------------------------------------------
The raw map cannot be compared across rollouts: the two runs visit different timesteps,
so their query/key sets differ in content even when the shapes match. Entropy and
column mass are *aggregates* over the key axis, so they are comparable when the shapes
match and interpretable when they do not. This is the same reasoning that made delta
entropy work where per-frame entropy did not in :mod:`wsp.entropy`.

What this module deliberately does NOT do
-----------------------------------------
It does not import torch and does not attach hooks. The reductions are pure array work,
so they are testable without a GPU. Attaching an attention hook needs the real model and
lives in :mod:`wsp.hooks`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Attention probabilities below this are treated as zero when computing entropy.
#: ``log(0)`` is ``-inf``, and a single zero-probability key would otherwise poison the
#: whole row's entropy. FlashAttention's output is already normalised, so this only
#: guards against float underflow, not against genuinely absent mass.
_EPS = 1e-12


def _as_attention(attention: np.ndarray) -> np.ndarray:
    """Coerce a hooked attention tensor to ``[heads, queries, keys]``.

    DiTs pass attention as ``[batch, heads, queries, keys]``. Batch is normally 1 for a
    single generation and is squeezed; a genuine multi-batch capture is rejected rather
    than averaged, because averaging across generations would mix two rollouts.
    """
    array = np.asarray(attention, dtype=np.float32)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(
                f"expected batch size 1, got {array.shape[0]}; "
                "average per-batch rather than across generations"
            )
        array = array[0]
    if array.ndim != 3:
        raise ValueError(
            f"expected [heads, queries, keys], got shape {array.shape}"
        )
    if array.shape[1] != array.shape[2]:
        raise ValueError(
            f"attention must be square in query/key, got "
            f"{array.shape[1]} queries x {array.shape[2]} keys"
        )
    return array


@dataclass
class AttentionMap:
    """A reduced summary of one attention tensor, keyed by head.

    ``row_entropy`` is the mean Shannon entropy of each query's attention distribution,
    in nats. A low value means each query reads from few keys — a focused, possibly
    collapsed pattern. A high value means attention is spread across many keys.

    ``column_mass`` is the mean attention each key receives, normalised to sum to 1
    across keys. It says *which* keys are read, which row entropy cannot.

    ``diagonal_mass`` is the mean attention each query places on its own key. High
    diagonal mass means tokens mostly attend to themselves, which in a video model
    means each patch reads its own position rather than its neighbours — a plausible
    signature of a rollout that has stopped propagating structure across space.
    """

    heads: int
    queries: int
    keys: int
    row_entropy: np.ndarray
    column_mass: np.ndarray
    diagonal_mass: np.ndarray
    layer: int = 0
    timestep: int = 0
    stream: str = "video"
    #: Per-head variance of row entropy across queries. ``row_entropy`` is already the
    #: per-head mean, so the within-head spread an effect size needs is not recoverable
    #: from it; it has to be accumulated alongside. ``None`` for maps built before this
    #: field existed.
    row_entropy_var: np.ndarray | None = None

    def summary(self) -> dict:
        return {
            "stream": self.stream,
            "layer": self.layer,
            "timestep": self.timestep,
            "heads": self.heads,
            "queries": self.queries,
            "keys": self.keys,
            "row_entropy_mean": float(self.row_entropy.mean()),
            # Spread of the per-head means across heads, not across queries.
            "row_entropy_std": float(self.row_entropy.std()),
            "diagonal_mass_mean": float(self.diagonal_mass.mean()),
            # Per head, then averaged. Flattening the [heads, keys] array first would
            # mix heads into one distribution and report log(heads * keys) for a
            # uniform map, which is not a property of any single head.
            "column_mass_entropy": float(
                np.mean(
                    [
                        _entropy_of_distribution(self.column_mass[head])
                        for head in range(self.heads)
                    ]
                )
            ),
        }


def _entropy_of_distribution(probabilities: np.ndarray) -> float:
    """Shannon entropy in nats of a 1-D probability vector."""
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    total = p.sum()
    if total <= 0:
        raise ValueError("distribution sums to zero; entropy is undefined")
    p = p / total
    p = np.clip(p, _EPS, None)
    return float(-(p * np.log(p)).sum())


def summarize_attention(
    attention: np.ndarray,
    layer: int = 0,
    timestep: int = 0,
    stream: str = "video",
) -> AttentionMap:
    """Reduce a ``[heads, queries, keys]`` attention tensor to an :class:`AttentionMap`.

    The reductions are chosen so the result is comparable across rollouts that visit
    different timesteps: every field is an aggregate over the key axis, so it does not
    depend on which specific keys are present, only on how mass is distributed over
    them.
    """
    array = _as_attention(attention)
    heads, queries, keys = array.shape

    # Row-normalise defensively. FlashAttention returns normalised probabilities, but a
    # hook on a pre-softmax tensor would not, and a silent mis-scale would make every
    # entropy value wrong in a plausible-looking way.
    row_sums = array.sum(axis=-1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError(
            "attention rows sum to zero; the tensor is not a probability "
            "distribution (a pre-softmax hook would look like this)"
        )
    probabilities = array / row_sums

    clipped = np.clip(probabilities, _EPS, None)
    per_query = -(probabilities * np.log(clipped)).sum(axis=-1)
    row_entropy = per_query.mean(axis=-1)
    row_entropy_var = per_query.var(axis=-1)

    column_mass = probabilities.mean(axis=1)
    column_mass = column_mass / column_mass.sum(axis=-1, keepdims=True)

    diagonal = np.einsum("hii->hi", probabilities).mean(axis=-1)

    return AttentionMap(
        heads=heads,
        queries=queries,
        keys=keys,
        row_entropy=row_entropy.astype(np.float32),
        column_mass=column_mass.astype(np.float32),
        diagonal_mass=diagonal.astype(np.float32),
        layer=layer,
        timestep=timestep,
        stream=stream,
        row_entropy_var=row_entropy_var.astype(np.float32),
    )


#: Query rows processed per chunk by :func:`summarize_attention_chunked`. 512 rows x
#: 27,900 keys x 24 heads in float32 is ~1.4 GB, which fits beside the model on an 80 GB
#: card without spilling.
DEFAULT_CHUNK = 512


def summarize_attention_chunked(
    q: np.ndarray,
    k: np.ndarray,
    layer: int = 0,
    timestep: int = 0,
    stream: str = "video",
    softmax_scale: float | None = None,
    chunk: int = DEFAULT_CHUNK,
) -> AttentionMap:
    """Compute an :class:`AttentionMap` from ``q``/``k`` without the full matrix.

    ``summarize_attention`` needs the whole ``[heads, queries, keys]`` tensor in memory.
    At 960x960 that is 18.7 billion entries — 74 GB in float32 for one layer — so the
    real model cannot produce one. This computes the *same* fields from the queries and
    keys directly, in query chunks, so peak memory is ``chunk x keys`` per head.

    The result is identical to ``summarize_attention`` on the materialised softmax: the
    softmax is taken over the key axis, and row entropy and column mass are both
    reductions over that same axis. Chunking the query axis does not change either.

    ``q`` is ``[heads, queries, head_dim]`` and ``k`` is ``[heads, keys, head_dim]`` —
    the layout Ovi's ``flash_attention`` receives after its own reshape. Queries and keys
    may differ in length (cross-attention); ``diagonal_mass`` is then zero, because a
    query has no "own key" when the key set is a different sequence.
    """
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(
            f"expected q and k as [heads, tokens, head_dim], got {q.shape} and {k.shape}"
        )
    heads, queries, head_dim = q.shape
    k_heads, keys, k_dim = k.shape
    if heads != k_heads:
        raise ValueError(f"head count mismatch: q has {heads}, k has {k_heads}")
    if head_dim != k_dim:
        raise ValueError(f"head dim mismatch: q has {head_dim}, k has {k_dim}")
    if queries == 0 or keys == 0:
        raise ValueError(f"empty attention: {queries} queries x {keys} keys")

    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5

    row_entropy = np.zeros(heads, dtype=np.float64)
    row_entropy_sq = np.zeros(heads, dtype=np.float64)
    column_mass = np.zeros((heads, keys), dtype=np.float64)
    diagonal = np.zeros(heads, dtype=np.float64)

    for start in range(0, queries, chunk):
        stop = min(start + chunk, queries)
        # [heads, chunk, keys]
        scores = np.einsum("hqd,hkd->hqk", q[:, start:stop], k) * scale
        scores -= scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)

        clipped = np.clip(weights, _EPS, None)
        per_query = -(weights * np.log(clipped)).sum(axis=-1)
        row_entropy += per_query.sum(axis=1)
        row_entropy_sq += (per_query**2).sum(axis=1)
        column_mass += weights.sum(axis=1)

        if queries == keys:
            idx = np.arange(start, stop)
            diagonal += weights[:, idx - start, idx].sum(axis=1)

    row_entropy /= queries
    row_entropy_var = np.maximum(row_entropy_sq / queries - row_entropy**2, 0.0)
    column_mass /= queries
    totals = column_mass.sum(axis=1, keepdims=True)
    column_mass = np.divide(
        column_mass, totals, out=np.zeros_like(column_mass), where=totals > 0
    )
    diagonal /= queries

    return AttentionMap(
        heads=heads,
        queries=queries,
        keys=keys,
        row_entropy=row_entropy.astype(np.float32),
        column_mass=column_mass.astype(np.float32),
        diagonal_mass=diagonal.astype(np.float32),
        layer=layer,
        timestep=timestep,
        stream=stream,
        row_entropy_var=row_entropy_var.astype(np.float32),
    )


def summarize_attention_torch(
    q: Any,
    k: Any,
    layer: int = 0,
    timestep: int = 0,
    stream: str = "video",
    softmax_scale: float | None = None,
    chunk: int = 512,
) -> AttentionMap:
    """GPU version of :func:`summarize_attention_chunked`.

    The numpy path is the reference and stays the tested one, but it cannot run on the
    real model: a 27,900-key attention call is 18.7 billion score entries, and reducing
    that in numpy took 49 minutes of CPU time for a single run. The same arithmetic in
    torch on the GPU is seconds.

    ``q`` and ``k`` are torch tensors shaped ``[heads, queries, head_dim]``. The result is
    an :class:`AttentionMap` with numpy arrays, so downstream code does not care which path
    produced it. Equivalence with the numpy path is tested in ``tests/test_attention.py``.
    """
    import torch

    if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor):
        raise TypeError("summarize_attention_torch needs torch tensors")
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(
            f"expected q and k as [heads, tokens, head_dim], got {q.shape} and {k.shape}"
        )
    heads, queries, head_dim = q.shape
    k_heads, keys, k_dim = k.shape
    if heads != k_heads:
        raise ValueError(f"head count mismatch: q has {heads}, k has {k_heads}")
    if head_dim != k_dim:
        raise ValueError(f"head dim mismatch: q has {head_dim}, k has {k_dim}")
    if queries == 0 or keys == 0:
        raise ValueError(f"empty attention: {queries} queries x {keys} keys")

    # Accumulate in float32 on the GPU. bfloat16 scores would lose the tail of the
    # distribution, which is exactly what entropy measures.
    qf = q.detach().to(torch.float32)
    kf = k.detach().to(torch.float32)
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5

    row_entropy = torch.zeros(heads, dtype=torch.float64, device=q.device)
    row_entropy_sq = torch.zeros(heads, dtype=torch.float64, device=q.device)
    column_mass = torch.zeros((heads, keys), dtype=torch.float64, device=q.device)
    diagonal = torch.zeros(heads, dtype=torch.float64, device=q.device)

    for start in range(0, queries, chunk):
        stop = min(start + chunk, queries)
        # [heads, chunk, keys]
        scores = torch.einsum("hqd,hkd->hqk", qf[:, start:stop], kf) * scale
        scores -= scores.max(dim=-1, keepdim=True).values
        weights = torch.softmax(scores, dim=-1)
        weights64 = weights.to(torch.float64)

        clipped = weights64.clamp_min(_EPS)
        per_query = -(weights64 * clipped.log()).sum(dim=-1)
        row_entropy += per_query.sum(dim=1)
        row_entropy_sq += (per_query**2).sum(dim=1)
        column_mass += weights64.sum(dim=1)

        if queries == keys:
            idx = torch.arange(start, stop, device=q.device)
            diagonal += weights64[:, idx - start, idx].sum(dim=1)

    row_entropy /= queries
    row_entropy_var = (row_entropy_sq / queries - row_entropy**2).clamp_min(0.0)
    column_mass /= queries
    totals = column_mass.sum(dim=1, keepdim=True)
    column_mass = torch.where(totals > 0, column_mass / totals, torch.zeros_like(column_mass))
    diagonal /= queries

    return AttentionMap(
        heads=heads,
        queries=queries,
        keys=keys,
        row_entropy=row_entropy.cpu().numpy().astype(np.float32),
        column_mass=column_mass.cpu().numpy().astype(np.float32),
        diagonal_mass=diagonal.cpu().numpy().astype(np.float32),
        layer=layer,
        timestep=timestep,
        stream=stream,
        row_entropy_var=row_entropy_var.cpu().numpy().astype(np.float32),
    )


@dataclass
class HeadContrast:
    """How much one head separates two conditions."""

    head: int
    entropy_delta: float
    diagonal_delta: float
    column_mass_l1: float
    #: See ``effect_kind`` for what this number is.
    effect_size: float
    #: ``"pooled_sd"``: ``|delta| / pooled within-head std across queries`` (a Cohen's-d
    #: style effect). ``"relative"``: ``|delta| / mean entropy level`` — used when either
    #: map lacks per-query variance. The two are not on the same scale.
    effect_kind: str = "pooled_sd"

    def summary(self) -> dict:
        return {
            "head": self.head,
            "entropy_delta": round(self.entropy_delta, 6),
            "diagonal_delta": round(self.diagonal_delta, 6),
            "column_mass_l1": round(self.column_mass_l1, 6),
            "effect_size": round(self.effect_size, 4),
            "effect_kind": self.effect_kind,
        }


def attention_contrast(
    a: AttentionMap, b: AttentionMap, min_effect: float = 1.0
) -> list[HeadContrast]:
    """Per-head difference between two attention summaries.

    ``a`` is the condition of interest (for example the drifting rollout) and ``b`` the
    reference (the stable one), so a positive ``entropy_delta`` means ``a`` is more
    spread. The sign convention matters: an earlier stage in this project shipped an
    inverted ``direction_similarity`` sign and reported the opposite conclusion.

    ``effect_size`` is the difference divided by the pooled standard deviation of the
    two heads' per-query row entropies (``effect_kind="pooled_sd"``). It is a crude but
    honest guard: without it, a head whose entropy differs by 0.001 while its own
    queries vary by 0.5 would rank alongside a head that genuinely separates. When a map
    carries no per-query variance, or the pooled spread is zero, it falls back to the
    relative change ``|delta| / mean level`` and says so with ``effect_kind="relative"``.

    ``min_effect`` filters the returned list. Pass ``0`` to see every head.
    """
    if a.heads != b.heads:
        raise ValueError(f"head count mismatch: {a.heads} vs {b.heads}")
    if a.queries != b.queries or a.keys != b.keys:
        raise ValueError(
            f"attention shape mismatch: {a.queries}x{a.keys} vs "
            f"{b.queries}x{b.keys}"
        )

    results: list[HeadContrast] = []
    for head in range(a.heads):
        entropy_delta = float(a.row_entropy[head] - b.row_entropy[head])
        diagonal_delta = float(a.diagonal_mass[head] - b.diagonal_mass[head])
        column_mass_l1 = float(
            np.abs(a.column_mass[head] - b.column_mass[head]).sum()
        )
        # A head that did not move is not a contrast, whatever the threshold. Without
        # this guard, min_effect=0.0 admits every head including identical ones, and a
        # caller asking for "all heads" gets a list of zeros that reads like a finding.
        if (
            abs(entropy_delta) <= _EPS
            and abs(diagonal_delta) <= _EPS
            and column_mass_l1 <= _EPS
        ):
            continue
        # Pooled spread of the two heads' per-query entropies. This is a within-head
        # spread, not a between-condition one, so it measures how much this head varies
        # across queries rather than how much the conditions differ. ``row_entropy`` is
        # the per-head mean, so the variance must come from ``row_entropy_var``; an
        # earlier version took np.var of the scalar mean, which is always 0, so every
        # contrast silently used the relative fallback below.
        pooled = 0.0
        if a.row_entropy_var is not None and b.row_entropy_var is not None:
            pooled = float(
                np.sqrt((a.row_entropy_var[head] + b.row_entropy_var[head]) / 2.0)
            )
        if pooled > _EPS:
            effect = abs(entropy_delta) / pooled
            kind = "pooled_sd"
        else:
            # No within-head spread to normalise by. Dividing by zero would report inf
            # and returning 0.0 would hide a real difference -- the "tool that cannot
            # look must say so" failure. Scale by the head's own entropy level instead
            # and label it, because it is not on the pooled-SD scale.
            level = float(
                np.mean([a.row_entropy[head], b.row_entropy[head]])
            )
            effect = abs(entropy_delta) / max(level, _EPS)
            kind = "relative"
        if effect < min_effect:
            continue
        results.append(
            HeadContrast(
                head=head,
                entropy_delta=entropy_delta,
                diagonal_delta=diagonal_delta,
                column_mass_l1=column_mass_l1,
                effect_size=effect,
                effect_kind=kind,
            )
        )
    return results


def head_ranking(
    contrasts: list[HeadContrast], by: str = "effect_size"
) -> list[HeadContrast]:
    """Order heads by how much they separate the conditions, largest first.

    ``by`` selects the criterion: ``effect_size`` (the default, which normalises by the
    head's own spread), ``entropy_delta`` (raw magnitude), or ``column_mass_l1`` (how
    much the read pattern moved).

    Ties are broken by head index so the ordering is deterministic. Stage 2's
    reproducibility failure came from a top-k cut sitting inside a plateau of
    near-equal scores, so a deterministic tie-break is not cosmetic here.
    """
    if by not in {"effect_size", "entropy_delta", "column_mass_l1"}:
        raise ValueError(
            f"by must be effect_size, entropy_delta or column_mass_l1, got {by!r}"
        )
    key = lambda c: (-abs(getattr(c, by)), c.head)  # noqa: E731
    return sorted(contrasts, key=key)


@dataclass
class AttentionCapture:
    """Accumulates attention summaries, keyed by (stream, layer, timestep)."""

    maps: dict[tuple[str, int, int], AttentionMap] = field(default_factory=dict)
    #: Calls the schedule declined to record, so a schedule that matched nothing is
    #: visible rather than looking like a capture that found no structure.
    skipped: int = 0

    def record(
        self,
        layer: int,
        timestep: int,
        attention: np.ndarray,
        stream: str = "video",
        keep: bool = True,
    ) -> bool:
        """Record one hooked attention tensor. Returns whether it was kept."""
        if not keep:
            self.skipped += 1
            return False
        self.maps[(stream, layer, timestep)] = summarize_attention(
            attention, layer=layer, timestep=timestep, stream=stream
        )
        return True

    def record_summary(
        self,
        layer: int,
        timestep: int,
        summary: AttentionMap,
        stream: str = "video",
        keep: bool = True,
    ) -> bool:
        """Record an already-summarised map. Returns whether it was kept.

        The GPU path cannot materialise a full attention tensor, so it computes the
        summary from ``q``/``k`` directly and hands the result here. ``record`` would
        try to re-summarise it as if it were a raw tensor and fail.
        """
        if not keep:
            self.skipped += 1
            return False
        self.maps[(stream, layer, timestep)] = summary
        return True

    def at(self, layer: int, timestep: int, stream: str = "video") -> AttentionMap:
        key = (stream, layer, timestep)
        if key not in self.maps:
            raise KeyError(
                f"nothing captured for stream={stream} layer {layer} "
                f"timestep {timestep}; available: {sorted(self.maps)}"
            )
        return self.maps[key]

    def layers(self, stream: str = "video") -> list[int]:
        return sorted({layer for s, layer, _ in self.maps if s == stream})

    def timesteps(self, stream: str = "video") -> list[int]:
        return sorted({step for s, _, step in self.maps if s == stream})

    def summary(self) -> dict:
        return {
            "n_maps": len(self.maps),
            "n_skipped": self.skipped,
            "streams": sorted({s for s, _, _ in self.maps}),
            "layers": {s: self.layers(s) for s in sorted({s for s, _, _ in self.maps})},
            "timesteps": {
                s: self.timesteps(s) for s in sorted({s for s, _, _ in self.maps})
            },
        }
