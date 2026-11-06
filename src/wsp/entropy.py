"""Differential entropy across generated frames, as an information-collapse proxy.

TBC's adapter is judged by "differential entropy across generated frames" as a
proxy for information collapse over long rollouts. That metric is the whole
target of the Ovi project: without it there is no behaviour to measure and
therefore nothing for WSP to derive an edit from.

The metric is deliberately implemented as pure array math so it can be tested
without a GPU or a video model. The model-side work — decoding latents to frames,
running Ovi — lives elsewhere. Everything here takes arrays in and returns
numbers out.

Why differential entropy and not pixel variance
-----------------------------------------------
Variance measures spread; differential entropy measures *information*. A rollout
can keep healthy pixel variance while collapsing into a repeating pattern, and it
can have low variance while still carrying structure. TBC's failure mode is the
former: the model drifts into degenerate, low-diversity states. Entropy is the
quantity that separates those cases, which is why TBC chose it.

The estimator matters
---------------------
Differential entropy of a continuous variable is not directly estimable by
binning without bias, and the bias grows as the number of bins grows relative to
the sample count. Three estimators are provided because they fail differently:

``histogram``
    Binned plug-in estimate. Simple and interpretable, but biased downward and
    sensitive to bin width. Useful as a sanity check, not as the primary metric.

``knn``
    Kozachenko-Leonenko k-nearest-neighbour estimator. Consistent and
    parameter-light (only ``k``), but degrades in high dimensions because
    distances concentrate. This is the default for per-frame latent patches.

``gaussian``
    Closed-form entropy of a fitted Gaussian. Exact when the data is Gaussian and
    cheap in any dimension, but blind to non-Gaussian structure — which is
    precisely the structure a collapse produces. Reported for comparison, not as
    the verdict.

The collapse signal is the *trend across frames*, not the absolute value. A
healthy rollout holds entropy roughly flat; a collapsing one falls. So the
primary output is a slope and a ratio, both scale-free, rather than a raw number
that would depend on the estimator and the frame count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Below this many samples the k-NN estimator is dominated by its own bias.
_MIN_SAMPLES_FOR_KNN = 32

#: A trend is only reported as a collapse when the fit is at least this good.
#: A poor fit means the entropy series is not monotone, so a slope would be
#: meaningless even if its sign looked right.
_MIN_TREND_R2 = 0.5


def _as_2d(x: np.ndarray) -> np.ndarray:
    """Coerce to ``(n_samples, n_dims)`` without copying when already 2-D."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D array, got shape {arr.shape}")
    return arr


def histogram_entropy(x: np.ndarray, bins: int = 32) -> float:
    """Binned plug-in differential entropy estimate.

    Biased downward, and the bias grows with ``bins`` relative to sample count.
    Kept because it is the estimator a reader is most likely to reach for, and
    disagreement with the k-NN estimate is itself informative.
    """
    arr = _as_2d(x)
    if arr.shape[0] < 2:
        return 0.0
    # Per-dimension binning, then average: a joint histogram over many dimensions
    # would be empty at any practical bin count.
    entropies = []
    for dim in range(arr.shape[1]):
        column = arr[:, dim]
        counts, edges = np.histogram(column, bins=bins)
        widths = np.diff(edges)
        # Drop empty bins: their log is undefined and they contribute nothing.
        occupied = counts > 0
        if not occupied.any():
            entropies.append(0.0)
            continue
        probs = counts[occupied] / counts.sum()
        entropies.append(float(-np.sum(probs * np.log(probs / widths[occupied]))))
    return float(np.mean(entropies))


def _digamma(x: float) -> float:
    """Digamma function for positive real ``x``.

    Implemented here rather than pulled from scipy so the metric has no dependency
    beyond numpy. Uses the recurrence ``psi(x+1) = psi(x) + 1/x`` to push the
    argument above 6, then the asymptotic expansion, which is accurate to well
    beyond the precision this metric needs at that point.
    """
    from math import log

    result = 0.0
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    # Asymptotic expansion: ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - ...
    inv = 1.0 / x
    inv2 = inv * inv
    return result + log(x) - 0.5 * inv - inv2 / 12.0 + inv2 * inv2 / 120.0


def knn_entropy(x: np.ndarray, k: int = 5) -> float:
    """Kozachenko-Leonenko k-nearest-neighbour differential entropy estimate.

    The estimator is

        h = psi(n) - psi(k) + ln(c_d) + d * mean(ln(eps_i))

    where ``c_d = pi^(d/2) / Gamma(d/2 + 1)`` is the unit-ball volume and ``eps_i``
    is the distance from sample ``i`` to its k-th nearest neighbour. The ``d``
    factor on the mean log-distance is easy to drop and dropping it makes the
    estimate collapse toward zero as the dimension grows — which reads as "no
    information" for data that has plenty. It is applied explicitly here.

    Consistent and parameter-light, but distances concentrate in high dimensions,
    so the estimate flattens as ``n_dims`` grows. For per-frame latent patches
    that flattening is tolerable because the comparison is between frames under
    the same estimator; it would not be tolerable as an absolute measurement.
    """
    arr = _as_2d(x)
    n, dim = arr.shape
    if n < _MIN_SAMPLES_FOR_KNN:
        # Fall back rather than return a number the sample count cannot support.
        return gaussian_entropy(arr)
    k = max(1, min(k, n - 1))

    # Pairwise distances. Fine at the sizes this is used on (a few thousand
    # patches); a blocked implementation would be needed for full-frame volumes.
    diff = arr[:, None, :] - arr[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    # k-th nearest neighbour distance, 0-indexed.
    kth = np.partition(dist, k - 1, axis=1)[:, k - 1]
    kth = np.maximum(kth, np.finfo(np.float64).tiny)

    from math import lgamma, log, pi

    unit_ball = (dim / 2.0) * log(pi) - lgamma(dim / 2.0 + 1.0)
    return float(
        _digamma(n) - _digamma(k) + unit_ball + dim * float(np.mean(np.log(kth)))
    )


def gaussian_entropy(x: np.ndarray) -> float:
    """Closed-form differential entropy of a fitted Gaussian.

    Exact for Gaussian data and cheap in any dimension, but blind to the
    non-Gaussian structure that a collapse actually produces. Reported for
    comparison; not the verdict.
    """
    arr = _as_2d(x)
    n, dim = arr.shape
    if n < 2:
        return 0.0
    # Covariance with the sample correction; a singular covariance (fewer samples
    # than dimensions) is regularised so the log-determinant stays finite.
    cov = np.cov(arr, rowvar=False)
    cov = np.atleast_2d(cov)
    if cov.shape[0] != dim:
        cov = cov.reshape(dim, dim)
    ridge = 1e-10 * max(1.0, float(np.trace(cov)) / max(dim, 1))
    cov = cov + ridge * np.eye(dim)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return 0.0
    return float(0.5 * (dim * (1.0 + np.log(2.0 * np.pi)) + logdet))


_ESTIMATORS = {
    "histogram": histogram_entropy,
    "knn": knn_entropy,
    "gaussian": gaussian_entropy,
}


@dataclass(frozen=True)
class EntropyReading:
    """One frame's entropy, plus enough context to judge whether it is usable."""

    frame: int
    entropy: float
    estimator: str
    n_samples: int
    n_dims: int
    #: True when the sample count was too low for the requested estimator and a
    #: fallback was used. A reading that silently changed estimator is not
    #: comparable to its neighbours, so this has to travel with the number.
    fell_back: bool = False

    def as_dict(self) -> dict:
        return {
            "frame": self.frame,
            "entropy": self.entropy,
            "estimator": self.estimator,
            "n_samples": self.n_samples,
            "n_dims": self.n_dims,
            "fell_back": self.fell_back,
        }


def frame_entropy(
    frame: np.ndarray, frame_index: int = 0, estimator: str = "knn", **kwargs: object
) -> EntropyReading:
    """Entropy of a single frame's latent patches.

    ``frame`` is ``(n_patches, n_dims)`` — the latent patches of one generated
    frame, flattened. A 1-D array is treated as a single dimension over samples.
    """
    if estimator not in _ESTIMATORS:
        raise ValueError(
            f"unknown estimator {estimator!r}; expected one of {sorted(_ESTIMATORS)}"
        )
    arr = _as_2d(frame)
    n, dim = arr.shape
    fell_back = estimator == "knn" and n < _MIN_SAMPLES_FOR_KNN
    value = _ESTIMATORS[estimator](arr, **kwargs)  # type: ignore[operator]
    return EntropyReading(
        frame=frame_index,
        entropy=float(value),
        estimator=estimator,
        n_samples=int(n),
        n_dims=int(dim),
        fell_back=fell_back,
    )


@dataclass(frozen=True)
class CollapseReading:
    """The trend across frames — the actual collapse signal.

    The absolute entropy depends on the estimator and the frame count, so it is
    not comparable across runs. The slope and the ratio are scale-free, which is
    why they carry the verdict.
    """

    readings: tuple[EntropyReading, ...]
    #: Least-squares slope of entropy against frame index, in entropy units per
    #: frame. Negative means entropy is falling, i.e. collapsing.
    slope: float
    #: R² of that fit. A low value means the series is not monotone and the slope
    #: should not be trusted.
    r_squared: float
    #: Mean entropy over the first third of the rollout.
    early_mean: float
    #: Mean entropy over the last third.
    late_mean: float
    #: ``late_mean / early_mean``. Below 1 means entropy fell. ``nan`` when the
    #: early mean is not positive — differential entropy is a relative measure and
    #: can be negative, and a ratio across zero is not meaningful. Check
    #: :attr:`ratio_defined` before using it.
    ratio: float
    #: True when the trend is negative and the fit is good enough to believe.
    collapsing: bool
    #: False when :attr:`ratio` is ``nan`` because the early mean was not positive.
    #: Callers must fall back to comparing slopes in that case.
    ratio_defined: bool = True

    def as_dict(self) -> dict:
        return {
            "frames": [r.as_dict() for r in self.readings],
            "slope": self.slope,
            "r_squared": self.r_squared,
            "early_mean": self.early_mean,
            "late_mean": self.late_mean,
            "ratio": self.ratio,
            "ratio_defined": self.ratio_defined,
            "collapsing": self.collapsing,
        }


def _linear_fit(y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope and R² of ``y`` against its index."""
    n = y.size
    if n < 2:
        return 0.0, 0.0
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = float(np.sum((x - x_mean) ** 2))
    if denom == 0.0:
        return 0.0, 0.0
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    predicted = y_mean + slope * (x - x_mean)
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    # A perfectly flat series has no variance to explain; call that a good fit
    # with zero slope rather than an undefined R².
    r_squared = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return slope, r_squared


def collapse_trend(
    readings: list[EntropyReading] | tuple[EntropyReading, ...],
    min_r2: float = _MIN_TREND_R2,
) -> CollapseReading:
    """Fit the entropy trend across frames and decide whether it is collapsing.

    ``collapsing`` requires both a negative slope and a fit good enough to
    believe. Requiring only the sign would call a noisy series a collapse on the
    strength of one bad frame — the same class of error that made three earlier
    substring tests report successes that were not there.
    """
    seq = tuple(readings)
    if not seq:
        raise ValueError("need at least one entropy reading")
    if len(seq) < 3:
        # Two points always fit a line perfectly, so R² carries no information.
        # Report the slope but never call it a collapse.
        y = np.array([r.entropy for r in seq], dtype=np.float64)
        slope, r2 = _linear_fit(y)
        mean = float(y.mean())
        return CollapseReading(
            readings=seq,
            slope=slope,
            r_squared=r2,
            early_mean=mean,
            late_mean=mean,
            ratio=1.0 if mean > 0.0 else float("nan"),
            collapsing=False,
            ratio_defined=mean > 0.0,
        )

    y = np.array([r.entropy for r in seq], dtype=np.float64)
    slope, r2 = _linear_fit(y)
    third = max(1, len(y) // 3)
    early = float(y[:third].mean())
    late = float(y[-third:].mean())
    # Differential entropy is not guaranteed positive — it is a relative measure
    # and can be negative for concentrated distributions. A ratio across zero is
    # not meaningful, so it is reported as nan with ratio_defined False rather than
    # silently returning a number that looks usable. Callers fall back to slopes.
    ratio_defined = early > 0.0
    ratio = late / early if ratio_defined else float("nan")
    return CollapseReading(
        readings=seq,
        slope=slope,
        r_squared=r2,
        early_mean=early,
        late_mean=late,
        ratio=ratio,
        collapsing=slope < 0.0 and r2 >= min_r2,
        ratio_defined=ratio_defined,
    )


def entropy_series(
    frames: list[np.ndarray] | tuple[np.ndarray, ...],
    estimator: str = "knn",
    **kwargs: object,
) -> CollapseReading:
    """Entropy across a whole rollout, one reading per frame."""
    readings = [
        frame_entropy(frame, frame_index=i, estimator=estimator, **kwargs)
        for i, frame in enumerate(frames)
    ]
    return collapse_trend(readings)


def frame_delta_entropy(
    frames: list[np.ndarray] | tuple[np.ndarray, ...], estimator: str = "knn", **kwargs: object
) -> CollapseReading:
    """Entropy of the *change* between consecutive frames.

    This is the metric that matches "respond only to what's changing". A healthy
    rollout keeps producing change; a collapsing one stops — consecutive frames
    become near-identical, so the delta entropy falls toward zero.

    It is a different quantity from :func:`entropy_series`, which measures the
    information in each frame independently. A rollout can hold per-frame entropy
    flat while its delta entropy falls to zero: every frame is rich, but they are
    all the same frame. That is exactly the degenerate state TBC's metric is
    meant to catch, so both are reported.
    """
    seq = [np.asarray(f, dtype=np.float64) for f in frames]
    if len(seq) < 2:
        raise ValueError("need at least two frames to measure change")
    deltas = [b - a for a, b in zip(seq, seq[1:])]
    return entropy_series(deltas, estimator=estimator, **kwargs)


@dataclass(frozen=True)
class MetricVerdict:
    """Whether the metric separates a degrading rollout from a healthy one.

    This is the go/no-go for the whole project. If the metric cannot tell the two
    apart on the same prompt, there is no behaviour for WSP to measure and no
    target to train an adapter against.
    """

    healthy: CollapseReading
    degrading: CollapseReading
    #: True when the degrading rollout's ratio is meaningfully lower than the
    #: healthy one's. The threshold is a ratio gap, not an absolute value, so it
    #: does not depend on the estimator's scale.
    separates: bool
    #: ``healthy.ratio - degrading.ratio``. Positive means the healthy rollout
    #: held its entropy better, which is the direction the metric needs.
    gap: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "healthy": self.healthy.as_dict(),
            "degrading": self.degrading.as_dict(),
            "separates": self.separates,
            "gap": self.gap,
            "reason": self.reason,
        }


#: A ratio gap smaller than this is inside the noise of the estimator and should
#: not be read as separation. Chosen conservatively: the project's history is a
#: record of thresholds that looked decisive and were not.
_MIN_SEPARATION_GAP = 0.15


def metric_separates(
    healthy: CollapseReading,
    degrading: CollapseReading,
    min_gap: float = _MIN_SEPARATION_GAP,
) -> MetricVerdict:
    """Decide whether the metric distinguishes a healthy rollout from a degrading one.

    The rule is deliberately strict and ordered, mirroring :mod:`wsp.verify`: a
    failure condition is checked before a success condition, so a helped healthy
    rollout cannot mask a metric that fails to catch degradation.
    """
    # A ratio across zero is not meaningful. When either side's ratio is undefined,
    # fall back to comparing slopes: the degrading rollout must fall faster than the
    # healthy one by a margin proportional to the healthy series' own spread.
    if not (healthy.ratio_defined and degrading.ratio_defined):
        healthy_spread = abs(healthy.slope) + 1e-12
        slope_gap = healthy.slope - degrading.slope
        separates = degrading.collapsing and not healthy.collapsing and slope_gap > healthy_spread
        return MetricVerdict(
            healthy=healthy,
            degrading=degrading,
            separates=separates,
            gap=slope_gap,
            reason=(
                "ratio undefined (non-positive entropy); compared slopes instead: "
                f"healthy {healthy.slope:.4f} vs degrading {degrading.slope:.4f}"
            ),
        )

    gap = healthy.ratio - degrading.ratio

    if not degrading.collapsing:
        return MetricVerdict(
            healthy=healthy,
            degrading=degrading,
            separates=False,
            gap=gap,
            reason=(
                "the degrading rollout did not register as collapsing "
                f"(slope {degrading.slope:.4f}, R² {degrading.r_squared:.3f}); "
                "the metric cannot be validated against a rollout it does not flag"
            ),
        )
    if not healthy.collapsing and gap >= min_gap:
        return MetricVerdict(
            healthy=healthy,
            degrading=degrading,
            separates=True,
            gap=gap,
            reason=(
                f"degrading rollout collapsed while healthy held "
                f"(ratio gap {gap:.3f} >= {min_gap})"
            ),
        )
    if healthy.collapsing:
        return MetricVerdict(
            healthy=healthy,
            degrading=degrading,
            separates=False,
            gap=gap,
            reason=(
                "the healthy rollout also registered as collapsing "
                f"(slope {healthy.slope:.4f}); the metric does not discriminate"
            ),
        )
    return MetricVerdict(
        healthy=healthy,
        degrading=degrading,
        separates=False,
        gap=gap,
        reason=(
            f"both rollouts held entropy; ratio gap {gap:.3f} < {min_gap} "
            "is inside estimator noise"
        ),
    )
