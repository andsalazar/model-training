"""Quality metrics for generated video, so "without losing quality" is measurable.

Why this module exists
----------------------
The speed goal is only meaningful against a quality constraint. "Faster but worse"
is easy; "faster and as good" needs a number that can fail. This module provides
reference-free metrics — they need no ground-truth video, only the generated one —
because there is no ground truth for a text-to-video prompt.

What is measured, and why each one
----------------------------------
``sharpness``
    Mean gradient magnitude across frames. A blurry or over-smoothed rollout loses
    high-frequency detail. This is the metric that catches "the model got faster by
    producing mush".

``temporal_flicker``
    Mean absolute difference between *consecutive* frames, normalised by frame
    brightness. High values mean the picture jitters frame to frame; low values mean
    it is static. Both extremes are bad, so this is reported as a number and judged
    against a baseline rather than against a threshold.

``motion_coherence``
    Fraction of frame-to-frame change energy that is spatially structured. Real
    motion produces a smooth difference field; noise produces a rough one. A drop
    here means the model is producing incoherent change rather than motion.

``colour_stability``
    Mean and spread of per-frame mean colour. A rollout that drifts in hue or
    brightness is degrading even if each frame looks fine alone.

What is deliberately NOT measured
---------------------------------
No FVD, no CLIP score, no human preference model. Each needs either a reference
distribution or a large model, and both would make the metric expensive enough that
it would not be run on every generation. These four are cheap, deterministic, and
each one fails in a way that maps to a visible defect.

The comparison rule
-------------------
Absolute quality numbers are not comparable across prompts — a foggy scene is
legitimately less sharp than a detailed one. So the unit of judgement is the
*ratio* against a baseline generation of the same prompt. A speed change is
acceptable when quality ratios stay within a tolerance band.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: A quality ratio within this band of 1.0 is treated as unchanged. Chosen because
#: the metrics below vary by a few percent between seeds of the same prompt, so a
#: tighter band would report seed noise as a quality change.
DEFAULT_TOLERANCE = 0.10


def _as_frames(frames: np.ndarray) -> np.ndarray:
    """Coerce to ``(n_frames, h, w, c)`` float64 in 0..255."""
    arr = np.asarray(frames)
    if arr.ndim == 3:
        arr = arr[..., None]
    if arr.ndim != 4:
        raise ValueError(f"expected 3 or 4 dims (n,h,w[,c]), got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"need at least 2 frames, got {arr.shape[0]}")
    return arr.astype(np.float64)


def _luma(frames: np.ndarray) -> np.ndarray:
    """Rec.601 luma, ``(n, h, w)``. Falls back to the mean channel."""
    if frames.shape[-1] >= 3:
        return (
            0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]
        )
    return frames[..., 0]


def sharpness(frames: np.ndarray) -> float:
    """Mean 4-neighbour Laplacian magnitude over all frames.

    Higher means more high-frequency detail. A Laplacian is used rather than a
    first-order gradient because ``np.gradient`` halves its estimate at the image
    border (one-sided differences), so a blur that pushes energy toward the edges
    can *raise* the mean gradient. The Laplacian is computed on the interior only,
    where all four neighbours exist, so the border cannot bias the number.
    """
    y = _luma(_as_frames(frames))
    if y.shape[1] < 3 or y.shape[2] < 3:
        return 0.0
    centre = y[:, 1:-1, 1:-1]
    lap = (
        y[:, :-2, 1:-1]
        + y[:, 2:, 1:-1]
        + y[:, 1:-1, :-2]
        + y[:, 1:-1, 2:]
        - 4.0 * centre
    )
    return float(np.mean(np.abs(lap)))


def temporal_flicker(frames: np.ndarray) -> float:
    """Mean absolute consecutive-frame difference, normalised by mean luma.

    Normalising matters: a bright scene has larger absolute differences for the
    same amount of change. Without it, comparing a dark and a bright prompt would
    report a difference that is purely exposure.
    """
    y = _luma(_as_frames(frames))
    diffs = np.abs(np.diff(y, axis=0))
    denom = max(float(np.mean(y)), 1e-6)
    return float(np.mean(diffs) / denom)


def _box_blur_2d(field: np.ndarray, k: int = 5) -> np.ndarray:
    """Box-blur a single 2-D field with edge padding."""
    pad = k // 2
    padded = np.pad(field, pad, mode="edge")
    acc = np.zeros_like(field)
    h, w = field.shape
    for dy in range(k):
        for dx in range(k):
            acc += padded[dy : dy + h, dx : dx + w]
    return acc / (k * k)


def motion_coherence(frames: np.ndarray, k: int = 5) -> float:
    """Fraction of frame-to-frame change energy that is spatially structured.

    The first definition tried here correlated consecutive difference fields
    elementwise. That is wrong for translation: when an object moves, its difference
    dipole moves *with* it, so the same pixel position sees a different value and the
    correlation collapses to zero even for perfectly coherent motion. The fixture
    caught this — a clean horizontal pan scored 0.0.

    What actually separates motion from noise is spatial structure. Coherent motion
    produces a smooth difference field (edges move together); noise produces a rough
    one. So this measures the energy fraction of the difference field that survives a
    small box blur:

        coherence = mean over frames of  ||blur(d)||^2 / ||d||^2

    Bounded in [0, 1]. Near 1 means the change is spatially coherent; near 0 means it
    is pixel-level noise. Returns ``nan`` when there is no change at all, because the
    ratio is then undefined rather than zero.
    """
    y = _luma(_as_frames(frames))
    d = np.diff(y, axis=0)
    if len(d) < 1:
        return float("nan")
    fractions = []
    for frame_diff in d:
        energy = float((frame_diff * frame_diff).sum())
        if energy < 1e-12:
            continue
        smoothed = _box_blur_2d(frame_diff, k)
        fractions.append(float((smoothed * smoothed).sum()) / energy)
    if not fractions:
        return float("nan")
    return float(np.mean(fractions))


def colour_stability(frames: np.ndarray) -> float:
    """Spread of per-frame mean colour across the rollout.

    Low means the colour balance holds. A rising value means the rollout is drifting
    in hue or exposure.
    """
    arr = _as_frames(frames)
    per_frame = arr.mean(axis=(1, 2))
    return float(np.mean(np.std(per_frame, axis=0)))


@dataclass
class QualityReading:
    """The four reference-free quality metrics for one rollout."""

    sharpness: float
    temporal_flicker: float
    motion_coherence: float
    colour_stability: float
    n_frames: int
    #: Per-metric values keyed by name, for ratio comparison.
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.metrics:
            self.metrics = {
                "sharpness": self.sharpness,
                "temporal_flicker": self.temporal_flicker,
                "motion_coherence": self.motion_coherence,
                "colour_stability": self.colour_stability,
            }

    def as_dict(self) -> dict:
        return {
            "sharpness": self.sharpness,
            "temporal_flicker": self.temporal_flicker,
            "motion_coherence": self.motion_coherence,
            "colour_stability": self.colour_stability,
            "n_frames": self.n_frames,
        }


def measure_quality(frames: np.ndarray) -> QualityReading:
    """Compute all four metrics for one rollout."""
    arr = _as_frames(frames)
    return QualityReading(
        sharpness=sharpness(arr),
        temporal_flicker=temporal_flicker(arr),
        motion_coherence=motion_coherence(arr),
        colour_stability=colour_stability(arr),
        n_frames=int(arr.shape[0]),
    )


@dataclass
class QualityComparison:
    """A candidate rollout's quality against a baseline of the same prompt."""

    baseline: QualityReading
    candidate: QualityReading
    #: ``candidate / baseline`` per metric. 1.0 means unchanged.
    ratios: dict[str, float]
    #: Metrics whose ratio fell outside the tolerance band.
    outside: list[str]
    tolerance: float
    #: True when every metric stayed inside the band.
    acceptable: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "baseline": self.baseline.as_dict(),
            "candidate": self.candidate.as_dict(),
            "ratios": self.ratios,
            "outside": self.outside,
            "tolerance": self.tolerance,
            "acceptable": self.acceptable,
            "reason": self.reason,
        }


def _safe_ratio(candidate: float, baseline: float) -> float:
    """Ratio that handles a zero baseline without inventing a number.

    A zero baseline is not always an error. ``colour_stability`` is exactly 0.0 for a
    perfectly static rollout, which is a *good* result, not a missing measurement. So
    when the baseline is zero this returns 1.0 if the candidate is also zero (no
    change) and ``inf`` if the candidate is not (a real change from nothing). ``inf``
    is reported as outside the band, which is the correct verdict.

    ``nan`` is reserved for a genuinely undefined case: a non-finite input.
    """
    if not np.isfinite(baseline) or not np.isfinite(candidate):
        return float("nan")
    if abs(baseline) < 1e-12:
        return 1.0 if abs(candidate) < 1e-12 else float("inf")
    return float(candidate / baseline)


def compare_quality(
    baseline: QualityReading,
    candidate: QualityReading,
    tolerance: float = DEFAULT_TOLERANCE,
) -> QualityComparison:
    """Compare a candidate rollout against a baseline of the same prompt.

    A metric is "outside" when its ratio is more than ``tolerance`` away from 1.0 in
    either direction. Direction matters for interpretation but not for the verdict:
    a rollout that got *sharper* while getting faster is still a change, and calling
    it acceptable without looking would hide a real difference.
    """
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")

    ratios: dict[str, float] = {}
    outside: list[str] = []
    for name, base_value in baseline.metrics.items():
        cand_value = candidate.metrics.get(name)
        if cand_value is None:
            continue
        ratio = _safe_ratio(cand_value, base_value)
        ratios[name] = ratio
        if np.isnan(ratio):
            outside.append(name)
        elif abs(ratio - 1.0) > tolerance:
            # ``inf`` lands here too, which is correct: a change from a zero baseline
            # is a real change and must not pass as "unchanged".
            outside.append(name)

    acceptable = not outside
    if acceptable:
        reason = (
            f"all {len(ratios)} metrics within {tolerance:.0%} of baseline"
        )
    else:
        detail = ", ".join(
            f"{n}={ratios[n]:.3f}" if np.isfinite(ratios.get(n, float("nan"))) else f"{n}=nan"
            for n in outside
        )
        # ``inf`` formats as "inf" through the finite branch's else, which is fine.
        reason = f"{len(outside)} of {len(ratios)} metrics outside {tolerance:.0%}: {detail}"

    return QualityComparison(
        baseline=baseline,
        candidate=candidate,
        ratios=ratios,
        outside=outside,
        tolerance=tolerance,
        acceptable=acceptable,
        reason=reason,
    )
