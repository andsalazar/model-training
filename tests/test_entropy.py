"""Tests for the differential-entropy collapse metric.

The metric is the go/no-go for the Ovi project, so these tests target the ways a
metric like this fails in practice: an estimator that reports a number the sample
count cannot support, a trend read from a non-monotone series, and a "separation"
that is really estimator noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from wsp.entropy import (
    CollapseReading,
    EntropyReading,
    collapse_trend,
    entropy_series,
    frame_delta_entropy,
    frame_entropy,
    gaussian_entropy,
    histogram_entropy,
    knn_entropy,
    metric_separates,
)


def _gaussian(n: int, dim: int, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=(n, dim))


def _collapsing_frames(
    n_frames: int = 12, n_patches: int = 200, dim: int = 16, base_scale: float = 0.5
) -> list[np.ndarray]:
    """A rollout whose frames progressively lose information.

    Each frame is a base pattern plus noise whose scale shrinks toward zero, so
    per-frame entropy falls monotonically. This is the synthetic stand-in for a
    rollout drifting into a degenerate state.

    ``base_scale`` controls how much of each frame is the shared base pattern
    versus independent noise. At 1.0 the collapse is real but mild — the ratio
    only falls to ~0.89, which sits inside the separation threshold. At 0.5 it
    falls to ~0.72, which clears it. The default is the clear case so the
    "separates" tests test separation; the mild case has its own test.
    """
    rng = np.random.default_rng(7)
    base = rng.normal(0.0, base_scale, size=(n_patches, dim))
    frames = []
    for i in range(n_frames):
        scale = 1.0 - 0.08 * i
        frames.append(base + rng.normal(0.0, max(scale, 0.02), size=(n_patches, dim)))
    return frames


def _healthy_frames(n_frames: int = 12, n_patches: int = 200, dim: int = 16) -> list[np.ndarray]:
    """A rollout that holds its information roughly flat."""
    rng = np.random.default_rng(11)
    return [rng.normal(0.0, 1.0, size=(n_patches, dim)) for _ in range(n_frames)]


class TestEstimators:
    def test_gaussian_entropy_matches_closed_form(self):
        # For N(0, s^2 I) in d dims, h = d/2 * (1 + ln(2*pi)) + d*ln(s).
        dim, scale = 4, 2.0
        x = _gaussian(20000, dim, scale=scale)
        expected = dim * (0.5 * (1.0 + np.log(2.0 * np.pi)) + np.log(scale))
        assert gaussian_entropy(x) == pytest.approx(expected, rel=0.05)

    def test_gaussian_entropy_rises_with_spread(self):
        narrow = gaussian_entropy(_gaussian(4000, 8, scale=0.5))
        wide = gaussian_entropy(_gaussian(4000, 8, scale=2.0))
        assert wide > narrow

    def test_knn_entropy_rises_with_spread(self):
        narrow = knn_entropy(_gaussian(2000, 8, scale=0.5))
        wide = knn_entropy(_gaussian(2000, 8, scale=2.0))
        assert wide > narrow

    def test_knn_entropy_matches_gaussian_on_gaussian_data(self):
        # The k-NN estimator must be in the right ballpark for Gaussian data, or
        # its absolute scale is wrong. An earlier version dropped the dimension
        # factor on the mean log-distance and returned ~-8 for 32-dim unit-variance
        # data whose true entropy is ~61 nats — a 70-nat error that read as "no
        # information" for data that had plenty.
        dim = 32
        x = _gaussian(4000, dim, scale=1.0)
        knn = knn_entropy(x, k=5)
        gauss = gaussian_entropy(x)
        assert knn == pytest.approx(gauss, rel=0.15), f"knn={knn:.2f} gaussian={gauss:.2f}"

    def test_knn_entropy_scales_with_dimension(self):
        # Entropy of N(0, I_d) grows linearly in d. A dimension-blind estimator
        # would return roughly the same number for d=4 and d=32.
        small = knn_entropy(_gaussian(4000, 4), k=5)
        large = knn_entropy(_gaussian(4000, 32), k=5)
        assert large > small * 3.0, f"d=4 gave {small:.2f}, d=32 gave {large:.2f}"

    def test_histogram_entropy_rises_with_spread(self):
        narrow = histogram_entropy(_gaussian(4000, 4, scale=0.5))
        wide = histogram_entropy(_gaussian(4000, 4, scale=2.0))
        assert wide > narrow

    def test_knn_falls_back_below_min_samples(self):
        # 10 samples cannot support a 5-NN estimate in 16 dims.
        reading = frame_entropy(_gaussian(10, 16), estimator="knn")
        assert reading.fell_back is True
        # The fallback must still produce a finite number, not NaN or -inf.
        assert np.isfinite(reading.entropy)

    def test_knn_does_not_fall_back_at_min_samples(self):
        reading = frame_entropy(_gaussian(32, 4), estimator="knn")
        assert reading.fell_back is False

    def test_unknown_estimator_rejected(self):
        with pytest.raises(ValueError, match="unknown estimator"):
            frame_entropy(_gaussian(100, 4), estimator="magic")

    def test_1d_input_treated_as_single_dimension(self):
        reading = frame_entropy(np.linspace(0.0, 1.0, 100), estimator="knn")
        assert reading.n_dims == 1
        assert reading.n_samples == 100

    def test_3d_input_rejected(self):
        with pytest.raises(ValueError, match="expected 1-D or 2-D"):
            frame_entropy(np.zeros((4, 4, 4)), estimator="knn")

    def test_identical_samples_give_finite_entropy(self):
        # A fully collapsed frame: every patch identical. The estimator must not
        # return -inf, which is what a naive log(0) would produce.
        x = np.ones((100, 8))
        for estimator in ("histogram", "knn", "gaussian"):
            assert np.isfinite(frame_entropy(x, estimator=estimator).entropy)


class TestCollapseTrend:
    def test_collapsing_rollout_has_negative_slope(self):
        trend = entropy_series(_collapsing_frames(), estimator="knn")
        assert trend.slope < 0.0
        assert trend.ratio < 1.0

    def test_collapsing_rollout_is_flagged(self):
        trend = entropy_series(_collapsing_frames(), estimator="knn")
        assert trend.collapsing is True
        assert trend.r_squared >= 0.5

    def test_healthy_rollout_is_not_flagged(self):
        trend = entropy_series(_healthy_frames(), estimator="knn")
        assert trend.collapsing is False

    def test_healthy_rollout_ratio_near_one(self):
        trend = entropy_series(_healthy_frames(), estimator="knn")
        assert trend.ratio == pytest.approx(1.0, abs=0.25)

    def test_noisy_series_is_not_called_a_collapse(self):
        # A series with a negative slope but no monotone structure: the sign is
        # right and the fit is bad. Requiring only the sign would call this a
        # collapse, which is the error this guard exists to prevent.
        rng = np.random.default_rng(3)
        values = rng.normal(0.0, 5.0, size=40) - np.linspace(0.0, 0.05, 40)
        readings = [
            EntropyReading(frame=i, entropy=float(v), estimator="knn", n_samples=100, n_dims=4)
            for i, v in enumerate(values)
        ]
        trend = collapse_trend(readings)
        assert trend.slope < 0.0
        assert trend.r_squared < 0.5
        assert trend.collapsing is False

    def test_two_points_never_collapse(self):
        # Two points always fit a line perfectly, so R² carries no information.
        readings = [
            EntropyReading(frame=i, entropy=float(v), estimator="knn", n_samples=100, n_dims=4)
            for i, v in enumerate([10.0, 1.0])
        ]
        trend = collapse_trend(readings)
        assert trend.collapsing is False
        assert trend.ratio == 1.0

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            collapse_trend([])

    def test_early_and_late_means_use_thirds(self):
        readings = [
            EntropyReading(frame=i, entropy=float(v), estimator="knn", n_samples=100, n_dims=4)
            for i, v in enumerate([9.0, 9.0, 9.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
        ]
        trend = collapse_trend(readings)
        assert trend.early_mean == pytest.approx(9.0)
        assert trend.late_mean == pytest.approx(1.0)
        assert trend.ratio == pytest.approx(1.0 / 9.0)

    def test_negative_entropy_marks_ratio_undefined(self):
        # Differential entropy is a relative measure and can be negative for
        # concentrated distributions. A ratio across zero is not meaningful, so it
        # must be reported as nan rather than silently returning 1.0 — which is
        # what an earlier version did, making every negative-entropy series look
        # perfectly healthy.
        readings = [
            EntropyReading(frame=i, entropy=float(v), estimator="knn", n_samples=100, n_dims=4)
            for i, v in enumerate([-8.0, -8.1, -8.2, -8.3, -8.4, -8.5])
        ]
        trend = collapse_trend(readings)
        assert trend.ratio_defined is False
        assert np.isnan(trend.ratio)
        # The slope is still valid and still shows the fall.
        assert trend.slope < 0.0
        assert trend.collapsing is True

    def test_negative_entropy_falls_back_to_slopes(self):
        # With both ratios undefined, metric_separates must compare slopes rather
        # than crash or return a meaningless ratio gap.
        def series(values):
            return collapse_trend(
                [
                    EntropyReading(
                        frame=i, entropy=float(v), estimator="knn", n_samples=100, n_dims=4
                    )
                    for i, v in enumerate(values)
                ]
            )

        healthy = series([-8.0, -8.0, -8.0, -8.0, -8.0, -8.0])
        degrading = series([-8.0, -8.2, -8.4, -8.6, -8.8, -9.0])
        verdict = metric_separates(healthy, degrading)
        assert verdict.separates is True
        assert "compared slopes" in verdict.reason

    def test_flat_series_has_zero_slope_and_finite_r2(self):
        readings = [
            EntropyReading(frame=i, entropy=4.0, estimator="knn", n_samples=100, n_dims=4)
            for i in range(10)
        ]
        trend = collapse_trend(readings)
        assert trend.slope == pytest.approx(0.0)
        assert trend.r_squared == 1.0
        assert trend.collapsing is False

    def test_reading_serialises(self):
        trend = entropy_series(_collapsing_frames(n_frames=6), estimator="knn")
        payload = trend.as_dict()
        assert set(payload) == {
            "frames",
            "slope",
            "r_squared",
            "early_mean",
            "late_mean",
            "ratio",
            "ratio_defined",
            "collapsing",
        }
        assert len(payload["frames"]) == 6
        assert set(payload["frames"][0]) == {
            "frame",
            "entropy",
            "estimator",
            "n_samples",
            "n_dims",
            "fell_back",
        }


class TestDeltaEntropy:
    def test_delta_entropy_falls_when_frames_stop_changing(self):
        # Frames become progressively more similar to each other, so the change
        # between them shrinks. This is the "every frame is rich but they are all
        # the same frame" failure.
        rng = np.random.default_rng(5)
        base = rng.normal(0.0, 1.0, size=(200, 16))
        frames = []
        for i in range(12):
            jitter = 1.0 - 0.08 * i
            frames.append(base + rng.normal(0.0, max(jitter, 0.02), size=(200, 16)))
        trend = frame_delta_entropy(frames, estimator="knn")
        assert trend.slope < 0.0
        assert trend.collapsing is True

    def test_delta_entropy_holds_when_frames_keep_changing(self):
        trend = frame_delta_entropy(_healthy_frames(), estimator="knn")
        assert trend.collapsing is False

    def test_delta_entropy_needs_two_frames(self):
        with pytest.raises(ValueError, match="at least two frames"):
            frame_delta_entropy([_gaussian(100, 4)])

    def test_delta_entropy_distinguishes_flat_entropy_from_collapse(self):
        # The case the two metrics exist to separate: per-frame entropy stays
        # flat while the frames themselves converge to one pattern.
        rng = np.random.default_rng(9)
        base = rng.normal(0.0, 1.0, size=(200, 16))
        frames = []
        for i in range(12):
            # Constant total spread, but the deviation shrinks and the base grows
            # to compensate — per-frame entropy is flat, frame-to-frame change is not.
            jitter = 1.0 - 0.08 * i
            frames.append(base * (1.0 + 0.02 * i) + rng.normal(0.0, max(jitter, 0.02), (200, 16)))
        per_frame = entropy_series(frames, estimator="knn")
        delta = frame_delta_entropy(frames, estimator="knn")
        # The delta metric must be the more sensitive of the two here.
        assert delta.slope < per_frame.slope


class TestMetricSeparates:
    def test_separates_when_degrading_collapses_and_healthy_holds(self):
        healthy = entropy_series(_healthy_frames(), estimator="knn")
        degrading = entropy_series(_collapsing_frames(), estimator="knn")
        verdict = metric_separates(healthy, degrading)
        assert verdict.separates is True
        assert verdict.gap > 0.0

    def test_does_not_separate_when_degrading_is_not_flagged(self):
        # The failure condition is checked first: a metric that cannot flag the
        # degrading rollout cannot be validated, however good the healthy one looks.
        healthy = entropy_series(_healthy_frames(), estimator="knn")
        degrading = entropy_series(_healthy_frames(), estimator="knn")
        verdict = metric_separates(healthy, degrading)
        assert verdict.separates is False
        assert "did not register as collapsing" in verdict.reason

    def test_does_not_separate_when_healthy_also_collapses(self):
        healthy = entropy_series(_collapsing_frames(), estimator="knn")
        degrading = entropy_series(_collapsing_frames(), estimator="knn")
        verdict = metric_separates(healthy, degrading)
        assert verdict.separates is False
        assert "does not discriminate" in verdict.reason

    def test_small_gap_is_not_separation(self):
        # A real but mild collapse: the degrading rollout IS flagged, but its
        # ratio only falls to ~0.89, so the gap against a healthy rollout is
        # ~0.11 — below the 0.15 threshold. This is the case the threshold
        # exists for: a difference too small to act on.
        healthy = entropy_series(_healthy_frames(), estimator="knn")
        degrading = entropy_series(_collapsing_frames(base_scale=1.0), estimator="knn")
        assert degrading.collapsing is True
        verdict = metric_separates(healthy, degrading, min_gap=0.15)
        assert verdict.separates is False
        assert "inside estimator noise" in verdict.reason

    def test_mild_collapse_clears_a_lower_threshold(self):
        # The same pair separates once the threshold is set below the observed
        # gap. This pins that the threshold, not the metric, is what rejected it.
        healthy = entropy_series(_healthy_frames(), estimator="knn")
        degrading = entropy_series(_collapsing_frames(base_scale=1.0), estimator="knn")
        verdict = metric_separates(healthy, degrading, min_gap=0.05)
        assert verdict.separates is True

    def test_verdict_serialises(self):
        healthy = entropy_series(_healthy_frames(n_frames=6), estimator="knn")
        degrading = entropy_series(_collapsing_frames(n_frames=6), estimator="knn")
        payload = metric_separates(healthy, degrading).as_dict()
        assert set(payload) == {"healthy", "degrading", "separates", "gap", "reason"}

    def test_verdict_is_frozen(self):
        healthy = entropy_series(_healthy_frames(n_frames=6), estimator="knn")
        degrading = entropy_series(_collapsing_frames(n_frames=6), estimator="knn")
        verdict = metric_separates(healthy, degrading)
        with pytest.raises(Exception):
            verdict.separates = True  # type: ignore[misc]


class TestEstimatorAgreement:
    def test_all_estimators_agree_on_direction(self):
        # The verdict should not depend on which estimator was chosen. Direction
        # agreement is the property that matters; absolute values will differ.
        for estimator in ("histogram", "knn", "gaussian"):
            healthy = entropy_series(_healthy_frames(), estimator=estimator)
            degrading = entropy_series(_collapsing_frames(), estimator=estimator)
            assert degrading.slope < healthy.slope, f"{estimator} disagreed on direction"

    def test_estimators_are_not_identical(self):
        # If they returned the same number the comparison above would be vacuous.
        x = _gaussian(2000, 8)
        values = {
            histogram_entropy(x),
            knn_entropy(x),
            gaussian_entropy(x),
        }
        assert len(values) == 3
