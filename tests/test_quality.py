"""Tests for the reference-free video quality metrics.

These metrics decide whether a speed change is acceptable, so the tests target the
ways a metric like this fails quietly: returning a plausible number for a degenerate
input, or reporting "acceptable" when the candidate is visibly worse.
"""

from __future__ import annotations

import numpy as np
import pytest

from wsp.quality import (
    DEFAULT_TOLERANCE,
    QualityReading,
    colour_stability,
    compare_quality,
    measure_quality,
    motion_coherence,
    sharpness,
    temporal_flicker,
)


def _smooth_frames(n_frames: int = 12, size: int = 64, shift: int = 2) -> np.ndarray:
    """A rollout with a consistent horizontal translation — coherent motion."""
    base = np.zeros((size, size), dtype=np.float64)
    base[:, size // 4 : size // 2] = 200.0
    base[size // 3 : 2 * size // 3, size // 2 : 3 * size // 4] = 120.0
    frames = []
    for i in range(n_frames):
        frames.append(np.roll(base, shift * i, axis=1))
    return np.stack(frames)[..., None].repeat(3, axis=-1)


def _noise_frames(n_frames: int = 12, size: int = 64, seed: int = 0) -> np.ndarray:
    """A rollout of independent noise — no coherent motion at all."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 255.0, size=(n_frames, size, size, 3))


def _blurred(frames: np.ndarray, k: int = 5) -> np.ndarray:
    """Box-blur each frame, to simulate a rollout that lost high-frequency detail."""
    arr = np.asarray(frames, dtype=np.float64)
    pad = k // 2
    out = np.empty_like(arr)
    for f in range(arr.shape[0]):
        for c in range(arr.shape[-1]):
            padded = np.pad(arr[f, :, :, c], pad, mode="edge")
            acc = np.zeros_like(arr[f, :, :, c])
            for dy in range(k):
                for dx in range(k):
                    acc += padded[dy : dy + arr.shape[1], dx : dx + arr.shape[2]]
            out[f, :, :, c] = acc / (k * k)
    return out


class TestSharpness:
    def test_blurring_lowers_sharpness(self):
        sharp = _smooth_frames()
        blurred = _blurred(sharp)
        assert sharpness(blurred) < sharpness(sharp) * 0.7

    def test_flat_frames_have_zero_sharpness(self):
        flat = np.full((4, 32, 32, 3), 128.0)
        assert sharpness(flat) == pytest.approx(0.0, abs=1e-9)

    def test_noise_is_sharper_than_a_smooth_gradient(self):
        rng = np.random.default_rng(3)
        smooth = np.tile(
            np.linspace(0.0, 255.0, 64)[None, :, None], (4, 1, 64)
        )[..., None].repeat(3, axis=-1)
        noisy = smooth + rng.normal(0.0, 20.0, size=smooth.shape)
        assert sharpness(noisy) > sharpness(smooth)

    def test_border_does_not_bias_the_estimate(self):
        # A first-order gradient halves its estimate at the border, so a blur that
        # pushes energy outward could raise the mean. The Laplacian is interior-only,
        # so a uniform field stays exactly zero however it is shifted.
        flat = np.full((4, 32, 32, 3), 77.0)
        assert sharpness(flat) == pytest.approx(0.0, abs=1e-9)
        shifted = np.roll(flat, 3, axis=2)
        assert sharpness(shifted) == pytest.approx(0.0, abs=1e-9)


class TestTemporalFlicker:
    def test_static_rollout_has_zero_flicker(self):
        frame = np.full((32, 32, 3), 100.0)
        static = np.stack([frame] * 6)
        assert temporal_flicker(static) == pytest.approx(0.0, abs=1e-9)

    def test_noise_has_higher_flicker_than_coherent_motion(self):
        assert temporal_flicker(_noise_frames()) > temporal_flicker(_smooth_frames())

    def test_normalisation_makes_bright_and_dark_comparable(self):
        # The same relative change at two exposures should give similar flicker.
        dark = _smooth_frames() * 0.25
        bright = _smooth_frames()
        assert temporal_flicker(dark) == pytest.approx(
            temporal_flicker(bright), rel=0.05
        )


class TestMotionCoherence:
    def test_coherent_motion_is_structured(self):
        # A clean horizontal pan produces a smooth difference field. The absolute
        # value depends on how much of the dipole a 5x5 blur captures, so the
        # meaningful assertion is the ratio against noise, tested below. This only
        # checks that coherent motion is clearly structured rather than noise-like.
        assert motion_coherence(_smooth_frames()) > 0.25

    def test_noise_is_not_structured(self):
        value = motion_coherence(_noise_frames())
        assert value < 0.2, f"noise gave coherence {value:.3f}"

    def test_coherent_motion_scores_far_above_noise(self):
        # The ratio is the usable signal, not the absolute value.
        coherent = motion_coherence(_smooth_frames())
        noisy = motion_coherence(_noise_frames())
        assert coherent > noisy * 4.0, f"coherent {coherent:.3f} vs noise {noisy:.3f}"

    def test_static_rollout_returns_nan(self):
        # No change at all means the energy ratio is undefined, not zero.
        assert np.isnan(motion_coherence(np.zeros((4, 16, 16, 3))))

    def test_bounded_in_unit_interval(self):
        for frames in (_smooth_frames(), _noise_frames()):
            value = motion_coherence(frames)
            assert 0.0 <= value <= 1.0, f"coherence {value:.3f} out of range"


class TestColourStability:
    def test_constant_colour_is_stable(self):
        frame = np.full((32, 32, 3), 90.0)
        assert colour_stability(np.stack([frame] * 6)) == pytest.approx(0.0, abs=1e-9)

    def test_drifting_colour_is_unstable(self):
        frames = np.stack([np.full((32, 32, 3), 40.0 + 20.0 * i) for i in range(6)])
        assert colour_stability(frames) > 10.0


class TestMeasureQuality:
    def test_returns_all_four_metrics(self):
        reading = measure_quality(_smooth_frames())
        assert set(reading.metrics) == {
            "sharpness",
            "temporal_flicker",
            "motion_coherence",
            "colour_stability",
        }
        assert reading.n_frames == 12

    def test_accepts_grayscale_and_rgb(self):
        gray = _smooth_frames()[..., 0]
        rgb = _smooth_frames()
        assert measure_quality(gray).n_frames == measure_quality(rgb).n_frames

    def test_rejects_single_frame(self):
        with pytest.raises(ValueError, match="at least 2 frames"):
            measure_quality(np.zeros((1, 16, 16, 3)))

    def test_rejects_wrong_dimensionality(self):
        with pytest.raises(ValueError, match="expected 3 or 4 dims"):
            measure_quality(np.zeros((4, 4)))

    def test_as_dict_round_trips(self):
        reading = measure_quality(_smooth_frames())
        d = reading.as_dict()
        assert d["n_frames"] == 12
        assert d["sharpness"] == reading.sharpness


class TestCompareQuality:
    def test_identical_rollouts_are_acceptable(self):
        reading = measure_quality(_smooth_frames())
        cmp = compare_quality(reading, reading)
        assert cmp.acceptable is True
        assert cmp.outside == []
        assert all(r == pytest.approx(1.0) for r in cmp.ratios.values())

    def test_blurred_candidate_is_rejected(self):
        base = measure_quality(_smooth_frames())
        cand = measure_quality(_blurred(_smooth_frames()))
        cmp = compare_quality(base, cand)
        assert cmp.acceptable is False
        assert "sharpness" in cmp.outside

    def test_noisy_candidate_is_rejected(self):
        base = measure_quality(_smooth_frames())
        cand = measure_quality(_noise_frames())
        cmp = compare_quality(base, cand)
        assert cmp.acceptable is False
        assert "motion_coherence" in cmp.outside

    def test_tolerance_widens_the_band(self):
        # A mild blur (k=3) is a small sharpness change, so a wide tolerance should
        # accept it while a tight one rejects it. k=5 is too destructive to accept at
        # any reasonable tolerance, so it is not used here.
        base = measure_quality(_smooth_frames())
        cand = measure_quality(_blurred(_smooth_frames(), k=3))
        strict = compare_quality(base, cand, tolerance=0.02)
        loose = compare_quality(base, cand, tolerance=0.90)
        assert strict.acceptable is False
        assert loose.acceptable is True

    def test_zero_baseline_with_nonzero_candidate_is_outside(self):
        # A flat baseline has sharpness 0.0. A candidate that has detail is a real
        # change from nothing, so it must be reported as outside — not as "unchanged".
        flat = np.full((4, 32, 32, 3), 128.0)
        base = measure_quality(flat)
        cand = measure_quality(_smooth_frames())
        cmp = compare_quality(base, cand)
        assert "sharpness" in cmp.outside
        assert cmp.ratios["sharpness"] == float("inf")
        assert cmp.acceptable is False

    def test_zero_baseline_with_zero_candidate_is_unchanged(self):
        # A static rollout has colour_stability 0.0, which is a good result. Two
        # static rollouts must compare as unchanged, not as undefined.
        flat = np.full((4, 32, 32, 3), 128.0)
        base = measure_quality(flat)
        cand = measure_quality(flat)
        cmp = compare_quality(base, cand)
        assert cmp.ratios["colour_stability"] == 1.0
        assert "colour_stability" not in cmp.outside

    def test_negative_tolerance_rejected(self):
        reading = measure_quality(_smooth_frames())
        with pytest.raises(ValueError, match="non-negative"):
            compare_quality(reading, reading, tolerance=-0.1)

    def test_reason_names_the_offending_metrics(self):
        base = measure_quality(_smooth_frames())
        cand = measure_quality(_blurred(_smooth_frames()))
        cmp = compare_quality(base, cand)
        assert "sharpness" in cmp.reason
        assert "outside" in cmp.reason

    def test_default_tolerance_is_ten_percent(self):
        assert DEFAULT_TOLERANCE == 0.10

    def test_as_dict_includes_ratios_and_verdict(self):
        base = measure_quality(_smooth_frames())
        cand = measure_quality(_blurred(_smooth_frames()))
        d = compare_quality(base, cand).as_dict()
        assert d["acceptable"] is False
        assert "sharpness" in d["ratios"]
        assert d["tolerance"] == DEFAULT_TOLERANCE


class TestQualityReadingConstruction:
    def test_metrics_dict_is_built_when_omitted(self):
        reading = QualityReading(
            sharpness=1.0,
            temporal_flicker=2.0,
            motion_coherence=0.5,
            colour_stability=0.1,
            n_frames=8,
        )
        assert reading.metrics["sharpness"] == 1.0
        assert reading.metrics["motion_coherence"] == 0.5

    def test_explicit_metrics_dict_is_preserved(self):
        reading = QualityReading(
            sharpness=1.0,
            temporal_flicker=2.0,
            motion_coherence=0.5,
            colour_stability=0.1,
            n_frames=8,
            metrics={"sharpness": 9.0},
        )
        assert reading.metrics == {"sharpness": 9.0}
