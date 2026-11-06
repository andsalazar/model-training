"""Tests for the Ovi metric script's frame handling.

The script itself needs a GPU and the model, so it cannot be run in CI. But the
parts that turn pixels into patches and construct a degrading rollout are pure
array math, and those are where a silent error would corrupt every downstream
number. They are tested here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pod_ovi_metric.py"


def _load_script():
    """Import the script as a module without running ``main``."""
    spec = importlib.util.spec_from_file_location("pod_ovi_metric", _SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pod_ovi_metric"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


class TestPatchesFromPixels:
    def test_patch_count_matches_grid(self, script):
        # 32x32 pixels at patch 16 is a 2x2 grid = 4 patches.
        frames = np.zeros((3, 32, 32, 3), dtype=np.uint8)
        patches = script._patches_from_pixels(frames, patch=16, max_patches=100)
        assert patches.shape == (3, 4, 16 * 16 * 3)

    def test_patch_dims_are_patch_area_times_channels(self, script):
        frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)
        patches = script._patches_from_pixels(frames, patch=8, max_patches=100)
        # 8x8 patch, 3 channels.
        assert patches.shape[2] == 8 * 8 * 3

    def test_uneven_dimensions_are_trimmed_not_padded(self, script):
        # 30x30 at patch 16 leaves a 14-pixel remainder on each axis. Trimming is
        # correct: padding would inject constant values that lower entropy and
        # would do so differently per frame.
        frames = np.zeros((2, 30, 30, 3), dtype=np.uint8)
        patches = script._patches_from_pixels(frames, patch=16, max_patches=100)
        assert patches.shape == (2, 1, 16 * 16 * 3)

    def test_subsampling_caps_patches(self, script):
        frames = np.zeros((2, 64, 64, 3), dtype=np.uint8)
        patches = script._patches_from_pixels(frames, patch=8, max_patches=10)
        assert patches.shape[1] == 10

    def test_subsampling_is_deterministic(self, script):
        # A random subsample would make every re-run give a different number.
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 255, size=(2, 64, 64, 3), dtype=np.uint8)
        first = script._patches_from_pixels(frames, patch=8, max_patches=10)
        second = script._patches_from_pixels(frames, patch=8, max_patches=10)
        assert np.array_equal(first, second)

    def test_patch_content_is_preserved(self, script):
        # A single patch covering the whole frame must equal the frame flattened.
        frame = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        patches = script._patches_from_pixels(frame[None, ...], patch=16, max_patches=10)
        assert np.array_equal(patches[0, 0], frame.reshape(-1).astype(np.float64))


class TestConvergeTowardFirst:
    def test_strength_zero_leaves_frames_untouched(self, script):
        rng = np.random.default_rng(1)
        frames = rng.normal(size=(6, 20, 8))
        out = script._converge_toward_first(frames, 0.0)
        assert np.array_equal(out, frames)

    def test_strength_one_makes_the_last_frame_the_first(self, script):
        # The blend is progressive: strength is the blend applied to the *last*
        # frame, ramping from 0 at frame 0. So strength 1 makes only the final
        # frame identical to the first — not every frame. An earlier version of
        # this test asserted every frame, which the progressive ramp correctly
        # fails.
        rng = np.random.default_rng(2)
        frames = rng.normal(size=(6, 20, 8))
        out = script._converge_toward_first(frames, 1.0)
        assert np.array_equal(out[-1], frames[0])
        # Intermediate frames are partially blended, not replaced.
        assert not np.array_equal(out[1], frames[0])
        assert not np.array_equal(out[1], frames[1])

    def test_convergence_is_progressive_not_uniform(self, script):
        # Frame i is blended by strength * i/(n-1), so later frames move further
        # toward frame 0. A uniform blend would lower entropy without producing a
        # trend, and a trend is the signal being tested.
        rng = np.random.default_rng(3)
        frames = rng.normal(size=(5, 20, 8))
        out = script._converge_toward_first(frames, 0.8)
        # Distance to frame 0 must fall monotonically as the blend deepens.
        distances = [float(np.linalg.norm(out[i] - frames[0])) for i in range(len(out))]
        assert distances[0] == 0.0
        for earlier, later in zip(distances[1:], distances[2:]):
            assert later <= earlier + 1e-9

    def test_first_frame_is_never_modified(self, script):
        rng = np.random.default_rng(4)
        frames = rng.normal(size=(6, 20, 8))
        out = script._converge_toward_first(frames, 0.9)
        assert np.array_equal(out[0], frames[0])

    def test_out_of_range_strength_rejected(self, script):
        frames = np.zeros((3, 4, 2))
        for bad in (-0.1, 1.1):
            with pytest.raises(ValueError, match="strength must be in"):
                script._converge_toward_first(frames, bad)

    def test_single_frame_is_returned_unchanged(self, script):
        frames = np.ones((1, 4, 2))
        out = script._converge_toward_first(frames, 0.9)
        assert np.array_equal(out, frames)


class TestLoadFrames:
    def test_2d_array_becomes_one_frame(self, script, tmp_path):
        path = tmp_path / "frames.npy"
        np.save(path, np.zeros((20, 8)))
        loaded = script._load_frames(path)
        assert loaded.shape == (1, 20, 8)

    def test_3d_array_passes_through(self, script, tmp_path):
        path = tmp_path / "frames.npy"
        np.save(path, np.zeros((6, 20, 8)))
        loaded = script._load_frames(path)
        assert loaded.shape == (6, 20, 8)

    def test_4d_array_rejected(self, script, tmp_path):
        path = tmp_path / "frames.npy"
        np.save(path, np.zeros((2, 2, 2, 2)))
        with pytest.raises(ValueError, match="expected 2-D or 3-D"):
            script._load_frames(path)


class TestSyntheticEndToEnd:
    def test_delta_metric_separates_a_constructed_collapse(self, script, tmp_path):
        # The finding this script exists to establish: per-frame entropy does not
        # separate a constructed collapse, but delta entropy does. If this ever
        # flips, the plan's metric choice needs revisiting.
        from wsp.entropy import entropy_series, frame_delta_entropy, metric_separates

        rng = np.random.default_rng(42)
        n_frames, n_patches, dim = 12, 300, 32
        base = rng.normal(0.0, 1.0, size=(n_patches, dim))
        frames = np.stack(
            [base + rng.normal(0.0, 1.0, size=(n_patches, dim)) for _ in range(n_frames)]
        )

        flat = [frames[i] for i in range(n_frames)]
        healthy = entropy_series(flat, estimator="knn")
        healthy_delta = frame_delta_entropy(flat, estimator="knn")

        degraded = script._converge_toward_first(frames, 0.75)
        flat_d = [degraded[i] for i in range(n_frames)]
        per_frame = entropy_series(flat_d, estimator="knn")
        delta = frame_delta_entropy(flat_d, estimator="knn")

        # Per-frame entropy does not separate: the frames stay individually rich.
        assert metric_separates(healthy, per_frame).separates is False
        # Delta entropy does: the change between frames has gone.
        assert metric_separates(healthy_delta, delta).separates is True

    def test_untouched_rollout_is_not_flagged(self, script):
        from wsp.entropy import entropy_series

        rng = np.random.default_rng(43)
        frames = np.stack([rng.normal(size=(200, 16)) for _ in range(12)])
        flat = [frames[i] for i in range(12)]
        assert entropy_series(flat, estimator="knn").collapsing is False
