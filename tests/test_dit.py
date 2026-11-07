"""Tests for DiT activation capture.

The capture logic is pure array math, so it can be tested without a GPU or the 23GB
Ovi checkpoint. That matters because every failure mode here is silent: a schedule
that matches nothing looks like a capture that found no structure, and a frame window
that indexes the wrong patches produces a direction that is arithmetic noise.

The project's recurring error is a tool that returns a plausible number for the wrong
reason, so each test below pins a case where that would happen.
"""

from __future__ import annotations

import numpy as np
import pytest

from wsp.dit import (
    STREAMS,
    CaptureSchedule,
    LayerCapture,
    ResidualCapture,
    _as_patch_matrix,
    _windowed_mean,
    frame_window,
    subsample_patches,
)


def _activation(n_patches: int = 12, hidden: int = 8, seed: int = 0) -> np.ndarray:
    """A ``[1, n_patches, hidden]`` hooked tensor, the shape a DiT block emits."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(1, n_patches, hidden)).astype(np.float32)


class TestCaptureSchedule:
    def test_wants_a_listed_layer_and_any_timestep(self):
        schedule = CaptureSchedule(layers=(0, 3), timesteps=None)
        assert schedule.wants(0, 0) is True
        assert schedule.wants(3, 49) is True

    def test_rejects_an_unlisted_layer(self):
        assert CaptureSchedule(layers=(0, 3)).wants(7, 0) is False

    def test_restricts_timesteps_when_given(self):
        schedule = CaptureSchedule(layers=(0,), timesteps=(0, 25, 49))
        assert schedule.wants(0, 25) is True
        assert schedule.wants(0, 10) is False

    def test_default_is_early_layers(self):
        # TBC's adapter sits in early DiT layers, so the default should be too.
        schedule = CaptureSchedule()
        assert all(layer < 13 for layer in schedule.layers)
        assert schedule.timesteps is None
        assert schedule.keep_patches is False

    def test_describe_names_the_restrictions(self):
        text = CaptureSchedule(layers=(0, 2), timesteps=(5,), keep_patches=True).describe()
        assert "layers=0,2" in text
        assert "timesteps=5" in text
        assert "keep" in text


class TestAsPatchMatrix:
    def test_squeezes_a_single_batch(self):
        assert _as_patch_matrix(_activation(12, 8)).shape == (12, 8)

    def test_accepts_already_2d(self):
        assert _as_patch_matrix(np.zeros((12, 8), dtype=np.float32)).shape == (12, 8)

    def test_rejects_multi_batch(self):
        # Averaging across generations would mix two different rollouts into one
        # direction, which is meaningless. Reject rather than average silently.
        with pytest.raises(ValueError, match="batch size 1"):
            _as_patch_matrix(np.zeros((2, 12, 8), dtype=np.float32))

    def test_rejects_wrong_rank(self):
        with pytest.raises(ValueError, match="expected"):
            _as_patch_matrix(np.zeros((12,), dtype=np.float32))


class TestFrameWindow:
    def test_none_returns_everything(self):
        matrix = np.arange(12 * 2, dtype=np.float32).reshape(12, 2)
        assert np.array_equal(frame_window(matrix, None, 4), matrix)

    def test_restricts_to_a_frame_range(self):
        # 12 patches over 4 frames = 3 patches per frame.
        matrix = np.arange(12 * 2, dtype=np.float32).reshape(12, 2)
        window = frame_window(matrix, "1:3", 4)
        assert window.shape == (6, 2)
        # Frames 1 and 2 start at patch 3.
        assert window[0, 0] == matrix[3, 0]

    def test_single_frame_window(self):
        matrix = np.arange(12 * 1, dtype=np.float32).reshape(12, 1)
        assert frame_window(matrix, "2:3", 4).shape == (3, 1)

    def test_open_start_and_stop(self):
        matrix = np.arange(12 * 1, dtype=np.float32).reshape(12, 1)
        assert frame_window(matrix, ":2", 4).shape == (6, 1)
        assert frame_window(matrix, "2:", 4).shape == (6, 1)

    def test_uneven_patch_count_is_rejected(self):
        # 10 patches over 4 frames means the layout is wrong; guessing would index
        # the wrong spatial region.
        with pytest.raises(ValueError, match="do not divide evenly"):
            frame_window(np.zeros((10, 2), dtype=np.float32), "0:2", 4)

    def test_window_outside_range_is_rejected(self):
        with pytest.raises(ValueError, match="outside"):
            frame_window(np.zeros((12, 2), dtype=np.float32), "0:9", 4)

    def test_empty_and_inverted_windows_are_rejected(self):
        matrix = np.zeros((12, 2), dtype=np.float32)
        for bad in ("2:2", "3:1"):
            with pytest.raises(ValueError, match="outside"):
                frame_window(matrix, bad, 4)

    def test_zero_frames_is_rejected(self):
        with pytest.raises(ValueError, match="n_frames must be positive"):
            frame_window(np.zeros((12, 2), dtype=np.float32), "0:1", 0)


class TestResidualCapture:
    def test_records_a_wanted_capture(self):
        cap = ResidualCapture(schedule=CaptureSchedule(layers=(0,)))
        assert cap.record(0, 5, _activation()) is True
        assert cap.at(0, 5).n_patches == 12
        assert cap.at(0, 5).hidden == 8

    def test_skips_an_unwanted_capture_and_counts_it(self):
        # A schedule that matches nothing must be visible, not look like an empty
        # capture that found no structure.
        cap = ResidualCapture(schedule=CaptureSchedule(layers=(0,)))
        assert cap.record(7, 5, _activation()) is False
        assert cap.skipped == 1
        assert cap.captures == {}

    def test_tracks_the_axes_it_saw(self):
        cap = ResidualCapture(schedule=CaptureSchedule(layers=(0, 3), timesteps=(0, 9)))
        cap.record(0, 0, _activation())
        cap.record(3, 9, _activation())
        assert cap.layers() == [0, 3]
        assert cap.timesteps() == [0, 9]

    def test_does_not_retain_patches_by_default(self):
        cap = ResidualCapture(schedule=CaptureSchedule(layers=(0,)))
        cap.record(0, 0, _activation())
        assert cap.at(0, 0).patches is None

    def test_retains_patches_when_asked(self):
        cap = ResidualCapture(
            schedule=CaptureSchedule(layers=(0,), keep_patches=True)
        )
        cap.record(0, 0, _activation())
        assert cap.at(0, 0).patches is not None
        assert cap.at(0, 0).patches.shape == (12, 8)

    def test_mean_requires_patches(self):
        cap = ResidualCapture(schedule=CaptureSchedule(layers=(0,)))
        cap.record(0, 0, _activation())
        with pytest.raises(ValueError, match="keep_patches"):
            _ = cap.at(0, 0).mean

    def test_missing_key_names_what_is_available(self):
        cap = ResidualCapture(schedule=CaptureSchedule(layers=(0,)))
        cap.record(0, 0, _activation())
        with pytest.raises(KeyError, match="available"):
            cap.at(5, 5)

    def test_summary_reports_skips(self):
        cap = ResidualCapture(schedule=CaptureSchedule(layers=(0,)))
        cap.record(0, 0, _activation())
        cap.record(9, 0, _activation())
        summary = cap.summary()
        assert summary["n_captures"] == 1
        assert summary["n_skipped"] == 1


class TestDirection:
    def _pair(self, shift: float = 1.0):
        schedule = CaptureSchedule(layers=(0,), keep_patches=True)
        base = _activation(12, 8, seed=1)
        other = ResidualCapture(schedule=schedule)
        other.record(0, 0, base)
        mine = ResidualCapture(schedule=schedule)
        mine.record(0, 0, base + shift)
        return mine, other

    def test_direction_is_unit_length(self):
        mine, other = self._pair()
        vec = mine.direction(0, 0, other)
        assert np.linalg.norm(vec) == pytest.approx(1.0, rel=1e-5)

    def test_direction_points_from_other_to_mine(self):
        mine, other = self._pair(shift=2.0)
        vec = mine.direction(0, 0, other)
        # The states differ by a uniform +2 shift, so the direction must be the
        # normalised all-ones vector.
        assert np.allclose(vec, 1.0 / np.sqrt(8), atol=1e-5)

    def test_identical_states_raise(self):
        mine, other = self._pair(shift=0.0)
        with pytest.raises(ValueError, match="identical"):
            mine.direction(0, 0, other)

    def test_requires_patches_on_both_sides(self):
        reduced = CaptureSchedule(layers=(0,))
        a = ResidualCapture(schedule=reduced)
        a.record(0, 0, _activation())
        full = CaptureSchedule(layers=(0,), keep_patches=True)
        b = ResidualCapture(schedule=full)
        b.record(0, 0, _activation())
        with pytest.raises(ValueError, match="keep_patches"):
            a.direction(0, 0, b)

    def test_hidden_mismatch_raises(self):
        schedule = CaptureSchedule(layers=(0,), keep_patches=True)
        a = ResidualCapture(schedule=schedule)
        a.record(0, 0, _activation(12, 8))
        b = ResidualCapture(schedule=schedule)
        b.record(0, 0, _activation(12, 4))
        with pytest.raises(ValueError, match="hidden size mismatch"):
            a.direction(0, 0, b)

    def test_frame_window_changes_the_direction(self):
        # If the two rollouts differ only in frame 0, restricting to frame 0 must
        # give a strong direction and restricting to a later frame a weak one.
        schedule = CaptureSchedule(layers=(0,), keep_patches=True, timesteps=(0,))
        base = _activation(12, 8, seed=2)  # 4 frames x 3 patches
        drift = base.copy()
        drift[0, :3] += 5.0  # only frame 0 changes (axis 0 is the batch axis)

        other = ResidualCapture(schedule=schedule)
        other.record(0, 0, base)
        mine = ResidualCapture(schedule=schedule)
        mine.record(0, 0, drift)

        # Restricting to frame 0 keeps the full difference in magnitude.
        vec0 = mine.direction(0, 0, other, frames="0:1", n_frames=4)
        # Frames 1..3 are identical, so the mean difference there is zero.
        with pytest.raises(ValueError, match="identical"):
            mine.direction(0, 0, other, frames="1:4", n_frames=4)
        assert np.linalg.norm(vec0) == pytest.approx(1.0, rel=1e-5)


class TestSubsample:
    """Captures must fit on disk without breaking the frame window.

    A full video residual at 960x960 is 27,900 tokens x 3072 wide = 171MB per
    (layer, step, stream). A second compression pass was already removed; this is the
    other half of making a multi-layer pair affordable.
    """

    def test_stride_one_is_a_no_op(self):
        matrix = np.arange(24, dtype=np.float32).reshape(12, 2)
        assert np.array_equal(subsample_patches(matrix, 1, 4), matrix)

    def test_stride_is_taken_within_each_frame(self):
        # 4 frames x 3 patches. A stride of 2 must keep 2 patches per frame, not
        # every other token across the whole sequence.
        matrix = np.arange(24, dtype=np.float32).reshape(12, 2)
        out = subsample_patches(matrix, 2, n_frames=4)
        assert out.shape == (8, 2)
        # Frame 0 keeps rows 0 and 2; frame 1 keeps rows 3 and 5.
        assert np.array_equal(out[0], matrix[0])
        assert np.array_equal(out[1], matrix[2])
        assert np.array_equal(out[2], matrix[3])
        assert np.array_equal(out[3], matrix[5])

    def test_result_still_divides_by_the_frame_count(self):
        # frame_window raises when the token count does not divide by n_frames, so a
        # flat stride that left 11 tokens for 4 frames would break every window.
        matrix = np.arange(48, dtype=np.float32).reshape(12, 4)
        out = subsample_patches(matrix, 2, n_frames=4)
        assert out.shape[0] % 4 == 0
        frame_window(out, "0:2", 4)  # must not raise

    def test_window_still_selects_the_same_frames(self):
        matrix = np.arange(24, dtype=np.float32).reshape(12, 2)
        out = subsample_patches(matrix, 2, n_frames=4)
        # Frame 1 after subsampling must be the stride of original frame 1.
        original = matrix.reshape(4, 3, 2)
        result = out.reshape(4, 2, 2)
        assert np.array_equal(result[1], original[1][::2])

    def test_no_frame_count_falls_back_to_a_flat_stride(self):
        matrix = np.arange(24, dtype=np.float32).reshape(12, 2)
        out = subsample_patches(matrix, 2, n_frames=None)
        assert out.shape == (6, 2)

    def test_schedule_records_the_stride(self):
        schedule = CaptureSchedule(layers=(0,), patch_stride=4, n_frames=121)
        assert "stride=4" in schedule.describe()

    def test_capture_applies_the_schedule_stride(self):
        schedule = CaptureSchedule(
            layers=(0,), keep_patches=True, patch_stride=2, n_frames=4
        )
        capture = ResidualCapture(schedule=schedule)
        capture.record(0, 0, _activation(12, 8))
        # 12 patches / 4 frames = 3 per frame; a stride of 2 keeps indices 0 and 2,
        # so 2 per frame and 8 in total.
        assert capture.at(0, 0).n_patches == 8
        assert capture.at(0, 0).n_patches % 4 == 0


class TestLayerCapture:
    def test_mean_reduces_over_patches(self):
        patches = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        cap = LayerCapture(
            layer=0, timestep=0, n_patches=2, hidden=2, patches=patches
        )
        assert np.allclose(cap.mean, [2.0, 3.0])

    def test_summary_marks_whether_patches_were_kept(self):
        cap = LayerCapture(layer=1, timestep=2, n_patches=5, hidden=3)
        summary = cap.summary()
        assert summary["kept_patches"] is False
        assert summary["n_patches"] == 5


class TestTwoStacks:
    """Ovi fuses a video stack and an audio stack, block for block.

    Both stacks have a block 3, so a capture keyed on (layer, timestep) alone would
    let the audio tensor overwrite the video one at the same index. These tests pin
    the stream in the key.
    """

    def _pair(self, stream: str, *, value: float) -> ResidualCapture:
        schedule = CaptureSchedule(layers=(3,), keep_patches=True, streams=(stream,))
        capture = ResidualCapture(schedule=schedule)
        base = np.zeros((1, 4, 6), dtype=np.float32)
        capture.record(3, 0, base + value, stream=stream)
        return capture

    def test_streams_default_to_both(self):
        assert STREAMS == ("video", "audio")

    def test_same_layer_index_does_not_collide_across_streams(self):
        schedule = CaptureSchedule(layers=(3,), keep_patches=True)
        capture = ResidualCapture(schedule=schedule)
        capture.record(3, 0, np.ones((1, 4, 6), dtype=np.float32), stream="video")
        capture.record(
            3, 0, np.full((1, 7, 4), 2.0, dtype=np.float32), stream="audio"
        )
        # Different hidden widths because the two stacks are different models.
        assert capture.at(3, 0, "video").hidden == 6
        assert capture.at(3, 0, "audio").hidden == 4
        assert capture.at(3, 0, "video").n_patches == 4
        assert capture.at(3, 0, "audio").n_patches == 7

    def test_schedule_can_select_one_stream(self):
        schedule = CaptureSchedule(layers=(3,), streams=("audio",))
        capture = ResidualCapture(schedule=schedule)
        assert capture.record(3, 0, np.ones((1, 4, 6)), stream="video") is False
        assert capture.record(3, 0, np.ones((1, 4, 6)), stream="audio") is True
        assert capture.skipped == 1

    def test_streams_reports_what_was_recorded(self):
        schedule = CaptureSchedule(layers=(3,), keep_patches=True)
        capture = ResidualCapture(schedule=schedule)
        capture.record(3, 0, np.ones((1, 4, 6), dtype=np.float32), stream="audio")
        assert capture.streams() == ["audio"]
        assert capture.layers("audio") == [3]
        assert capture.layers("video") == []

    def test_direction_rejects_a_cross_stream_comparison(self):
        # Two captures of different streams must not be subtracted: their hidden
        # sizes differ, and a width mismatch is the one case that would fail loudly.
        video = self._pair("video", value=1.0)
        audio = self._pair("audio", value=2.0)
        with pytest.raises(KeyError, match="stream=audio"):
            video.direction(3, 0, audio, stream="audio")


class TestWindowedMean:
    def test_frame_window_on_audio_is_refused(self):
        # Audio has no (F, H, W) grid. Slicing it by "frames" would average a time
        # range unrelated to the video's frame boundaries and return a number.
        matrix = np.ones((8, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="only defined for the video stream"):
            _windowed_mean(matrix, "0:1", 4, "audio")

    def test_no_window_works_on_audio(self):
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        assert np.allclose(_windowed_mean(matrix, None, 4, "audio"), [2.0, 3.0])

    def test_video_window_still_slices(self):
        matrix = np.arange(16, dtype=np.float32).reshape(8, 2)
        windowed = _windowed_mean(matrix, "0:1", 4, "video")
        assert np.allclose(windowed, matrix[:2].mean(axis=0))


class TestPatchCountMismatch:
    def test_direction_rejects_different_patch_counts(self):
        # A direction between a 5s and a 10s rollout subtracts vectors of different
        # length; without this guard numpy broadcasts and returns a plausible wrong
        # answer.
        schedule = CaptureSchedule(layers=(0,), keep_patches=True)
        short = ResidualCapture(schedule=schedule)
        short.record(0, 0, np.ones((1, 8, 4), dtype=np.float32))
        long = ResidualCapture(schedule=schedule)
        long.record(0, 0, np.ones((1, 16, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="patch count mismatch"):
            short.direction(0, 0, long)
