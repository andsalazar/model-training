"""Tests for the audio metric in pod_ovi_audio_check.py.

The script needs ffmpeg and real video files to run end to end, so the tests cover the
pure signal-processing part: given PCM samples, does it report the right descriptors?

The failure this guards against is the one that has bitten this project repeatedly —
a metric that returns a plausible number for the wrong reason. A silent track must not
look like a quiet one, and a trending timbre must not look like a stable one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pod_ovi_audio_check import audio_metrics  # noqa: E402

SR = 16000


def _tone(freq: float, seconds: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


class TestAudioMetrics:
    def test_silence_has_zero_rms(self):
        m = audio_metrics(np.zeros(SR, dtype=np.float32))
        assert m["rms"] == pytest.approx(0.0, abs=1e-9)
        assert m["peak"] == pytest.approx(0.0, abs=1e-9)

    def test_tone_level_is_reported(self):
        # A 0.5-amplitude sine has RMS 0.5/sqrt(2) = 0.3536.
        m = audio_metrics(_tone(440.0, 1.0))
        assert m["rms"] == pytest.approx(0.3536, rel=0.02)
        assert m["peak"] == pytest.approx(0.5, rel=0.02)

    def test_centroid_tracks_tone_frequency(self):
        # The spectral centroid of a pure tone is its frequency.
        m = audio_metrics(_tone(1000.0, 1.0))
        assert m["centroid_mean_hz"] == pytest.approx(1000.0, rel=0.05)

    def test_centroid_rises_for_a_rising_sweep(self):
        # A sweep from 300 Hz to 3000 Hz must show a positive centroid trend. This is
        # the audio analogue of the colour-drift check, so a sign error here would
        # invert the meaning of every audio verdict.
        n = SR * 2
        t = np.arange(n, dtype=np.float64) / SR
        freq = 300.0 + (3000.0 - 300.0) * t / t[-1]
        phase = 2.0 * np.pi * np.cumsum(freq) / SR
        samples = (0.5 * np.sin(phase)).astype(np.float32)
        m = audio_metrics(samples)
        assert m["centroid_slope_hz_per_frame"] > 0.0
        assert m["centroid_r2"] > 0.5

    def test_centroid_is_flat_for_a_steady_tone(self):
        m = audio_metrics(_tone(800.0, 2.0))
        assert abs(m["centroid_slope_hz_per_frame"]) < 5.0

    def test_empty_audio_is_an_error_not_a_zero(self):
        # A silent track and a missing track are different problems. Returning 0.0 for
        # both would hide a broken decode behind a plausible number.
        m = audio_metrics(np.zeros(0, dtype=np.float32))
        assert "error" in m

    def test_short_audio_does_not_crash(self):
        m = audio_metrics(_tone(440.0, 0.05))
        assert "error" not in m
        assert m["seconds"] > 0.0

    def test_seconds_matches_sample_count(self):
        m = audio_metrics(_tone(440.0, 1.5))
        assert m["seconds"] == pytest.approx(1.5, rel=0.01)


class TestStreamDetection:
    """``streams=0`` must never mean "ffprobe was missing".

    The first run of this check reported zero audio streams for every video because
    imageio_ffmpeg ships ffmpeg without ffprobe. That reads as "Ovi produces no audio",
    which is a much stronger claim than "the tool did not look".
    """

    def test_missing_ffprobe_is_reported_not_silently_zero(self, tmp_path):
        from pod_ovi_audio_check import streams

        # A file that is not a video at all: both paths fail, but the report must say
        # which mechanism was used so a zero cannot be mistaken for a measurement.
        bogus = tmp_path / "not_a_video.mp4"
        bogus.write_bytes(b"not a video")
        info = streams(bogus)
        assert "via" in info
        assert info["via"] in {"ffprobe", "ffmpeg-banner", "none"}
        # The key point: the mechanism is named, so "0 streams" is interpretable.
        assert info["via"] != "none" or info["streams"] == []
