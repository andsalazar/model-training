"""Tests for deriving a WSP direction from a capture pair.

The dangerous failure here is silent: pairing step 3 of a 20-step run with step 3 of a
30-step run subtracts two states from different points on the trajectory. The result
still has unit length, still produces plausible alignment scores, and means nothing. So
the alignment-by-timestep requirement is tested directly rather than assumed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pod_derive_direction.py"


def _load():
    spec = importlib.util.spec_from_file_location("pod_derive_direction", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def derive_script():
    return _load()


def _meta(steps_a: int, steps_b: int, n_a: int = 4, n_b: int = 4) -> dict:
    """A report whose two rollouts visit different timestep grids."""
    return {
        str(steps_a): {"step_timesteps": {str(i): 1000.0 - i * (1000.0 / n_a) for i in range(n_a)}},
        str(steps_b): {"step_timesteps": {str(i): 1000.0 - i * (1000.0 / n_b) for i in range(n_b)}},
    }


class TestParseCaptureKey:
    def test_parses_a_video_key(self, derive_script):
        assert derive_script.parse_capture_key("capture_20_video_3_2") == (20, "video", 3, 2)

    def test_parses_an_audio_key(self, derive_script):
        assert derive_script.parse_capture_key("capture_30_audio_11_0") == (30, "audio", 11, 0)

    def test_ignores_a_non_capture_key(self, derive_script):
        assert derive_script.parse_capture_key("video_20") is None

    def test_rejects_malformed_parts(self, derive_script):
        assert derive_script.parse_capture_key("capture_x_video_1_2") is None


class TestPairByTimestep:
    def test_pairs_equal_grids_one_to_one(self, derive_script):
        pairs = derive_script.pair_by_timestep(_meta(20, 30, 4, 4), 20, 30)
        assert [a for a, _, _, _ in pairs] == [0, 1, 2, 3]
        assert [b for _, b, _, _ in pairs] == [0, 1, 2, 3]

    def test_pairs_by_nearest_timestep_not_by_index(self, derive_script):
        # A 4-step run and a 7-step run: index 1 of each is a different noise level,
        # so the pairing must not be the identity.
        pairs = derive_script.pair_by_timestep(_meta(4, 7, 4, 7), 4, 7)
        assert [a for a, _, _, _ in pairs] == [0, 1, 2, 3]
        # Each pair's timesteps must be closer to each other than to any other option.
        for _, _, ta, tb in pairs:
            assert abs(ta - tb) <= 1000.0 / 7 + 1e-9

    def test_refuses_when_timesteps_were_not_recorded(self, derive_script):
        # Without timesteps the only option is index pairing, which is wrong. Refusing
        # is the point: the alternative produces a direction that looks valid.
        meta = {"20": {"steps": 20}, "30": {"steps": 30}}
        with pytest.raises(ValueError, match="cannot align by noise level"):
            derive_script.pair_by_timestep(meta, 20, 30)

    def test_refuses_when_one_rollout_is_missing(self, derive_script):
        with pytest.raises(ValueError, match="cannot align by noise level"):
            derive_script.pair_by_timestep({"20": {"step_timesteps": {"0": 1000.0}}}, 20, 30)


def _arrays(steps_a: int, steps_b: int, layer: int = 0, stream: str = "video") -> dict:
    rng = np.random.default_rng(0)
    out = {}
    for run in (steps_a, steps_b):
        for step in range(2):
            out[f"capture_{run}_{stream}_{layer}_{step}"] = rng.normal(
                size=(8, 4)
            ).astype(np.float16)
    return out


class TestDerive:
    def test_derives_a_direction_per_pair(self, derive_script):
        report = derive_script.derive(_arrays(20, 30), _meta(20, 30, 2, 2), 20, 30)
        assert len(report["results"]) == 2
        assert all("direction" in r for r in report["results"])

    def test_directions_are_unit_length(self, derive_script):
        report = derive_script.derive(_arrays(20, 30), _meta(20, 30, 2, 2), 20, 30)
        for entry in report["results"]:
            assert np.linalg.norm(entry["direction"]) == pytest.approx(1.0, rel=1e-5)

    def test_identical_states_produce_no_direction(self, derive_script):
        meta = _meta(20, 30, 1, 1)
        arrays = {k: np.ones((8, 4), dtype=np.float16) for k in _arrays(20, 30)}
        # Collapse to one step per rollout so the single pair is the identical one.
        arrays = {
            k: v for k, v in arrays.items() if k.endswith("_0")
        }
        report = derive_script.derive(arrays, meta, 20, 30)
        assert report["results"][0]["delta_norm"] == 0.0
        assert "direction" not in report["results"][0]
        assert "identical" in report["results"][0]["note"]

    def test_shape_mismatch_is_reported_not_subtracted(self, derive_script):
        # A 5s and a 10s rollout have different token counts; numpy would broadcast
        # or raise. Reporting it keeps the row visible instead of silently absent.
        arrays = _arrays(20, 30)
        key = "capture_30_video_0_1"
        arrays[key] = np.zeros((4, 4), dtype=np.float16)
        report = derive_script.derive(arrays, _meta(20, 30, 2, 2), 20, 30)
        skipped = [r for r in report["results"] if "skipped" in r]
        assert skipped and "shape mismatch" in skipped[0]["skipped"]

    def test_cosine_is_reported_and_bounded(self, derive_script):
        report = derive_script.derive(_arrays(20, 30), _meta(20, 30, 2, 2), 20, 30)
        for entry in report["results"]:
            assert -1.0 - 1e-5 <= entry["cosine"] <= 1.0 + 1e-5

    def test_only_the_requested_stream_is_used(self, derive_script):
        arrays = _arrays(20, 30, stream="video")
        report = derive_script.derive(arrays, _meta(20, 30, 2, 2), 20, 30, stream="audio")
        assert report["results"] == []

    def test_layers_are_discovered_from_the_arrays(self, derive_script):
        arrays = _arrays(20, 30, layer=7)
        report = derive_script.derive(arrays, _meta(20, 30, 2, 2), 20, 30)
        assert report["layers"] == [7]


class TestSummarize:
    def test_drops_raw_vectors_but_keeps_counts(self, derive_script):
        report = derive_script.derive(_arrays(20, 30), _meta(20, 30, 2, 2), 20, 30)
        summary = derive_script.summarize(report)
        assert all("direction" not in row for row in summary["rows"])
        assert summary["n_directions"] == 2
        assert summary["n_pairs"] == 2

    def test_reports_the_delta_norm_range(self, derive_script):
        report = derive_script.derive(_arrays(20, 30), _meta(20, 30, 2, 2), 20, 30)
        summary = derive_script.summarize(report)
        assert summary["delta_norm_min"] is not None
        assert summary["delta_norm_max"] >= summary["delta_norm_min"]

    def test_counts_identical_pairs_separately(self, derive_script):
        meta = _meta(20, 30, 1, 1)
        arrays = {
            k: np.ones((8, 4), dtype=np.float16)
            for k in _arrays(20, 30) if k.endswith("_0")
        }
        summary = derive_script.summarize(derive_script.derive(arrays, meta, 20, 30))
        assert summary["n_identical"] == 1
        assert summary["n_directions"] == 0


class TestProject:
    def test_scores_columns_against_the_direction(self, derive_script):
        report = derive_script.derive(_arrays(20, 30), _meta(20, 30, 2, 2), 20, 30)
        direction = report["results"][0]["direction"]
        # A weight whose first column is exactly the direction must score 1.0 there.
        weight = np.zeros((4, 6), dtype=np.float32)
        weight[:, 0] = direction
        findings = derive_script.project(report, {(0, "self_attn.o"): weight}, depth=3)
        assert findings[0]["best"] == pytest.approx(1.0, abs=1e-5)
        assert 0 in findings[0]["top_columns"]

    def test_dimension_mismatch_is_reported(self, derive_script):
        report = derive_script.derive(_arrays(20, 30), _meta(20, 30, 2, 2), 20, 30)
        findings = derive_script.project(
            report, {(0, "self_attn.o"): np.ones((99, 4), dtype=np.float32)}
        )
        assert any("skipped" in f for f in findings)

    def test_only_the_matching_layer_is_scored(self, derive_script):
        report = derive_script.derive(_arrays(20, 30), _meta(20, 30, 2, 2), 20, 30)
        weight = np.ones((4, 6), dtype=np.float32)
        findings = derive_script.project(
            report, {(0, "a"): weight, (5, "b"): weight}, depth=2
        )
        assert {f["key"] for f in findings if "skipped" not in f} == {"a"}
