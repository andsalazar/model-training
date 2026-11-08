"""Tests for the Ovi attention driver's argument handling and contrast logic.

The script imports torch, omegaconf and Ovi only inside ``main``, so argument parsing and
the contrast reduction are testable without a GPU.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from wsp.attention import AttentionCapture, summarize_attention_chunked

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pod_ovi_attention.py"


def _load():
    spec = importlib.util.spec_from_file_location("pod_ovi_attention", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pod_ovi_attention"] = module
    spec.loader.exec_module(module)
    return module


attention_script = _load()


def _base_args(**overrides):
    args = {
        "--config": "/local/Ovi/ovi/configs/inference/inference_fusion_wsp.yaml",
        "--prompt": "a red fox",
        "--out": "/local/captures/attention.json",
    }
    args.update(overrides)
    # Only these options take several tokens. Splitting every value would break the
    # prompt, which legitimately contains spaces.
    multi = {"--rollouts", "--layers", "--capture-steps"}
    argv: list[str] = []
    for key, value in args.items():
        argv.append(key)
        if value is None:
            continue
        if key in multi:
            argv.extend(str(value).split())
        else:
            argv.append(str(value))
    return argv


class TestParseArgs:
    def test_defaults(self):
        args = attention_script.parse_args(_base_args())
        assert args.seed == 42
        assert args.rollouts == [20, 30]
        assert args.chunk == 512
        assert args.layers == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
        assert args.streams == ["video", "audio"]
        assert args.cfg_pass == "cond"

    def test_rollouts_accepts_two_values(self):
        args = attention_script.parse_args(_base_args(**{"--rollouts": "20 30"}))
        assert args.rollouts == [20, 30]

    def test_chunk_is_configurable(self):
        args = attention_script.parse_args(_base_args(**{"--chunk": "128"}))
        assert args.chunk == 128

    def test_negative_prompt_defaults_to_the_config(self):
        # None means "use the config's per-stream negative prompts", as inference.py
        # does. The old default of "" silently replaced them with empty strings.
        args = attention_script.parse_args(_base_args())
        assert args.negative_prompt is None

    def test_max_gap_defaults_to_the_strict_tolerance(self):
        # The default timestep-matched schedules pair exactly, so the default tolerance
        # is near zero: any real gap means the schedules were overridden.
        args = attention_script.parse_args(_base_args())
        assert args.max_gap == 1.0

    def test_max_gap_is_configurable(self):
        args = attention_script.parse_args(_base_args(**{"--max-gap": "600"}))
        assert args.max_gap == 600.0


class TestParseHeightWidth:
    def test_parses_hxw(self):
        assert attention_script._parse_height_width("960x960") == [960, 960]

    def test_accepts_uppercase_x(self):
        assert attention_script._parse_height_width("704X704") == [704, 704]

    def test_accepts_comma(self):
        assert attention_script._parse_height_width("704,704") == [704, 704]

    def test_none_stays_none(self):
        assert attention_script._parse_height_width(None) is None

    def test_rejects_one_part(self):
        with pytest.raises(ValueError, match="must be HxW"):
            attention_script._parse_height_width("960")

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError, match="must be positive"):
            attention_script._parse_height_width("0x960")


def _capture_with(layer, step, heads=2, n=4, seed=0):
    """A capture holding one summary computed from real q/k."""
    gen = np.random.default_rng(seed)
    q = gen.normal(size=(heads, n, 8)).astype(np.float32)
    k = gen.normal(size=(heads, n, 8)).astype(np.float32)
    summary = summarize_attention_chunked(q, k, layer=layer, timestep=step)
    capture = AttentionCapture()
    capture.record_summary(layer, step, summary)
    return capture


class TestContrast:
    def test_identical_captures_produce_no_contrast(self):
        a = _capture_with(0, 0, seed=0)
        b = _capture_with(0, 0, seed=0)
        rows = attention_script._contrast(a, b, {0: 1000.0}, {0: 1000.0}, min_effect=0.05)
        assert len(rows) == 1
        assert rows[0]["n_heads_changed"] == 0
        assert rows[0]["max_effect"] == 0.0

    def test_different_captures_produce_a_contrast(self):
        a = _capture_with(0, 0, seed=0)
        b = _capture_with(0, 0, seed=1)
        rows = attention_script._contrast(a, b, {0: 1000.0}, {0: 1000.0}, min_effect=0.0)
        assert len(rows) == 1
        assert rows[0]["n_heads_changed"] > 0
        assert rows[0]["max_effect"] > 0.0

    def test_alignment_is_by_timestep_not_index(self):
        # The confound this guards: under Ovi's UniPC schedule a 20-step run's step 18
        # is at t=356 and a 30-step run's step 18 at t=768. Aligning by index would
        # compare those two. Aligning by timestep pairs step 18 of the 20-step run with
        # step 27 of the 30-step run.
        a = _capture_with(0, 18, seed=0)
        b = _capture_with(0, 15, seed=1)
        rows = attention_script._contrast(
            a, b, {18: 233.24}, {15: 233.24}, min_effect=0.0
        )
        assert len(rows) == 1
        assert rows[0]["step_a"] == 18
        assert rows[0]["step_b"] == 15
        assert rows[0]["timestep_a"] == 233.24
        assert rows[0]["timestep_gap"] == 0.0

    def test_nearest_timestep_is_paired_when_none_match_exactly(self):
        # With overridden schedules the timesteps need not coincide. Nearest pairing
        # gives a partner in B, with the gap recorded.
        a = _capture_with(0, 18, seed=0)
        b = _capture_with(0, 15, seed=1)
        rows = attention_script._contrast(
            a, b, {18: 233.24}, {15: 250.0}, min_effect=0.0, max_gap=50.0
        )
        assert len(rows) == 1
        assert rows[0]["step_b"] == 15
        # The gap is reported so a loose match is visible, not hidden.
        assert rows[0]["timestep_gap"] == pytest.approx(16.76, abs=0.01)

    def test_a_gap_beyond_tolerance_is_dropped_not_contrasted(self):
        # Contrasting two far-apart timesteps would compare different points on the
        # trajectory and report the schedule difference as drift. The pair must be kept
        # as a skipped row with the reason, not contrasted.
        a = _capture_with(0, 18, seed=0)
        b = _capture_with(0, 18, seed=1)
        rows = attention_script._contrast(
            a, b, {18: 233.24}, {18: 755.85}, min_effect=0.0
        )
        assert len(rows) == 1
        assert "skipped" in rows[0]
        assert "522.61" in rows[0]["skipped"]
        assert "max_effect" not in rows[0]

    def test_tolerance_is_configurable(self):
        # A caller who wants the loose pairs can raise the tolerance; the default is
        # strict so the reported rows are near-matched by default.
        a = _capture_with(0, 18, seed=0)
        b = _capture_with(0, 18, seed=1)
        rows = attention_script._contrast(
            a, b, {18: 233.24}, {18: 755.85}, min_effect=0.0, max_gap=600.0
        )
        assert len(rows) == 1
        assert "skipped" not in rows[0]
        assert rows[0]["timestep_gap"] == pytest.approx(522.61, abs=0.01)

    def test_missing_timesteps_produce_no_contrast(self):
        # A capture without a timestep table cannot be aligned at all. Returning nothing
        # is honest; aligning by index would invent a comparison.
        a = _capture_with(0, 0, seed=0)
        b = _capture_with(0, 0, seed=1)
        rows = attention_script._contrast(a, b, {}, {}, min_effect=0.0)
        assert rows == []

    def test_shape_mismatch_is_skipped_not_raised(self):
        # A cross-attention map cannot be compared with a self-attention map. The driver
        # must record the skip and keep going, not abort the whole contrast.
        a = _capture_with(0, 0, seed=0)
        gen = np.random.default_rng(3)
        q = gen.normal(size=(2, 4, 8)).astype(np.float32)
        k = gen.normal(size=(2, 9, 8)).astype(np.float32)
        b = AttentionCapture()
        b.record_summary(0, 0, summarize_attention_chunked(q, k, layer=0, timestep=0))
        rows = attention_script._contrast(a, b, {0: 1000.0}, {0: 1000.0}, min_effect=0.0)
        assert len(rows) == 1
        assert "skipped" in rows[0]

    def test_summaries_are_json_safe(self):
        capture = _capture_with(0, 0)
        payload = attention_script._summaries(capture)
        assert "video_0_0" in payload
        # Round-trip through JSON: a numpy scalar would raise here.
        text = __import__("json").dumps(payload)
        assert "row_entropy_mean" in text


class TestContrastPairsCapturedStepsOnly:
    def test_nearest_uncaptured_step_does_not_silently_drop_the_row(self):
        # Regression: run B's full schedule had a nearer step (17) that was never
        # captured. The old code paired with it and then `continue`d, so the row
        # vanished. Pairing must be among captured steps (here 18), reported.
        a = _capture_with(0, 10, seed=0)
        b = _capture_with(0, 18, seed=1)
        rows = attention_script._contrast(
            a, b, {10: 768.0}, {17: 769.0, 18: 768.0}, min_effect=0.0
        )
        assert len(rows) == 1
        assert rows[0]["step_b"] == 18
        assert "skipped" not in rows[0]

    def test_every_captured_step_produces_a_row(self):
        a = _capture_with(0, 0, seed=0)
        b = _capture_with(0, 0, seed=1)
        rows = attention_script._contrast(
            a, b, {0: 999.0}, {0: 500.0}, min_effect=0.0
        )
        assert len(rows) == 1
        assert "skipped" in rows[0]

    def test_streams_are_contrasted_separately(self):
        gen = np.random.default_rng(0)
        a, b = AttentionCapture(), AttentionCapture()
        for capture, seed in ((a, 0), (b, 1)):
            for stream, n in (("video", 6), ("audio", 3)):
                g = np.random.default_rng(seed)
                q = g.normal(size=(2, n, 8)).astype(np.float32)
                summary = summarize_attention_chunked(
                    q, q, layer=0, timestep=0, stream=stream
                )
                capture.record_summary(0, 0, summary, stream=stream)
        rows = attention_script._contrast(a, b, {0: 999.0}, {0: 999.0}, min_effect=0.0)
        assert sorted(r["stream"] for r in rows) == ["audio", "video"]
        assert all("skipped" not in r for r in rows)
        assert all(r["effect_kind"] == "pooled_sd" for r in rows)


class TestCaptureStepsFor:
    def test_defaults_are_timestep_matched(self):
        args = attention_script.parse_args(_base_args())
        assert attention_script.capture_steps_for(args, 20) == tuple(range(0, 20, 2))
        assert attention_script.capture_steps_for(args, 30) == tuple(range(0, 30, 3))

    def test_unknown_rollout_records_every_step(self):
        args = attention_script.parse_args(_base_args())
        assert attention_script.capture_steps_for(args, 50) is None

    def test_override_per_rollout(self):
        argv = _base_args() + ["--capture-steps-for", "30=1,2", "20=5"]
        args = attention_script.parse_args(argv)
        assert attention_script.capture_steps_for(args, 30) == (1, 2)
        assert attention_script.capture_steps_for(args, 20) == (5,)

    def test_global_capture_steps_applies_to_all(self):
        argv = _base_args() + ["--capture-steps", "0", "1"]
        args = attention_script.parse_args(argv)
        assert attention_script.capture_steps_for(args, 30) == (0, 1)
