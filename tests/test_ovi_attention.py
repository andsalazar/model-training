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
        assert args.layers == [0, 2, 4, 8, 11]

    def test_rollouts_accepts_two_values(self):
        args = attention_script.parse_args(_base_args(**{"--rollouts": "20 30"}))
        assert args.rollouts == [20, 30]

    def test_chunk_is_configurable(self):
        args = attention_script.parse_args(_base_args(**{"--chunk": "128"}))
        assert args.chunk == 128

    def test_negative_prompt_defaults_to_empty(self):
        # Ovi's sampler always passes a value; an absent one must not be None.
        args = attention_script.parse_args(_base_args())
        assert args.negative_prompt == ""


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
        rows = attention_script._contrast(a, b, min_effect=0.05)
        assert len(rows) == 1
        assert rows[0]["n_heads_changed"] == 0
        assert rows[0]["max_effect"] == 0.0

    def test_different_captures_produce_a_contrast(self):
        a = _capture_with(0, 0, seed=0)
        b = _capture_with(0, 0, seed=1)
        rows = attention_script._contrast(a, b, min_effect=0.0)
        assert len(rows) == 1
        assert rows[0]["n_heads_changed"] > 0
        assert rows[0]["max_effect"] > 0.0

    def test_only_common_keys_are_contrasted(self):
        # A 20-step run and a 30-step run do not share every step index. Contrasting a
        # key present in only one capture would compare against nothing.
        a = _capture_with(0, 0, seed=0)
        a.record_summary(0, 1, summarize_attention_chunked(
            np.ones((2, 4, 8), dtype=np.float32),
            np.ones((2, 4, 8), dtype=np.float32),
            layer=0, timestep=1,
        ))
        b = _capture_with(0, 0, seed=1)
        rows = attention_script._contrast(a, b, min_effect=0.0)
        assert len(rows) == 1
        assert rows[0]["step"] == 0

    def test_shape_mismatch_is_skipped_not_raised(self):
        # A cross-attention map cannot be compared with a self-attention map. The driver
        # must record the skip and keep going, not abort the whole contrast.
        a = _capture_with(0, 0, seed=0)
        gen = np.random.default_rng(3)
        q = gen.normal(size=(2, 4, 8)).astype(np.float32)
        k = gen.normal(size=(2, 9, 8)).astype(np.float32)
        b = AttentionCapture()
        b.record_summary(0, 0, summarize_attention_chunked(q, k, layer=0, timestep=0))
        rows = attention_script._contrast(a, b, min_effect=0.0)
        assert len(rows) == 1
        assert "skipped" in rows[0]

    def test_summaries_are_json_safe(self):
        capture = _capture_with(0, 0)
        payload = attention_script._summaries(capture)
        assert "video_0_0" in payload
        # Round-trip through JSON: a numpy scalar would raise here.
        text = __import__("json").dumps(payload)
        assert "row_entropy_mean" in text
