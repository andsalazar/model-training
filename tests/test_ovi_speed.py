"""Tests for the speed experiment script's pure logic.

The script needs a GPU and the model to run end to end, so it cannot execute in CI.
But the config writer and the log parser are pure functions and both fail silently
when wrong — a malformed YAML produces a confusing Ovi error, and a bad VRAM regex
reports a number that looks plausible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pod_ovi_speed import (  # noqa: E402
    DEFAULT_PROMPT,
    DEFAULT_STEPS,
    _parse_peak_vram,
    _write_config,
)


class TestWriteConfig:
    def test_writes_all_required_keys(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        _write_config(
            cfg,
            model_name="960x960_10s",
            steps=20,
            prompt="a test prompt",
            output_dir=tmp_path / "out",
            seed=7,
        )
        text = cfg.read_text()
        for key in (
            "ckpt_dir",
            "output_dir",
            "sample_steps",
            "solver_name",
            "model_name",
            "shift",
            "sp_size",
            "audio_guidance_scale",
            "video_guidance_scale",
            "mode",
            "fp8",
            "cpu_offload",
            "seed",
            "text_prompt",
            "slg_layer",
        ):
            assert f"{key}:" in text, f"missing key {key}"

    def test_steps_and_model_are_written(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        _write_config(
            cfg,
            model_name="960x960_5s",
            steps=12,
            prompt="p",
            output_dir=tmp_path,
            seed=1,
        )
        text = cfg.read_text()
        assert "sample_steps: 12" in text
        assert 'model_name: "960x960_5s"' in text

    def test_prompt_with_colon_and_commas_is_quoted(self, tmp_path):
        # An unquoted YAML scalar containing ": " is parsed as a mapping and Ovi
        # fails with a confusing error. The default prompt has exactly this shape.
        cfg = tmp_path / "cfg.yaml"
        _write_config(
            cfg,
            model_name="960x960_5s",
            steps=50,
            prompt=DEFAULT_PROMPT,
            output_dir=tmp_path,
            seed=1,
        )
        text = cfg.read_text()
        line = next(ln for ln in text.splitlines() if ln.startswith("text_prompt:"))
        assert line.startswith('text_prompt: "')
        assert line.endswith('"')

    def test_embedded_quotes_are_escaped(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        _write_config(
            cfg,
            model_name="960x960_5s",
            steps=50,
            prompt='a "quoted" phrase',
            output_dir=tmp_path,
            seed=1,
        )
        text = cfg.read_text()
        assert '\\"quoted\\"' in text

    def test_creates_parent_directories(self, tmp_path):
        cfg = tmp_path / "nested" / "deep" / "cfg.yaml"
        _write_config(
            cfg,
            model_name="960x960_5s",
            steps=50,
            prompt="p",
            output_dir=tmp_path,
            seed=1,
        )
        assert cfg.exists()

    def test_output_dir_is_absolute(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        _write_config(
            cfg,
            model_name="960x960_5s",
            steps=50,
            prompt="p",
            output_dir=tmp_path / "out",
            seed=1,
        )
        text = cfg.read_text()
        line = next(ln for ln in text.splitlines() if ln.startswith("output_dir:"))
        assert str(tmp_path) in line


class TestParsePeakVram:
    def test_finds_gib_figure(self):
        log = "torch.cuda: allocated 12.34 GiB of memory"
        assert _parse_peak_vram(log) == pytest.approx(12.34)

    def test_finds_mib_figure_converted_to_gib(self):
        log = "peak 2048 MiB"
        assert _parse_peak_vram(log) == pytest.approx(2.0)

    def test_returns_largest_of_several(self):
        log = "start 4.00 GiB\nmid 9.50 GiB\nend 7.00 GiB"
        assert _parse_peak_vram(log) == pytest.approx(9.5)

    def test_returns_none_when_absent(self):
        assert _parse_peak_vram("no memory figures here") is None

    def test_ignores_unrelated_numbers(self):
        log = "step 30 of 50, loss 0.123"
        assert _parse_peak_vram(log) is None


class TestDefaults:
    def test_default_steps_descend_from_the_ovi_default(self):
        # 50 is Ovi's own default and must be the reference condition.
        assert DEFAULT_STEPS[0] == 50
        assert DEFAULT_STEPS == sorted(DEFAULT_STEPS, reverse=True)

    def test_default_prompt_has_audio_and_motion(self):
        # The prompt must exercise both branches, or an audio regression is invisible.
        lowered = DEFAULT_PROMPT.lower()
        assert "audio" in lowered
        assert "panning" in lowered or "flowing" in lowered
