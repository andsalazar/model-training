"""Tests for the Ovi capture driver's argument handling.

The script imports torch, omegaconf and Ovi only inside ``main``, so argument parsing
is testable without a GPU. That matters because a misparsed rollout list would run the
wrong pair of step counts and the resulting direction would contrast two arbitrary
points on the trajectory rather than a drifting run against a stable one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pod_ovi_capture.py"


def _load():
    spec = importlib.util.spec_from_file_location("pod_ovi_capture", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def capture_script():
    return _load()


def _base_args() -> list[str]:
    return ["--config", "cfg.yaml", "--prompt", "a fox", "--out", "o.npz"]


class TestRolloutPair:
    def test_default_pair_is_drifting_against_stable(self, capture_script):
        # The controlled contrast this whole instrument exists for: 20 steps drifted
        # colour in 2 of 3 seeds, 30 steps was stable.
        args = capture_script.parse_args(_base_args())
        assert args.rollouts == [20, 30]

    def test_rollouts_accepts_one_value(self, capture_script):
        args = capture_script.parse_args(_base_args() + ["--rollouts", "30"])
        assert args.rollouts == [30]

    def test_rollouts_accepts_three_values(self, capture_script):
        args = capture_script.parse_args(_base_args() + ["--rollouts", "20", "30", "50"])
        assert args.rollouts == [20, 30, 50]


class TestDefaults:
    def test_captures_both_streams_by_default(self, capture_script):
        # The project question is whether one method can be aimed at video and audio
        # separately, which needs both captured from the same rollout.
        args = capture_script.parse_args(_base_args())
        assert args.streams == ["video", "audio"]

    def test_layers_are_a_tupleable_list(self, capture_script):
        args = capture_script.parse_args(_base_args())
        assert args.layers == list(capture_script.DEFAULT_LAYERS)

    def test_capture_steps_are_denoising_indices(self, capture_script):
        args = capture_script.parse_args(_base_args())
        assert args.capture_steps == list(capture_script.DEFAULT_STEPS)
        assert 0 in args.capture_steps, "step 0 (noisiest) is where an adapter acts"

    def test_seed_is_reproducible_default(self, capture_script):
        args = capture_script.parse_args(_base_args())
        assert args.seed == 42


class TestValidation:
    def test_unknown_stream_is_rejected(self, capture_script):
        with pytest.raises(SystemExit):
            capture_script.parse_args(_base_args() + ["--streams", "latent"])

    def test_missing_config_is_rejected(self, capture_script):
        with pytest.raises(SystemExit):
            capture_script.parse_args(["--prompt", "x", "--out", "o.npz"])

    def test_missing_output_is_rejected(self, capture_script):
        with pytest.raises(SystemExit):
            capture_script.parse_args(["--config", "c.yaml", "--prompt", "x"])


class TestHeightWidth:
    """Ovi asserts video_frame_height_width is not None in t2v mode.

    The first real capture died on exactly that: the engine loaded the 11.66B model,
    printed the schedule, and then returned None with its own logged traceback.
    """

    def test_parses_hxw(self, capture_script):
        assert capture_script._parse_height_width("704x704") == [704, 704]

    def test_accepts_an_x_separator_case_insensitively(self, capture_script):
        assert capture_script._parse_height_width("704X1280") == [704, 1280]

    def test_accepts_a_comma_separator(self, capture_script):
        assert capture_script._parse_height_width("960,960") == [960, 960]

    def test_none_stays_none(self, capture_script):
        assert capture_script._parse_height_width(None) is None

    def test_malformed_input_is_rejected(self, capture_script):
        with pytest.raises(ValueError, match="must be HxW"):
            capture_script._parse_height_width("704")

    def test_non_positive_is_rejected(self, capture_script):
        with pytest.raises(ValueError, match="must be positive"):
            capture_script._parse_height_width("0x704")


class TestOutputPath:
    def test_out_is_a_path_not_a_string(self, capture_script):
        args = capture_script.parse_args(_base_args())
        assert isinstance(args.out, Path)

    def test_negative_prompt_defaults_to_the_config(self, capture_script):
        # None means "use the config's per-stream negative prompts", as inference.py
        # does. The old default of "" silently replaced them with empty strings.
        args = capture_script.parse_args(_base_args())
        assert args.negative_prompt is None
        assert args.cfg_pass == "cond"
