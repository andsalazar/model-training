"""Tests for the deciding experiment's analysis (prompt-level statistics, placement)."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyse_decide.py"


def _load():
    spec = importlib.util.spec_from_file_location("analyse_decide", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyse_decide"] = module
    spec.loader.exec_module(module)
    return module


decide = _load()
PROMPTS = [f"p{i}" for i in range(10)]
SEEDS = [100, 200, 300]
#: Sharpness as a multiple of the teacher's, per variant. The others match the teacher.
FACTORS = {"teacher48": 1.0, "base_unipc30": 1.1, "base_unipc20": 1.3, "base_unipc16": 1.5,
           "base_unipc12": 2.0, "base_euler12": 2.4, "v2_euler12": 1.4}


def _quality(sharpness: float) -> dict:
    return {"sharpness": sharpness, "temporal_flicker": 0.2, "motion_coherence": 0.5, "colour_stability": 3.0}


def _write(tmp_path, factors=FACTORS, prompts=PROMPTS, seeds=SEEDS, drop=()):
    rows = []
    for i, prompt in enumerate(prompts):
        for seed in seeds:
            base = 10.0 + i + seed / 100
            for variant, factor in factors.items():
                if (prompt, seed, variant) in drop:
                    continue
                rows.append({"tag": f"{prompt}_s{seed}", "variant": variant, "quality": _quality(base * factor),
                             "seconds": 58.0 if "12" in variant else 100.0})
    (tmp_path / "eval.json").write_text(json.dumps(rows))
    return tmp_path


class TestHelpers:
    def test_tag_with_underscores_in_the_prompt(self):
        assert decide.split_tag("rainy_city_s300") == ("rainy_city", 300)

    def test_by_prompt_averages_the_seeds(self):
        values = {"a_s100": 1.0, "a_s200": 3.0, "b_s100": 5.0}
        assert decide.by_prompt(values) == {"a": 2.0, "b": 5.0}
        assert decide.by_prompt(values, {"a_s100", "b_s100"}) == {"a": 1.0, "b": 5.0}


class TestEquivalentSteps:
    CURVE = {12: 0.6, 16: 0.5, 20: 0.4, 30: 0.3}

    def test_interpolates_between_neighbouring_step_counts(self):
        assert decide.equivalent_steps(self.CURVE, 0.45)["steps"] == pytest.approx(18.0)
        assert decide.equivalent_steps(self.CURVE, 0.5)["steps"] == pytest.approx(16.0)

    def test_off_either_end_is_reported_not_extrapolated(self):
        assert decide.equivalent_steps(self.CURVE, 0.7) == {"steps": None, "note": "no better than 12 steps"}
        assert decide.equivalent_steps(self.CURVE, 0.2)["note"] == "at least as good as 30 steps"

    def test_a_curve_that_does_not_fall_places_nothing(self):
        result = decide.equivalent_steps({12: 0.5, 16: 0.55, 20: 0.4, 30: 0.3}, 0.45)
        assert result["steps"] is None and "not monotone" in result["note"]


class TestAnalyse:
    def test_prompt_is_the_unit(self, tmp_path):
        result = decide.analyse(_write(tmp_path))
        assert len(result["clips"]) == 30 and len(result["prompts"]) == 10
        s = result["paired"]["log"]["v2_euler12 - base_unipc12"]
        assert s["n"] == 10 and s["negative"] == 10
        # Ten prompts all on one side: 2 of the 1,024 sign patterns.
        assert s["sign_flip_p"] == pytest.approx(2 / 1024)
        # Only sharpness differs, so the 4-metric mean is a quarter of its log ratio.
        assert s["mean"] == pytest.approx((math.log(1.4) - math.log(2.0)) / 4)

    def test_adapter_is_placed_between_16_and_20_steps(self, tmp_path):
        place = decide.analyse(_write(tmp_path))["placement"]["log"]
        assert 16 < place["steps"] < 20
        assert place["curve"][12] > place["curve"][16] > place["curve"][20] > place["curve"][30]

    def test_pooled_matches_per_clip_when_only_a_factor_differs(self, tmp_path):
        result = decide.analyse(_write(tmp_path))
        assert result["variants"]["v2_euler12"]["pooled"] == pytest.approx(math.log(1.4) / 4)
        assert result["variants"]["v2_euler12"]["log"] == pytest.approx(math.log(1.4) / 4)

    def test_adapter_worse_than_16_steps_shows_as_positive(self, tmp_path):
        result = decide.analyse(_write(tmp_path))
        s = result["paired"]["log"]["v2_euler12 - base_unipc20"]
        assert s["mean"] > 0 and s["positive"] == 10

    def test_a_run_cut_short_compares_only_shared_clips(self, tmp_path):
        drop = {(p, 300, v) for p in PROMPTS for v in FACTORS} | {("p9", 200, "v2_euler12")}
        result = decide.analyse(_write(tmp_path, drop=drop))
        assert len(result["clips"]) == 20
        s = result["paired"]["log"]["v2_euler12 - base_unipc12"]
        assert s["n"] == 10
        assert result["variants"]["v2_euler12"]["clips"] == 19

    def test_error_rows_are_ignored(self, tmp_path):
        directory = _write(tmp_path)
        rows = json.loads((directory / "eval.json").read_text())
        rows.append({"tag": "p0_s100", "variant": "base_unipc16", "error": "generate failed"})
        (directory / "eval.json").write_text(json.dumps(rows))
        assert decide.analyse(directory)["variants"]["base_unipc16"]["clips"] == 30

    def test_audio_is_read_from_the_mp4_records(self, tmp_path):
        directory = _write(tmp_path)
        rows = []
        for p in PROMPTS:
            for s in SEEDS:
                rows.append({"file": f"{p}_s{s}.mp4", "dir": "teacher",
                             "audio": {"rms": 0.010, "centroid_mean_hz": 2000.0}})
                for v in FACTORS:
                    if v != "teacher48":
                        quiet = 0.5 if v == "base_unipc12" else 0.9
                        rows.append({"file": f"{v}_{p}_s{s}.mp4", "dir": "eval",
                                     "audio": {"rms": 0.010 * quiet, "centroid_mean_hz": 2100.0}})
        rows.append({"file": "v2_euler12_p0_s100_copy.mp4", "dir": "eval", "audio": {"error": "decode failed"}})
        (directory / "audio_mp4.json").write_text(json.dumps(rows))
        block = decide.analyse(directory)["audio"]
        assert block["loudness"]["mean"]["base_unipc12"] == pytest.approx(math.log(2.0))
        assert block["loudness"]["mean"]["v2_euler12"] == pytest.approx(-math.log(0.9))
        assert block["brightness"]["mean"]["v2_euler12"] == pytest.approx(math.log(1.05))
        s = block["loudness"]["paired"]["v2_euler12 - base_unipc12"]
        assert s["negative"] == 10 and s["n"] == 10

    def test_adherence_is_compared_by_prompt(self, tmp_path):
        directory = _write(tmp_path)
        rows = [{"tag": f"{p}_s{s}", "variant": v, "clip": 25.0, "clip_margin": 19.0 - 3.0 * (v == "base_unipc12"),
                 "clap_margin": None}
                for p in PROMPTS for s in SEEDS for v in FACTORS]
        (directory / "adherence.json").write_text(json.dumps(rows))
        block = decide.analyse(directory)["adherence"]["clip_margin"]
        assert block["mean"]["base_unipc12"] == pytest.approx(16.0)
        s = block["paired"]["v2_euler12 - base_unipc12"]
        assert s["mean"] == pytest.approx(3.0) and s["positive"] == 10 and s["n"] == 10
