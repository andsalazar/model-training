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


_EVIDENCE = Path(__file__).resolve().parents[1] / "artifacts" / "v4_decide"


@pytest.fixture(scope="module")
def result():
    return decide.analyse(_EVIDENCE)


@pytest.mark.skipif(not (_EVIDENCE / "eval.json").exists(), reason="v4 evidence not present")
class TestCommittedEvidence:
    """The numbers quoted in docs/stage7-v4-decide.md, recomputed from the committed JSON."""

    def test_complete_design(self, result):
        assert len(result["clips"]) == 30 and len(result["prompts"]) == 10
        assert all(v["clips"] == 30 for v in result["variants"].values())
        assert len(result["variants"]) == 7

    def test_wall_clock(self, result):
        seconds = {v: round(row["seconds"], 1) for v, row in result["variants"].items() if row["seconds"]}
        assert seconds == {"base_euler12": 56.4, "base_unipc12": 56.5, "base_unipc16": 71.1,
                           "base_unipc20": 85.7, "base_unipc30": 122.3, "v2_euler12": 56.5}

    def test_adapter_is_not_separated_from_12_or_16_plain_steps(self, result):
        log = result["paired"]["log"]
        vs12, vs16 = log["v2_euler12 - base_unipc12"], log["v2_euler12 - base_unipc16"]
        assert vs12["mean"] == pytest.approx(0.006, abs=0.001) and vs12["negative"] == 5
        assert vs12["sign_flip_p"] == pytest.approx(0.889, abs=0.001)
        assert vs16["mean"] == pytest.approx(0.028, abs=0.001) and vs16["negative"] == 2
        assert vs16["sign_flip_p"] == pytest.approx(0.564, abs=0.001)

    def test_20_and_30_plain_steps_beat_the_adapter(self, result):
        log = result["paired"]["log"]
        vs20, vs30 = log["v2_euler12 - base_unipc20"], log["v2_euler12 - base_unipc30"]
        assert vs20["mean"] == pytest.approx(0.100, abs=0.001) and vs20["negative"] == 1
        assert vs20["sign_flip_p"] == pytest.approx(0.020, abs=0.001)
        assert vs30["negative"] == 0 and vs30["sign_flip_p"] == pytest.approx(2 / 1024)

    def test_the_base_curve_resolves_from_16_steps_up_but_not_between_12_and_16(self, result):
        log = result["paired"]["log"]
        assert log["base_unipc16 - base_unipc12"]["sign_flip_p"] == pytest.approx(0.312, abs=0.001)
        assert log["base_unipc20 - base_unipc16"]["sign_flip_p"] == pytest.approx(0.023, abs=0.001)
        assert log["base_unipc30 - base_unipc20"]["negative"] == 10

    def test_adapter_against_the_same_sampler_without_it(self, result):
        s = result["paired"]["log"]["v2_euler12 - base_euler12"]
        assert s["mean"] == pytest.approx(-0.092, abs=0.001) and s["negative"] == 7
        assert s["sign_flip_p"] == pytest.approx(0.076, abs=0.001)

    def test_placement(self, result):
        assert result["placement"]["log"]["note"] == "no better than 12 steps"
        assert result["placement"]["pooled"]["note"] == "no better than 12 steps"
        assert result["placement"]["relative"]["steps"] == pytest.approx(17.4, abs=0.1)

    def test_pooled_measure_puts_the_adapter_behind_plain_12_steps(self, result):
        s = result["paired"]["pooled"]["v2_euler12 - base_unipc12"]
        assert s["mean"] == pytest.approx(0.076, abs=0.001) and s["negative"] == 3
        assert s["sign_flip_p"] == pytest.approx(0.041, abs=0.001)

    def test_adapter_is_softer_than_the_teacher(self, result):
        sharp = result["decomposition"]["v2_euler12"]["sharpness"]
        assert sharp["below_teacher"] == 24 and sharp["n"] == 30
        assert sharp["mean_signed"] == pytest.approx(-0.199, abs=0.001)
        assert result["decomposition"]["base_unipc12"]["sharpness"]["mean_signed"] == pytest.approx(0.008, abs=0.001)

    def test_prompt_adherence(self, result):
        margin = result["adherence"]["clip_margin"]
        assert margin["mean"]["teacher48"] == pytest.approx(19.30, abs=0.01)
        assert margin["mean"]["base_unipc12"] == pytest.approx(17.96, abs=0.01)
        assert margin["mean"]["v2_euler12"] == pytest.approx(18.59, abs=0.01)
        assert margin["paired"]["v2_euler12 - base_unipc12"]["sign_flip_p"] == pytest.approx(0.352, abs=0.001)
        assert margin["paired"]["base_unipc16 - base_unipc12"]["sign_flip_p"] == pytest.approx(0.021, abs=0.001)
        assert margin["paired"]["base_unipc12 - teacher48"]["sign_flip_p"] == pytest.approx(0.008, abs=0.001)

    def test_clips_shared_with_v3_were_reproduced_exactly_on_another_pod(self):
        v3_path = _EVIDENCE.parent / "v3_5s" / "eval.json"
        if not v3_path.exists():
            pytest.skip("v3 evidence not present")
        v3 = {(r["tag"], r["variant"]): r["quality"] for r in json.loads(v3_path.read_text()) if "error" not in r}
        v4 = {key: row["quality"] for key, row in decide.load_eval(_EVIDENCE).items()}
        shared = sorted(set(v3) & set(v4))
        assert len(shared) == 24
        assert all(v3[key][m] == v4[key][m] for key in shared for m in decide.METRICS)

    def test_starting_noise_is_per_seed_and_matches_a_gpu_redraw(self):
        noise = json.loads((_EVIDENCE / "noise.json").read_text())
        assert sorted(noise["by_seed"]) == ["100", "200", "300"]
        for entry in noise["by_seed"].values():
            assert len(entry["files"]) == 10
            for stream in ("video", "audio"):
                assert entry[stream]["same_for_every_prompt"] is True
                assert entry[stream]["redraw_cuda_matches"] is True
                assert entry[stream]["redraw_cpu_matches"] is False

    def test_audio(self, result):
        loud = result["audio"]["loudness"]["mean"]
        assert loud["v2_euler12"] == pytest.approx(0.366, abs=0.001)
        assert loud["base_unipc12"] == pytest.approx(0.254, abs=0.001)
        assert loud["base_unipc30"] == pytest.approx(0.190, abs=0.001)


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
