"""Tests for the paired statistics behind the Stage 7 v3 write-up."""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "analyse_v3_paired.py"
_EVIDENCE = _ROOT / "artifacts" / "v3_5s"


def _load():
    spec = importlib.util.spec_from_file_location("analyse_v3_paired", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyse_v3_paired"] = module
    spec.loader.exec_module(module)
    return module


stats = _load()


class TestDistances:
    def test_relative_distance_is_asymmetric(self):
        ref = {"sharpness": 10.0}
        over = stats.relative_distance({"sharpness": 20.0}, ref, ["sharpness"])
        under = stats.relative_distance({"sharpness": 5.0}, ref, ["sharpness"])
        assert over == pytest.approx(1.0)
        assert under == pytest.approx(0.5)

    def test_log_distance_treats_double_and_half_alike(self):
        ref = {"sharpness": 10.0}
        over = stats.log_distance({"sharpness": 20.0}, ref, ["sharpness"])
        under = stats.log_distance({"sharpness": 5.0}, ref, ["sharpness"])
        assert over == pytest.approx(under)

    def test_identical_reading_is_zero(self):
        q = {m: 1.5 for m in stats.METRICS}
        assert stats.relative_distance(q, q) == 0
        assert stats.log_distance(q, q) == 0

    def test_log_distance_rejects_a_non_positive_ratio(self):
        with pytest.raises(ValueError):
            stats.log_distance({"sharpness": 0.0}, {"sharpness": 1.0}, ["sharpness"])


class TestSignFlip:
    def test_all_one_sign_gives_the_smallest_possible_p(self):
        # Only the observed pattern and its mirror reach the observed mean: 2 / 2**9.
        assert stats.sign_flip_p([-1.0] * 9) == pytest.approx(2 / 512)

    def test_six_clips_cannot_go_below_one_in_thirty_two(self):
        assert stats.sign_flip_p([0.3, 0.1, 0.2, 0.5, 0.4, 0.6]) == pytest.approx(2 / 64)

    def test_symmetric_differences_are_not_significant(self):
        assert stats.sign_flip_p([1.0, -1.0, 2.0, -2.0]) == pytest.approx(1.0)

    def test_too_many_pairs_is_refused_rather_than_enumerated(self):
        with pytest.raises(ValueError):
            stats.sign_flip_p([1.0] * (stats.MAX_EXACT + 1))


class TestPaired:
    def test_counts_and_interval(self):
        s = stats.paired([-0.2, -0.1, 0.05, -0.3])
        assert s["n"] == 4
        assert s["negative"] == 3 and s["positive"] == 1
        assert s["ci95"][0] <= s["mean"] <= s["ci95"][1]

    def test_bootstrap_is_reproducible(self):
        x = [-0.2, -0.1, 0.05, -0.3, 0.02]
        assert stats.bootstrap_interval(x) == stats.bootstrap_interval(x)


@pytest.fixture(scope="module")
def result():
    by, tags = stats.load(_EVIDENCE)
    return {
        "tags": tags,
        "distance": stats.distance_comparisons(by, tags),
        "decomposition": stats.decomposition(by, tags),
        "adherence": stats.adherence_comparisons(_EVIDENCE, tags),
    }


@pytest.mark.skipif(not (_EVIDENCE / "eval.json").exists(), reason="v3 evidence not present")
class TestCommittedEvidence:
    """The numbers quoted in docs/stage7-v3-5s.md, recomputed from the committed JSON."""

    def test_nine_clips_three_of_them_river(self, result):
        assert len(result["tags"]) == 9
        assert sum(t.startswith("river") for t in result["tags"]) == 3

    def test_adapter_against_plain_12_steps(self, result):
        rel = result["distance"]["relative"]["paired"]["v2_euler12 - base_unipc12"]
        assert rel["all"]["mean"] == pytest.approx(-0.152, abs=0.001)
        assert rel["all"]["negative"] == 6
        assert rel["all"]["sign_flip_p"] == pytest.approx(0.074, abs=0.001)
        assert rel["unseen"]["negative"] == 3
        log = result["distance"]["log"]["paired"]["v2_euler12 - base_unipc12"]
        assert log["all"]["negative"] == 7
        assert log["all"]["sign_flip_p"] == pytest.approx(0.016, abs=0.001)

    def test_30_steps_beats_the_adapter_on_every_unseen_prompt(self, result):
        for name in ("relative", "log"):
            s = result["distance"][name]["paired"]["v2_euler12 - base_unipc30"]["unseen"]
            assert s["negative"] == 0 and s["n"] == 6

    def test_30_steps_beats_12_steps_on_every_clip(self, result):
        s = result["distance"]["relative"]["paired"]["base_unipc30 - base_unipc12"]["all"]
        assert s["negative"] == 9
        assert s["sign_flip_p"] == pytest.approx(2 / 512)

    def test_the_gain_is_mostly_the_sharpness_term(self, result):
        d = result["decomposition"]
        assert d["base_unipc12"]["sharpness"]["mean_abs"] == pytest.approx(0.892, abs=0.001)
        assert d["v2_euler12"]["sharpness"]["mean_abs"] == pytest.approx(0.404, abs=0.001)
        assert d["v2_euler12"]["sharpness"]["below_teacher"] == 7

    def test_adherence(self, result):
        margin = result["adherence"]["clip_margin"]["v2_euler12 - base_unipc12"]["all"]
        assert margin["mean"] == pytest.approx(2.34, abs=0.01)
        assert margin["positive"] == 8
        raw = result["adherence"]["clip"]["v2_euler12 - base_unipc12"]["all"]
        assert raw["positive"] == 6
