"""Tests for the Stage 8 weight-reading replication (no API calls, no GPU)."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from wsp import reading

_ROOT = Path(__file__).resolve().parents[1]
_EVIDENCE = _ROOT / "artifacts" / "stage8_reading"
_SCRIPT = _ROOT / "scripts" / "read_weights.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("read_weights", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["read_weights"] = module
    spec.loader.exec_module(module)
    return module


def _net(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    h = reading.HIDDEN
    shapes = {"W1": (160, h), "b1": (h,), "W2": (h, h), "b2": (h,), "W3": (h, 144), "b3": (144,)}
    return {k: rng.normal(0, 0.2, size=s).astype(np.float32) for k, s in shapes.items()}


class TestShapes:
    @pytest.mark.parametrize("name", reading.SHAPES)
    def test_image_is_12_by_12_in_range_with_something_drawn(self, name):
        image = reading.render_shape(name, np.random.default_rng(0))
        assert image.shape == (12, 12)
        assert image.min() >= -1 and image.max() <= 1
        assert (image > 0).sum() >= 4

    @pytest.mark.parametrize("name", reading.SHAPES)
    def test_shape_keeps_clear_of_the_border(self, name):
        rng = np.random.default_rng(1)
        for _ in range(200):
            image = reading.render_shape(name, rng)
            border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
            assert border.max() <= 0, name

    def test_unknown_shape_is_refused(self):
        with pytest.raises(ValueError):
            reading.render_shape("hexagon", np.random.default_rng(0))

    def test_dataset_is_a_fixed_collection(self):
        a = reading.make_dataset("cross", 16, seed=3)
        assert a.shape == (16, 144)
        assert np.array_equal(a, reading.make_dataset("cross", 16, seed=3))
        assert not np.array_equal(a, reading.make_dataset("cross", 16, seed=4))


class TestDdpm:
    def test_step_encoding_matches_the_description(self):
        enc = reading.step_encoding(np.array([0, 5]))
        assert enc.shape == (2, 16)
        assert np.allclose(enc[0, :8], 0) and np.allclose(enc[0, 8:], 1)
        assert enc[1, 0] == pytest.approx(np.sin(5.0)) and enc[1, 8] == pytest.approx(np.cos(5.0))
        assert enc[1, 1] == pytest.approx(np.sin(5.0 / 10000 ** (1 / 8)), rel=1e-5)

    def test_abar_follows_the_linear_schedule(self):
        cum = reading.abar()
        assert cum[0] == pytest.approx(1 - 1e-4)
        assert cum[1] == pytest.approx((1 - 1e-4) * (1 - np.linspace(1e-4, 0.02, 1000)[1]))
        assert np.all(np.diff(cum) < 0)

    def test_forward_is_the_formula_in_the_description(self):
        net = _net()
        x = np.random.default_rng(1).normal(size=(3, 160)).astype(np.float32)
        h1 = np.maximum(x @ net["W1"] + net["b1"], 0)
        h2 = np.maximum(h1 @ net["W2"] + net["b2"], 0)
        assert np.allclose(reading.forward(net, x), h2 @ net["W3"] + net["b3"])

    def test_sampling_ends_on_the_clipped_prediction(self):
        # A network that always predicts all-white must generate all-white images.
        net = {k: np.zeros_like(v) for k, v in _net().items()}
        net["b3"][:] = 5.0
        assert np.allclose(reading.sample(net, 2, seed=0), 1.0, atol=1e-5)

    def test_training_returns_weights_in_the_description_layout(self):
        net, curve = reading.train_ddpm(reading.make_dataset("square", 64, seed=0), steps=5, batch=8)
        assert [net[k].shape for k in reading.TENSORS] == [(160, 64), (64,), (64, 64), (64,), (64, 144), (144,)]
        assert sum(net[k].size for k in reading.TENSORS) == 23_824
        assert curve == []


class TestText:
    def test_layout_and_count(self):
        text = reading.format_weights(_net())
        lines = text.split("\n")
        assert len(lines) == 305 and lines[-1] == ""
        assert lines[0] == "The network's weights (H = 64). Values rounded to 3 decimals."
        assert lines[2].startswith("W1 (layer 1 weights), shape 160 x 64")
        numbers = [tok for line in lines if line[:1] in "-0123456789" and line for tok in line.split()]
        assert len(numbers) == 23_824

    def test_negative_zero_is_printed_as_zero(self):
        net = _net()
        net["b3"][:3] = [-0.0004, 0.0004, -0.0006]
        last = reading.format_weights(net).rstrip("\n").split("\n")[-1].split()
        assert last[:3] == ["0.000", "0.000", "-0.001"]

    def test_prompt_names_no_shape(self):
        text = (reading.DESCRIPTION + reading.QUESTION).lower()
        for pattern in reading.KEYWORDS.values():
            for word in pattern.split("|"):
                assert word.replace("?", "") not in text, word

    def test_prompt_is_description_question_weights(self):
        net = _net()
        prompt = reading.build_prompt(net)
        assert prompt.startswith("THE MODEL\n")
        assert prompt.endswith(reading.format_weights(net))
        assert "\n\nQUESTION. The weights below" in prompt and "THE WEIGHTS\nThe network's weights" in prompt

    def test_network_id_depends_only_on_the_weights(self):
        a, b = _net(0), _net(1)
        assert reading.network_id(a) == reading.network_id({k: v.copy() for k, v in a.items()})
        assert reading.network_id(a) != reading.network_id(b)
        assert len(reading.network_id(a)) == 8


class TestShuffle:
    def test_values_kept_structure_destroyed(self):
        net = _net()
        shuffled = reading.shuffle_within_tensors(net, seed=0)
        for name in reading.TENSORS:
            assert shuffled[name].shape == net[name].shape
            assert np.array_equal(np.sort(shuffled[name], axis=None), np.sort(net[name], axis=None))
            assert not np.array_equal(shuffled[name], net[name])

    def test_source_is_not_modified(self):
        net = _net()
        before = {k: v.copy() for k, v in net.items()}
        reading.shuffle_within_tensors(net, seed=0)
        assert all(np.array_equal(net[k], before[k]) for k in net)


ANSWER = """Observations:
- W3 row 42 has curved rim segments, unlike a square.

Reasoning:
Curved bands suggest circular outlines, not triangles.

Images:
My best reading is **a single white circle outline on black**, roughly 8-10 pixels across,
with modest variation in size, position, and roundness.

Example:
```text
....####....
```

Confidence: 84%
"""


class TestScoring:
    def test_only_the_images_section_is_scored(self):
        score = reading.score_answer(ANSWER, "circle")
        assert score["guess"] == "circle" and score["correct"] is True
        assert score["confidence"] == 84
        assert "square" not in score["images_line"]

    def test_markdown_headings_are_recognised(self):
        answer = "**Observations:** x\n**Images:** hollow triangles on black\n**Confidence:** 70%"
        assert reading.score_answer(answer, "triangle")["correct"] is True

    @pytest.mark.parametrize(("line", "expected"), [
        ("a white plus sign in the centre", "cross"),
        ("a cross-shaped figure", "cross"),
        ("rectangular outlines on black", "square"),
        ("a bright wedge", "triangle"),
        ("a round blob", "circle"),
        ("a bright blob on a dark background", "none"),
        ("structure spread across the image, some roundness", "none"),
        ("a non-circular smear", "none"),
        ("a circle or perhaps a square", "ambiguous"),
        ("", "none"),
        (None, "missing"),
    ])
    def test_classify(self, line, expected):
        assert reading.classify(line) == expected

    def test_two_shapes_named_is_never_correct(self):
        answer = "Images: either a circle or a square\nConfidence: 50%"
        assert reading.score_answer(answer, "circle")["correct"] is False

    def test_control_has_no_right_answer(self):
        score = reading.score_answer("Images: no coherent image\nConfidence: 10%", None)
        assert score["correct"] is None and score["guess"] == "none" and score["confidence"] == 10

    def test_answer_without_sections(self):
        score = reading.score_answer("I cannot tell.", "circle")
        assert score["guess"] == "missing" and score["correct"] is False and score["confidence"] is None


class TestBudget:
    def test_refuses_a_call_that_would_cross_the_limit(self):
        script = _load_script()
        budget = script.Budget(limit=2.0)
        assert budget.reserve(1.5) is True
        assert budget.reserve(1.0) is False
        budget.settle(1.5, actual=0.4)
        assert budget.spent == pytest.approx(0.4) and budget.reserved == pytest.approx(0.0)
        assert budget.reserve(1.5) is True


_PAPER = _ROOT / "runs" / "paper-logs" / "exp2_diffusion"


@pytest.mark.skipif(not _PAPER.exists(), reason="the paper's log release is not checked out")
class TestAgainstThePaperRelease:
    """Needs github.com/RikiyaT/lm_understand-logs under runs/paper-logs (local only)."""

    def test_prompts_are_rebuilt_byte_for_byte(self):
        key = json.loads((_PAPER / "eval" / "answer_key.json").read_text())
        for network_id, info in key.items():
            with np.load(_PAPER / "networks" / f"{network_id}.npz") as z:
                net = {k: z[k] for k in reading.TENSORS}
            theirs = (_PAPER / "runs" / info["shape"] / "prompt.txt").read_text(encoding="utf-8")
            assert reading.build_prompt(net) == theirs, network_id

    def test_scorer_agrees_with_the_paper_on_its_own_replies(self):
        for row in json.loads((_PAPER / "eval" / "scores.json").read_text()):
            answer = (_PAPER / "runs" / row["true"] / "answer.md").read_text(encoding="utf-8")
            score = reading.score_answer(answer, row["true"])
            assert score["guess"] == row["guess"] and score["correct"] is row["correct"]
            assert score["confidence"] == row["confidence"]


@pytest.fixture(scope="module")
def manifest():
    return json.loads((_EVIDENCE / "manifest.json").read_text())


@pytest.mark.skipif(not (_EVIDENCE / "manifest.json").exists(), reason="stage 8 evidence not present")
class TestCommittedNetworks:
    def _net(self, network_id):
        with np.load(_EVIDENCE / "networks" / f"{network_id}.npz") as z:
            return {k: z[k] for k in reading.TENSORS}

    def test_design(self, manifest):
        trials = manifest["trials"]
        assert sorted(t["truth"] for t in trials if t["source"] == "paper") == ["circle", "square", "triangle"]
        assert sorted(t["truth"] for t in trials if t["source"] == "fresh") == [
            "circle", "cross", "cross", "square", "triangle"]
        assert [t["truth"] for t in trials if t["source"] == "control"] == [None]

    def test_every_prompt_rebuilds_to_the_recorded_hash(self, manifest):
        for trial in manifest["trials"]:
            prompt = reading.build_prompt(self._net(trial["id"]))
            assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == trial["prompt_sha256"], trial["id"]

    def test_file_names_are_the_weight_hashes(self, manifest):
        for trial in manifest["trials"]:
            if trial["source"] != "paper":
                assert reading.network_id(self._net(trial["id"])) == trial["id"]

    def test_control_is_a_shuffle_of_its_source(self, manifest):
        control = next(t for t in manifest["trials"] if t["source"] == "control")
        a, b = self._net(control["id"]), self._net(control["shuffled_from"])
        for name in reading.TENSORS:
            assert np.array_equal(np.sort(a[name], axis=None), np.sort(b[name], axis=None))

    def test_every_real_network_generates_its_shape(self, manifest):
        for trial in manifest["trials"]:
            if trial["truth"] is not None:
                assert trial["quality"]["classified_as"][trial["truth"]] >= 0.9, trial["id"]
