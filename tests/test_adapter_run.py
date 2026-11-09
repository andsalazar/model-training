"""Tests for the Stage 7 adapter driver's pure logic (no Ovi, no GPU)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pod_adapter_run.py"


def _load():
    spec = importlib.util.spec_from_file_location("pod_adapter_run", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pod_adapter_run"] = module
    spec.loader.exec_module(module)
    return module


run = _load()


class TestSets:
    def test_unseen_prompts_never_train_or_validate(self):
        for name, spec in run.SETS.items():
            seen = {n for n, _ in spec["train"] + spec["val"]}
            if name.startswith("v3"):
                assert not seen & set(run.UNSEEN), name

    def test_v3_5s_evaluates_three_river_seeds_and_six_unseen_prompts(self):
        ev = run.SETS["v3_5s"]["eval"]
        assert [n for n, _ in ev].count("river") == 3
        assert {n for n, _ in ev} - {"river"} == set(run.UNSEEN)

    def test_every_prompt_is_defined(self):
        for spec in run.SETS.values():
            for name, _ in spec["train"] + spec["val"] + spec["eval"]:
                assert name in run.P


class TestAdapterSpec:
    def test_path_only(self):
        s = run.parse_adapter_spec("v2=/local/a/adapter_best.pt")
        assert s["name"] == "v2" and s["path"] == Path("/local/a/adapter_best.pt")
        assert s["blocks"] is None and s["audio_off"] == set()

    def test_options(self):
        s = run.parse_adapter_spec("fix=/x/y.pt@blocks=0,1,2,3@audio_off=0,1")
        assert s["blocks"] == [0, 1, 2, 3] and s["audio_off"] == {0, 1}

    def test_malformed_and_unknown_are_rejected(self):
        with pytest.raises(ValueError):
            run.parse_adapter_spec("/no/name.pt")
        with pytest.raises(ValueError, match="unknown"):
            run.parse_adapter_spec("a=/x.pt@colour=1")


class TestFlashStats:
    def test_detects_a_jump_in_the_last_frames(self):
        video = np.zeros((3, 40, 4, 4), dtype=np.float32)
        video[:, -3:] = 0.5  # a sudden brightening near the end
        stats = run.flash_stats(video)
        assert stats["last8_max_jump"] == pytest.approx(63.75, abs=0.01)
        assert stats["body_max_jump"] == 0.0


class _FakeTrainer:
    """Enough of Trainer for ``collect``: a velocity field instead of Ovi."""

    def __init__(self, field):
        self.torch = torch
        self.device = "cpu"
        self.field = field
        self.model = type("M", (), {"train": lambda self, flag: None, "gradient_checkpointing": False})()
        self.adapters = type("A", (), {"enabled": True})()
        self.calls = []

    def initial_noise(self, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(2, 3, 4, 4, generator=g).to(torch.bfloat16), torch.randn(5, 2, generator=g).to(torch.bfloat16)

    def _velocity(self, prompt, xv, xa, t):
        self.calls.append((self.adapters.enabled, t))
        return self.field(xv.float(), t, self.adapters.enabled), self.field(xa.float(), t, self.adapters.enabled)


def _grids():
    s48 = list(np.linspace(1.0, 0.0, 49))
    t48 = [s * 1000 for s in s48[:-1]]
    return (t48[::4], s48[::4]), (t48, s48)


class TestCollect:
    def test_constant_field_gives_its_own_velocity(self):
        # A constant velocity is integrated exactly by any number of Euler steps, so the
        # fine target velocity must equal it.
        fake = _FakeTrainer(lambda x, t, on: torch.full_like(x, 0.3))
        grid12, grid48 = _grids()
        (traj,) = run.Trainer.collect(fake, [("river", 7)], grid12, grid48, substeps=4)
        assert traj["video"].shape[0] == 12 and traj["video_vel"].shape[0] == 12
        assert torch.allclose(traj["video_vel"], torch.full_like(traj["video_vel"], 0.3), atol=1e-5)

    def test_targets_use_the_base_model_from_student_states(self):
        # Student (adapter on) and base model (adapter off) follow different fields; the
        # states must come from the student and the targets from the base model.
        field = lambda x, t, on: torch.full_like(x, 1.0 if on else -2.0)  # noqa: E731
        fake = _FakeTrainer(field)
        grid12, grid48 = _grids()
        (traj,) = run.Trainer.collect(fake, [("river", 7)], grid12, grid48, substeps=2)
        x0v, _ = fake.initial_noise(7)
        expected_v1 = (x0v.float() + (grid12[1][1] - grid12[1][0]) * 1.0).to(torch.bfloat16)
        assert torch.equal(traj["video"][1], expected_v1)
        assert torch.allclose(traj["video_vel"], torch.full_like(traj["video_vel"], -2.0), atol=1e-5)
        on = [c for c in fake.calls if c[0]]
        off = [c for c in fake.calls if not c[0]]
        assert len(on) == 12 and len(off) == 12 * 2
        assert fake.adapters.enabled is True

    def test_linear_field_target_matches_explicit_substeps(self):
        fake = _FakeTrainer(lambda x, t, on: -x)
        grid12, grid48 = _grids()
        (traj,) = run.Trainer.collect(fake, [("river", 3)], grid12, grid48, substeps=4)
        k = 5
        x = traj["video"][k].float()
        for j in range(4):
            i = 4 * k + j
            x = x + (grid48[1][i + 1] - grid48[1][i]) * (-x.to(torch.bfloat16).float())
        expected = (x - traj["video"][k].float()) / (grid12[1][k + 1] - grid12[1][k])
        assert torch.allclose(traj["video_vel"][k], expected, atol=1e-5)

    def test_substeps_must_divide_the_grid_ratio(self):
        fake = _FakeTrainer(lambda x, t, on: x)
        with pytest.raises(ValueError, match="divide"):
            run.Trainer.collect(fake, [("river", 1)], *_grids(), substeps=3)

    def test_off_grid_sigmas_are_rejected(self):
        fake = _FakeTrainer(lambda x, t, on: x)
        (t12, s12), grid48 = _grids()
        s12 = [s + 0.01 for s in s12]
        with pytest.raises(ValueError, match="not on the 48-step grid"):
            run.Trainer.collect(fake, [("river", 1)], (t12, s12), grid48, substeps=4)


class TestAdherenceHelpers:
    def test_prompts_split_into_visual_and_audio(self):
        sys.path.insert(0, str(_SCRIPT.parent))
        import importlib

        adherence = importlib.import_module("pod_adherence")
        visual, audio = adherence.split_prompt(run.P["piano"])
        assert visual.startswith("Close-up of hands") and "Audio" not in visual
        assert audio == "a slow classical piano melody."
        assert adherence.split_prompt("no audio part") == ("no audio part", "")

    def test_every_eval_prompt_has_an_audio_part(self):
        sys.path.insert(0, str(_SCRIPT.parent))
        import importlib

        adherence = importlib.import_module("pod_adherence")
        for name, _ in run.SETS["v3_5s"]["eval"]:
            assert adherence.split_prompt(run.P[name])[1], name
