"""Tests for attaching DiT capture to a real Ovi model.

Every test here runs against a torch stand-in that reproduces the one structural fact
that matters: Ovi's fused path calls the block's *submodules* rather than the block
itself. A hook on the block would never fire, and the capture would come back empty —
which is indistinguishable from a capture that found no structure. So the stand-in
deliberately omits any block-level ``forward`` on the fused path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from wsp.dit import CaptureSchedule, ResidualCapture
from wsp.hooks import DitHooks, StepTracker, _timestep_key

pytest.importorskip("torch")


class _Block(nn.Module):
    """Stands in for Ovi's WanAttentionBlock, minus a usable forward.

    ``forward`` exists only so the class is a valid module; the fused path below
    never calls it. That is the whole point.

    ``norm1`` is a scaling layer rather than ``Identity`` so a pre-hook on it can be
    told apart from a hook on its output: with an identity norm the residual and the
    normalised value are the same tensor and the distinction is untestable.
    """

    def __init__(self, dim: int = 8):
        super().__init__()
        self.norm1 = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.Identity()
        self.norm3 = nn.Identity()
        self.self_attn = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Linear(dim, dim, bias=False)
        self.cross_attn = nn.Linear(dim, dim, bias=False)

    def forward(self, x, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError(
            "the unfused block forward was called; the fused stand-in below "
            "reaches into the submodules instead"
        )


class _Stack(nn.Module):
    def __init__(self, n_blocks: int, dim: int, n_tokens: int):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(dim) for _ in range(n_blocks)])
        self.n_tokens = n_tokens


class _Fusion(nn.Module):
    """Reproduces Ovi's fused loop: submodules only, never ``block(...)``."""

    def __init__(self, n_blocks=4, vid_dim=8, audio_dim=8, vid_tokens=12, aud_tokens=6):
        super().__init__()
        self.video_model = _Stack(n_blocks, vid_dim, vid_tokens)
        self.audio_model = _Stack(n_blocks, audio_dim, aud_tokens)
        self.vid_dim = vid_dim
        self.audio_dim = audio_dim
        self.vid_tokens = vid_tokens
        self.aud_tokens = aud_tokens
        self.seen: list[float] = []

    def forward(self, vid, audio, t, **_kwargs):
        value = float(t.flatten().item())
        self.seen.append(value)
        for i in range(len(self.video_model.blocks)):
            vb = self.video_model.blocks[i]
            ab = self.audio_model.blocks[i]
            # The fused path: norm1 is called on each stream's residual, and the
            # block module itself is never invoked.
            audio = audio + ab.self_attn(ab.norm1(audio)) * 0.1
            vid = vid + vb.self_attn(vb.norm1(vid)) * 0.1
            audio = audio + ab.cross_attn(ab.norm3(audio)) * 0.1
            vid = vid + vb.cross_attn(vb.norm3(vid)) * 0.1
        return vid, audio


def _model(**kwargs) -> _Fusion:
    torch.manual_seed(0)
    return _Fusion(**kwargs)


def _capture(schedule: CaptureSchedule) -> ResidualCapture:
    return ResidualCapture(schedule=schedule)


def _run(model: _Fusion, t: float = 1.0, seed: int = 0) -> None:
    gen = torch.Generator().manual_seed(seed)
    vid = torch.randn(1, model.vid_tokens, model.vid_dim, generator=gen)
    audio = torch.randn(1, model.aud_tokens, model.audio_dim, generator=gen)
    t_tensor = torch.tensor([t])
    model(vid, audio, t_tensor)


class TestStepTracker:
    def test_first_timestep_is_step_zero(self):
        tracker = StepTracker()
        assert tracker.observe(999.0) == 0
        assert tracker.observe(500.0) == 1

    def test_repeated_timestep_resolves_to_the_same_step(self):
        # Classifier-free guidance calls the model twice at one timestep, once
        # conditional and once not. Numbering calls would make those two steps.
        tracker = StepTracker()
        assert tracker.observe(999.0) == 0
        assert tracker.observe(999.0) == 0
        assert tracker.observe(500.0) == 1
        assert tracker.steps_seen == 2
        assert tracker.calls == 3

    def test_tensor_timestep_is_accepted(self):
        tracker = StepTracker()
        assert tracker.observe(torch.tensor([750.0])) == 0

    def test_non_scalar_timestep_is_refused(self):
        # A per-element timestep means one pass spans several steps, so there is no
        # single index to file the capture under.
        with pytest.raises(ValueError, match="expected a scalar timestep"):
            _timestep_key(torch.tensor([1.0, 2.0]))


class TestHookSite:
    def test_hook_records_the_residual_not_the_normalised_value(self):
        # The hook is a pre-hook on norm1, so it must see the residual stream --
        # the additive highway an adapter would modify -- not norm1's output.
        schedule = CaptureSchedule(layers=(0,), keep_patches=True, streams=("video",))
        capture = _capture(schedule)
        model = _model()
        gen = torch.Generator().manual_seed(0)
        vid = torch.randn(1, model.vid_tokens, model.vid_dim, generator=gen)
        audio = torch.randn(1, model.aud_tokens, model.audio_dim, generator=gen)
        with DitHooks(model, capture):
            model(vid, audio, torch.tensor([1.0]))
        recorded = capture.at(0, 0).patches
        assert np.allclose(recorded, vid[0].numpy(), atol=1e-6)
        # And demonstrably not what norm1 returns for the same input.
        normalised = model.video_model.blocks[0].norm1(vid)
        assert not np.allclose(recorded, normalised[0].detach().numpy(), atol=1e-6)


class TestAttach:
    def test_hooks_fire_on_the_fused_path(self):
        # The regression this module exists for: a hook on blocks[i] would never
        # fire here, because the fused loop calls norm1 and never the block.
        schedule = CaptureSchedule(layers=(0, 1), keep_patches=True)
        capture = _capture(schedule)
        model = _model()
        with DitHooks(model, capture) as hooks:
            _run(model)
        report = hooks.report()
        assert report["fired_total"] > 0
        assert report["recorded"] == 4  # 2 layers x 2 streams
        assert report["notes"] == []

    def test_a_block_level_hook_would_not_fire(self):
        # Pins the actual reason norm1 is the hook site, rather than trusting it.
        model = _model()
        fired = []
        for block in model.video_model.blocks:
            block.register_forward_hook(lambda *a: fired.append(1))
        _run(model)
        assert fired == []

    def test_both_streams_are_captured_from_one_rollout(self):
        schedule = CaptureSchedule(layers=(0,), keep_patches=True)
        capture = _capture(schedule)
        model = _model()
        with DitHooks(model, capture):
            _run(model)
        assert capture.streams() == ["audio", "video"]
        assert capture.at(0, 0, "video").n_patches == 12
        assert capture.at(0, 0, "audio").n_patches == 6

    def test_schedule_can_decline_a_stream(self):
        schedule = CaptureSchedule(layers=(0,), keep_patches=True, streams=("audio",))
        capture = _capture(schedule)
        model = _model()
        with DitHooks(model, capture) as hooks:
            _run(model)
        assert capture.streams() == ["audio"]
        assert hooks.report()["decoded_declined"] > 0

    def test_layer_beyond_the_stack_is_rejected(self):
        schedule = CaptureSchedule(layers=(99,), keep_patches=True)
        model = _model(n_blocks=2)
        with pytest.raises(IndexError, match="layer 99 requested"):
            DitHooks(model, _capture(schedule)).attach()

    def test_model_without_stacks_is_rejected(self):
        schedule = CaptureSchedule(layers=(0,), keep_patches=True)
        with pytest.raises(AttributeError, match="neither video_model nor audio_model"):
            DitHooks(nn.Linear(2, 2), _capture(schedule)).attach()


class TestDetach:
    def test_detach_removes_hooks_and_restores_forward(self):
        schedule = CaptureSchedule(layers=(0,), keep_patches=True)
        capture = _capture(schedule)
        model = _model()
        original = model.forward
        hooks = DitHooks(model, capture).attach()
        hooks.detach()
        assert model.forward == original
        _run(model)
        assert hooks.report()["fired_total"] == 0

    def test_detach_is_idempotent(self):
        schedule = CaptureSchedule(layers=(0,), keep_patches=True)
        model = _model()
        hooks = DitHooks(model, _capture(schedule)).attach()
        hooks.detach()
        hooks.detach()

    def test_attaching_twice_is_refused(self):
        schedule = CaptureSchedule(layers=(0,), keep_patches=True)
        model = _model()
        hooks = DitHooks(model, _capture(schedule)).attach()
        with pytest.raises(RuntimeError, match="already attached"):
            hooks.attach()


class TestReport:
    def test_empty_capture_says_it_measured_nothing(self):
        # The rule this project keeps relearning: a tool that cannot look must say
        # so, not return zero.
        schedule = CaptureSchedule(layers=(0,), keep_patches=True)
        model = _model()
        hooks = DitHooks(model, _capture(schedule)).attach()
        report = hooks.report()
        assert report["fired_total"] == 0
        assert any("never called" in n for n in report["notes"])
        assert any("not evidence" in n for n in report["notes"])

    def test_declined_by_schedule_is_reported(self):
        schedule = CaptureSchedule(layers=(0,), keep_patches=True, timesteps=(7,))
        capture = _capture(schedule)
        model = _model()
        with DitHooks(model, capture) as hooks:
            _run(model)
        report = hooks.report()
        assert report["recorded"] == 0
        assert report["fired_total"] > 0
        assert any("check that the schedule matches" in n for n in report["notes"])

    def test_report_names_the_hook_site(self):
        schedule = CaptureSchedule(layers=(0,), keep_patches=True)
        model = _model()
        report = DitHooks(model, _capture(schedule)).attach().report()
        assert "norm1" in report["hook_site"]

    def test_steps_seen_counts_distinct_timesteps(self):
        schedule = CaptureSchedule(layers=(0,), keep_patches=True)
        model = _model()
        with DitHooks(model, _capture(schedule)) as hooks:
            _run(model, t=999.0)
            _run(model, t=500.0)
        report = hooks.report()
        assert report["steps_seen"] == 2
        assert report["model_calls"] == 2


class TestGuidance:
    def test_two_passes_at_one_timestep_land_on_one_step(self):
        # CFG runs the model twice per step. If that consumed two step indices the
        # schedule would select the conditional and unconditional passes as if they
        # were different denoising steps.
        schedule = CaptureSchedule(layers=(0,), keep_patches=True, streams=("video",))
        capture = _capture(schedule)
        model = _model()
        with DitHooks(model, capture) as hooks:
            _run(model, t=750.0, seed=1)
            _run(model, t=750.0, seed=2)
        tracker = hooks.tracker
        assert tracker.steps_seen == 1
        # The second pass overwrote the first at the same key rather than creating
        # a second step.
        assert tracker.calls == 2
        assert capture.timesteps("video") == [0]
