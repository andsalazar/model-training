"""Tests for the TBC-style adapter and the sampler plumbing used to train it."""

import pytest
import torch
from torch import nn

from wsp.adapter import (
    EulerGridScheduler,
    RecordingScheduler,
    ResidualAdapter,
    install_adapters,
    normalised_mse,
    patched_schedulers,
    target_velocity,
)


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim


class _Stack(nn.Module):
    def __init__(self, n, dim):
        super().__init__()
        self.blocks = nn.ModuleList(_Block(dim) for _ in range(n))


class _Fusion(nn.Module):
    """Mirrors the call shape Ovi uses: the loop calls the block forward by attribute."""

    def __init__(self, n=4, vdim=8, adim=6):
        super().__init__()
        self.video_model = _Stack(n, vdim)
        self.audio_model = _Stack(n, adim)
        self.seen = []

    def single_fusion_block_forward(self, *, vid_block, audio_block, vid, audio, **kwargs):
        self.seen.append((vid.clone(), audio.clone()))
        return vid + 1, audio + 1

    def forward(self, vid, audio):
        for vb, ab in zip(self.video_model.blocks, self.audio_model.blocks):
            vid, audio = self.single_fusion_block_forward(
                vid_block=vb, audio_block=ab, vid=vid, audio=audio
            )
        return vid, audio


class TestResidualAdapter:
    def test_identity_at_initialisation(self):
        # An untrained adapter must leave the model exactly as it was, or the base-model
        # baseline and the adapted model start from different places.
        adapter = ResidualAdapter(8, 2)
        x = torch.randn(1, 5, 8)
        assert torch.equal(adapter(x), x)

    def test_preserves_dtype(self):
        adapter = ResidualAdapter(8, 2)
        x = torch.randn(1, 5, 8).to(torch.bfloat16)
        assert adapter(x).dtype == torch.bfloat16

    def test_up_receives_gradient(self):
        adapter = ResidualAdapter(8, 2)
        adapter(torch.randn(1, 5, 8)).pow(2).sum().backward()
        assert adapter.up.weight.grad is not None
        assert adapter.up.weight.grad.abs().sum() > 0


class TestInstall:
    def test_adapters_run_only_at_chosen_blocks(self):
        model = _Fusion(n=4)
        adapters = install_adapters(model, blocks=[1], rank=2)
        with torch.no_grad():
            adapters.video["1"].up.bias.fill_(10.0)
        vid, audio = torch.zeros(1, 3, 8), torch.zeros(1, 3, 6)
        model(vid, audio)
        inputs = [v[0, 0, 0].item() for v, _ in model.seen]
        # Block 0 sees 0, block 1 sees 1 + 10 (adapter), then +1 per block after.
        assert inputs == [0.0, 11.0, 12.0, 13.0]

    def test_disabled_adapters_are_a_no_op(self):
        model = _Fusion(n=3)
        adapters = install_adapters(model, blocks=[0, 1], rank=2)
        with torch.no_grad():
            adapters.video["0"].up.bias.fill_(5.0)
        adapters.enabled = False
        out, _ = model(torch.zeros(1, 2, 8), torch.zeros(1, 2, 6))
        assert torch.all(out == 3.0)

    def test_both_streams_are_adapted(self):
        model = _Fusion(n=2)
        adapters = install_adapters(model, blocks=[0], rank=2)
        with torch.no_grad():
            adapters.audio["0"].up.bias.fill_(2.0)
        _, audio = model(torch.zeros(1, 2, 8), torch.zeros(1, 2, 6))
        assert torch.all(audio == 4.0)

    def test_out_of_range_block_is_rejected(self):
        with pytest.raises(IndexError):
            install_adapters(_Fusion(n=2), blocks=[5], rank=2)

    def test_missing_hook_site_is_rejected(self):
        class Bare(nn.Module):
            pass

        with pytest.raises(AttributeError, match="hook site moved"):
            install_adapters(Bare(), blocks=[0], rank=2)


class TestTargets:
    def test_euler_step_with_target_velocity_lands_on_the_target(self):
        # The whole training objective rests on this identity.
        x_k, x_next = torch.randn(4, 3), torch.randn(4, 3)
        v = target_velocity(x_k, x_next, 0.9, 0.6)
        sched = EulerGridScheduler(timesteps=[900.0], sigmas=[0.9, 0.6])
        (out,) = sched.step(v, 900.0, x_k)
        assert torch.allclose(out, x_next, atol=1e-6)

    def test_flow_convention_moves_toward_x0(self):
        # x_t = (1 - s) x0 + s eps, v = eps - x0. One Euler step to sigma 0 gives x0.
        x0, eps = torch.randn(5), torch.randn(5)
        sigma = 0.7
        x_t = (1 - sigma) * x0 + sigma * eps
        sched = EulerGridScheduler(timesteps=[700.0], sigmas=[sigma, 0.0])
        (out,) = sched.step(eps - x0, 700.0, x_t)
        assert torch.allclose(out, x0, atol=1e-6)

    def test_zero_length_interval_is_rejected(self):
        with pytest.raises(ValueError):
            target_velocity(torch.zeros(2), torch.ones(2), 0.5, 0.5)

    def test_normalised_mse_is_scale_free(self):
        target = torch.randn(100)
        pred = target + 0.1 * torch.randn(100)
        a = normalised_mse(pred, target)
        b = normalised_mse(pred * 1000, target * 1000)
        assert torch.allclose(a, b, rtol=1e-4)


class TestEulerGrid:
    def test_off_grid_timestep_is_rejected(self):
        sched = EulerGridScheduler(timesteps=[999.0, 500.0], sigmas=[1.0, 0.5, 0.0])
        with pytest.raises(ValueError, match="not on the grid"):
            sched.step(torch.zeros(2), 750.0, torch.zeros(2))

    def test_sigma_count_is_checked(self):
        with pytest.raises(ValueError, match="one more sigma"):
            EulerGridScheduler(timesteps=[999.0], sigmas=[1.0])


class TestRecording:
    def test_records_chosen_inputs_and_the_final_output(self):
        inner = EulerGridScheduler(timesteps=[999.0, 600.0, 300.0], sigmas=[1.0, 0.6, 0.3, 0.0])
        rec = RecordingScheduler(inner=inner, record=[999.0, 300.0], n_steps=3)
        x = torch.ones(1, 4)
        states = [x]
        for t in (999.0, 600.0, 300.0):
            (x,) = rec.step(torch.ones(1, 4), t, x, return_dict=False)
            states.append(x)
        assert sorted(rec.samples) == [300.0, 999.0]
        assert torch.equal(rec.samples[999.0], states[0][0])
        assert torch.equal(rec.samples[300.0], states[2][0])
        assert torch.equal(rec.final, states[3][0])

    def test_delegates_attributes(self):
        inner = EulerGridScheduler(timesteps=[999.0], sigmas=[1.0, 0.0])
        rec = RecordingScheduler(inner=inner, record=[], n_steps=1)
        assert rec.sigmas == [1.0, 0.0]


class TestPatchedSchedulers:
    def test_video_then_audio_and_restores(self):
        class Engine:
            def get_scheduler_time_steps(self, sampling_steps, solver_name="unipc",
                                         device=0, shift=5.0):
                return f"sched{sampling_steps}", [1, 2]

        engine = Engine()
        with patched_schedulers(engine, lambda stream, s, t: (stream, s)) as made:
            v, _ = engine.get_scheduler_time_steps(sampling_steps=12)
            a, _ = engine.get_scheduler_time_steps(sampling_steps=12)
        assert v == ("video", "sched12") and a == ("audio", "sched12")
        assert len(made) == 2
        assert engine.get_scheduler_time_steps(sampling_steps=3)[0] == "sched3"
