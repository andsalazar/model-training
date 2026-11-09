"""Which part of the adapter causes the end-of-clip brightness flash? A causal ablation.

The WSP read (pod_wsp_adapter.py) found the video adapters write uniformly across latent
frames, so the flash is not written directly into the last frames. This switches parts
of the adapter off one at a time and measures the flash on the clips that showed it.

Variants (all 12 Euler steps on the UniPC grid, same seed):

``full``           the adapter as trained
``video_only``     audio adapters off
``audio_only``     video adapters off
``no_last2_steps`` adapter off for the last two sampling steps
``no_first2_steps`` adapter off for the first two
``mask_last2``     adapter writes zeroed on the last two latent frames / audio bins
``audio_off_first2`` audio adapters off for the first two steps, video adapters always on
``base``           adapter off everywhere

The flash measure is the largest frame-to-frame change in mean brightness over the last
8 frames, next to the same over the rest of the clip.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OVI_ROOT = Path(os.environ.get("OVI_ROOT", "/local/Ovi"))
for path in (str(OVI_ROOT), str(Path(__file__).resolve().parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from pod_adapter_run import P, STUDENT_STEPS, load_engine  # noqa: E402

VARIANTS = ["full", "video_only", "audio_only", "no_last2_steps", "no_first2_steps", "no_first1_step",
            "mask_last2", "audio_off_first2", "base"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(OVI_ROOT / "ovi/configs/inference/inference_fusion_wsp.yaml"))
    p.add_argument("--model-name", default=None)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, default=None, help="also save MP4s and score them")
    return p.parse_args(argv)


def flash_stats(video: np.ndarray) -> dict:
    """``video`` is Ovi's (C, F, H, W) output in [-1, 1]."""
    frames = (np.clip(video, -1, 1) + 1) * 127.5
    luma = (frames[0] * 0.299 + frames[1] * 0.587 + frames[2] * 0.114).mean(axis=(1, 2))
    jumps = np.abs(np.diff(luma))
    return {"last8_max_jump": round(float(jumps[-8:].max()), 2),
            "body_max_jump": round(float(jumps[:-8].max()), 2),
            "end_luma_change": round(float(luma[-1] - luma[-9]), 2)}


class Gate:
    """Controls which adapters act, at which steps, on which tokens."""

    def __init__(self, engine, adapters):
        self.adapters = adapters
        self.engine = engine
        self.streams = {"video": True, "audio": True}
        self.off_steps: set[int] = set()
        self.mask_last = 0
        self.audio_off_steps: set[int] = set()
        self.step = -1
        self._last_t = None
        self._wrap_modules()
        self._wrap_forward()

    def _wrap_forward(self):
        model = self.engine.model
        original = model.forward
        gate = self

        def forward(*args, **kwargs):
            t = kwargs.get("t", args[2] if len(args) > 2 else None)
            value = float(t.flatten()[0]) if t is not None else None
            if value != gate._last_t:
                gate.step += 1
                gate._last_t = value
            gate.adapters.enabled = gate.step not in gate.off_steps
            return original(*args, **kwargs)

        model.forward = forward

    def _wrap_modules(self):
        import torch

        n_frames = self.engine.video_latent_length
        for stream, table in (("video", self.adapters.video), ("audio", self.adapters.audio)):
            for module in table.values():
                original = module.forward

                def forward(x, original=original, stream=stream):
                    if not self.streams[stream]:
                        return x
                    if stream == "audio" and self.step in self.audio_off_steps:
                        return x
                    out = original(x)
                    if self.mask_last:
                        bins = torch.tensor_split(torch.arange(x.shape[1], device=x.device), n_frames)
                        start = int(bins[-self.mask_last][0])
                        out = out.clone()
                        out[:, start:] = x[:, start:]
                    return out

                module.forward = forward

    def configure(self, variant: str):
        self.streams = {"video": True, "audio": True}
        self.off_steps, self.mask_last, self.audio_off_steps = set(), 0, set()
        self.step, self._last_t = -1, None
        if variant == "video_only":
            self.streams["audio"] = False
        elif variant == "audio_only":
            self.streams["video"] = False
        elif variant == "no_last2_steps":
            self.off_steps = {STUDENT_STEPS - 2, STUDENT_STEPS - 1}
        elif variant == "no_first2_steps":
            self.off_steps = {0, 1}
        elif variant == "no_first1_step":
            self.off_steps = {0}
        elif variant == "audio_off_first2":
            self.audio_off_steps = {0, 1}
        elif variant == "mask_last2":
            self.mask_last = 2
        elif variant == "base":
            self.off_steps = set(range(STUDENT_STEPS))


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch

    from wsp.adapter import EulerGridScheduler, install_adapters, patched_schedulers

    engine, sampler = load_engine(args)
    adapters = install_adapters(engine.model, [0, 1, 2], 8).to(next(engine.model.parameters()).device)
    adapters.load_state_dict(torch.load(args.adapter))
    gate = Gate(engine, adapters)
    euler = lambda stream, s, t: EulerGridScheduler([float(x) for x in t], [float(x) for x in s.sigmas])  # noqa: E731
    rows = []
    for tag in args.tags:
        name, seed = tag.rsplit("_", 1)
        for variant in args.variants:
            gate.configure(variant)
            t0 = time.time()
            with patched_schedulers(engine, euler):
                result = engine.generate(text_prompt=P[name], seed=int(seed), sample_steps=STUDENT_STEPS,
                                         video_frame_height_width=[960, 960], **sampler)
            if result is None:
                rows.append({"tag": tag, "variant": variant, "error": "generate failed"})
                continue
            video, audio, _ = result
            row = {"tag": tag, "variant": variant, "steps_seen": gate.step + 1,
                   "seconds": round(time.time() - t0, 1), **flash_stats(video)}
            if args.save_dir:
                from pod_adapter_run import _audio, _quality, save_mp4

                args.save_dir.mkdir(parents=True, exist_ok=True)
                mp4 = args.save_dir / f"{variant}_{name}_s{seed}.mp4"
                save_mp4(mp4, video, audio)
                row["quality"] = _quality(mp4)
                row["audio"] = _audio(audio)
            rows.append(row)
            print(json.dumps(row), flush=True)
            args.out.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
