"""WSP second stage: read where the adapter does, and does not, bring Ovi to the teacher.

Two measurements per prompt/seed, all on the conditional CFG pass:

1. **Attention against the teacher.** Four rollouts are captured with the Stage 6
   attention hooks at the 12-step grid's timesteps (which the 48-step teacher visits at
   every 4th step, so pairs are exact): teacher-48, base Euler-12, adapter Euler-12 and
   base UniPC-12. For every (stream, block, timestep) the per-head contrast against the
   teacher is computed. ``ratio = effect(adapter) / effect(base Euler)`` says whether the
   adapter moved that site *toward* the teacher (< 1) or not (>= 1). Sites where it did
   not are where a targeted fix or extra capacity should go.

2. **Where the adapter writes.** During the adapter rollout, each adapter's own change
   ``|adapter(x) - x|`` is recorded per latent frame (video) or per time bin (audio),
   relative to ``|x|``. The end-of-clip brightness flash (stage7-v2-5s.md) predicts a
   write concentrated in the last latent frames.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

OVI_ROOT = Path(os.environ.get("OVI_ROOT", "/local/Ovi"))
for path in (str(OVI_ROOT), str(Path(__file__).resolve().parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from pod_adapter_run import P, STUDENT_STEPS, load_engine  # noqa: E402

TEACHER_STEPS = 48
LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(OVI_ROOT / "ovi/configs/inference/inference_fusion_wsp.yaml"))
    p.add_argument("--model-name", default=None)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--blocks", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--tags", nargs="+", required=True, help="prompt_seed, e.g. violin_100")
    p.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=512)
    return p.parse_args(argv)


class WriteRecorder:
    """Records each adapter's relative write per latent frame / audio time bin."""

    def __init__(self, adapters, hooks, n_frames: int):
        self.hooks = hooks
        self.n_frames = n_frames
        self.rows: list[dict] = []
        self._originals = []
        for stream, table in (("video", adapters.video), ("audio", adapters.audio)):
            for block, module in table.items():
                self._wrap(module, stream, int(block))

    def _wrap(self, module, stream: str, block: int) -> None:
        import torch

        original = module.forward
        self._originals.append((module, original))
        recorder = self

        def forward(x):
            out = original(x)
            hooks = recorder.hooks
            if hooks._current_pass == 0 and hooks._current_step is not None:
                with torch.no_grad():
                    delta = (out.float() - x.float())[0]
                    base = x.float()[0]
                    # Video tokens are frame-major, so equal splits are latent frames;
                    # audio tokens are split into the same number of time bins.
                    pairs = zip(torch.tensor_split(delta, recorder.n_frames, dim=0),
                                torch.tensor_split(base, recorder.n_frames, dim=0))
                    rel = [round(float(d.norm() / b.norm().clamp_min(1e-12)), 6) for d, b in pairs]
                recorder.rows.append({"stream": stream, "block": block,
                                      "step": hooks._current_step, "rel_per_bin": rel})
            return out

        module.forward = forward

    def restore(self) -> None:
        for module, original in self._originals:
            module.forward = original


def rollout(engine, sampler, adapters, prompt: str, seed: int, steps: int, euler: bool,
            enabled: bool, capture_steps, layers, chunk: int, record_writes: bool):
    from wsp.adapter import EulerGridScheduler, patched_schedulers
    from wsp.attention import AttentionCapture
    from wsp.attention_hooks import AttentionHooks
    from wsp.dit import CaptureSchedule

    adapters.enabled = enabled
    schedule = CaptureSchedule(layers=tuple(layers), timesteps=tuple(capture_steps),
                               keep_patches=False, streams=("video", "audio"))
    capture = AttentionCapture()
    hooks = AttentionHooks(model=engine.model, schedule=schedule, capture=capture,
                           chunk=chunk, cfg_pass=0)

    def factory(stream, s, t):
        if euler:
            return EulerGridScheduler([float(x) for x in t], [float(x) for x in s.sigmas])
        return s

    recorder = None
    started = time.time()
    try:
        with hooks:
            if record_writes:
                recorder = WriteRecorder(adapters, hooks, engine.video_latent_length)
            with patched_schedulers(engine, factory):
                result = engine.generate(text_prompt=prompt, seed=seed, sample_steps=steps,
                                         video_frame_height_width=[960, 960], **sampler)
    finally:
        hooks.detach()
        if recorder:
            recorder.restore()
    report = hooks.report()
    report["seconds"] = round(time.time() - started, 1)
    report["generate_ok"] = result is not None
    return capture, report, (recorder.rows if recorder else None)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    from pod_ovi_attention import _contrast
    from wsp.adapter import install_adapters

    engine, sampler = load_engine(args)
    adapters = install_adapters(engine.model, args.blocks, args.rank).to(next(engine.model.parameters()).device)
    adapters.load_state_dict(torch.load(args.adapter))
    every = TEACHER_STEPS // STUDENT_STEPS
    variants = {
        # label: (steps, euler, adapter enabled, capture steps)
        "teacher48": (TEACHER_STEPS, False, False, list(range(0, TEACHER_STEPS, every))),
        "base_euler12": (STUDENT_STEPS, True, False, list(range(STUDENT_STEPS))),
        "adapter_euler12": (STUDENT_STEPS, True, True, list(range(STUDENT_STEPS))),
        "base_unipc12": (STUDENT_STEPS, False, False, list(range(STUDENT_STEPS))),
    }
    payload: dict = {"adapter": str(args.adapter), "model": args.model_name, "layers": args.layers, "tags": {}}
    for tag in args.tags:
        name, seed = tag.rsplit("_", 1)
        seed = int(seed)
        captures, reports, writes = {}, {}, None
        for label, (steps, euler, enabled, cap) in variants.items():
            print(f"=== {tag} {label} ===", flush=True)
            capture, report, rows = rollout(engine, sampler, adapters, P[name], seed, steps, euler,
                                            enabled, cap, args.layers, args.chunk,
                                            record_writes=(label == "adapter_euler12"))
            captures[label], reports[label] = capture, report
            if rows is not None:
                writes = rows
            print(json.dumps({k: report[k] for k in ("fired", "recorded", "unlabelled", "seconds", "generate_ok")}),
                  flush=True)
        t_teacher = reports["teacher48"]["timesteps"]
        contrasts = {}
        for label in ("base_euler12", "adapter_euler12", "base_unipc12"):
            rows = _contrast(captures[label], captures["teacher48"], reports[label]["timesteps"],
                             t_teacher, min_effect=0.05, max_gap=1.0)
            contrasts[label] = [{k: r.get(k) for k in ("stream", "layer", "timestep_a", "timestep_gap",
                                                        "n_heads_changed", "max_effect", "skipped")}
                                for r in rows]
        payload["tags"][tag] = {
            "reports": {k: {kk: v[kk] for kk in ("fired", "recorded", "unlabelled", "other_pass_declined",
                                                  "seconds", "generate_ok", "notes")}
                        for k, v in reports.items()},
            "contrast_vs_teacher": contrasts,
            "adapter_writes": writes,
        }
        args.out.write_text(json.dumps(payload, indent=2))
    adapters.enabled = True
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
