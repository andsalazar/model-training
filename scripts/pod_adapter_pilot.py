"""Stage 7 pilot: train a TBC-style adapter so 12 Euler steps follow a 48-step teacher.

Stages (each writes to disk, so a later failure does not discard earlier GPU time):

``teacher``  48-step UniPC rollouts. Records the latent at each of the 12-step grid's
             timesteps plus the final latent, and saves the decoded mp4.
``train``    Teacher-forced: for each recorded interval, the adapted model's guided
             velocity is regressed onto the velocity that carries the teacher's state
             across that interval in one Euler step. Ovi is frozen.
``eval``     Held-out seeds, full rollouts: base UniPC-12, base Euler-12, adapter
             Euler-12, base UniPC-30. Final latents are compared with the teacher's;
             decoded videos are scored with the Stage 6 quality metrics.

See ``wsp.adapter`` for why the student uses Euler on the UniPC grid.

Validation and selection
------------------------
The held-out river seeds double as the teacher-forced validation set. The model is not
selected on them -- the final epoch is evaluated whatever its validation loss -- so they
remain held out for the rollout evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

OVI_ROOT = Path(os.environ.get("OVI_ROOT", "/local/Ovi"))
if str(OVI_ROOT) not in sys.path:
    sys.path.insert(0, str(OVI_ROOT))

RIVER = ("A river flowing through a forest, camera slowly panning. "
         "Audio: gentle water sounds and birdsong.")
TRAIN = [
    ("river", RIVER, 1),
    ("river", RIVER, 2),
    ("fan", "A ceiling fan spinning in an empty room. Audio: steady mechanical hum.", 1),
    ("coast", "A medium shot of a coastal road at golden hour, waves breaking against "
              "rocks. Audio: passing engines and gulls.", 1),
    ("fox", "A red fox trotting through fresh snow in a pine forest. "
            "Audio: wind and soft footsteps.", 1),
    ("city", "A rainy city street at night, neon reflections on wet pavement, people "
             "with umbrellas. Audio: rain, distant traffic and footsteps.", 1),
]
HELDOUT = [("river", RIVER, 100), ("river", RIVER, 200), ("river", RIVER, 300)]
TEACHER_STEPS = 48
STUDENT_STEPS = 12

#: PILOT_SMOKE=1 runs every code path in minutes: one training trajectory, one held-out
#: seed, a 12-step "teacher" (whose grid is the student grid), two training samples and
#: no 30-step baseline. Its numbers mean nothing; it only proves the plumbing.
SMOKE = os.environ.get("PILOT_SMOKE") == "1"
if SMOKE:
    TRAIN, HELDOUT, TEACHER_STEPS = TRAIN[:1], HELDOUT[:1], 12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["teacher", "train", "eval"])
    p.add_argument("--config", default=str(OVI_ROOT / "ovi/configs/inference/inference_fusion_wsp.yaml"))
    p.add_argument("--work", type=Path, default=Path("/local/pilot5s"))
    p.add_argument("--blocks", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--audio-weight", type=float, default=1.0)
    p.add_argument("--adapter", type=Path, default=None, help="adapter state for eval")
    return p.parse_args(argv)


def tag(name: str, seed: int) -> str:
    return f"{name}_s{seed}"


def load_engine(args):
    from omegaconf import OmegaConf
    from ovi.ovi_fusion_engine import OviFusionEngine

    from wsp.dit import ovi_sampler_kwargs

    config = OmegaConf.load(args.config)
    engine = OviFusionEngine(config=config, device=0)
    return engine, ovi_sampler_kwargs(config)


def student_grid(engine) -> tuple[list[float], list[float]]:
    sched, ts = engine.get_scheduler_time_steps(
        sampling_steps=STUDENT_STEPS, solver_name="unipc", device="cpu", shift=5.0
    )
    return [float(t) for t in ts], [float(s) for s in sched.sigmas]


def save_mp4(path: Path, video, audio) -> None:
    from ovi.utils.io_utils import save_video

    save_video(str(path), video, audio, fps=24, sample_rate=16000)


# --------------------------------------------------------------------------- teacher


def stage_teacher(args) -> int:
    import torch

    from wsp.adapter import RecordingScheduler, patched_schedulers

    engine, sampler = load_engine(args)
    grid_t, grid_s = student_grid(engine)
    out_dir = args.work / "teacher"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for name, prompt, seed in TRAIN + HELDOUT:
        path = out_dir / f"{tag(name, seed)}.pt"
        if path.exists():
            print(f"skip {path.name} (exists)", flush=True)
            continue
        started = time.time()
        with patched_schedulers(
            engine, lambda stream, s, t: RecordingScheduler(inner=s, record=grid_t, n_steps=TEACHER_STEPS)
        ) as made:
            result = engine.generate(
                text_prompt=prompt, seed=seed, sample_steps=TEACHER_STEPS,
                video_frame_height_width=[960, 960], **sampler,
            )
        if result is None:
            raise RuntimeError(f"generate failed for {tag(name, seed)}; see Ovi's traceback")
        video, audio, _ = result
        vrec, arec = made[0], made[1]
        for rec, stream in ((vrec, "video"), (arec, "audio")):
            missing = [t for t in grid_t if t not in rec.samples]
            if missing or rec.final is None:
                raise RuntimeError(f"{stream} recording incomplete: missing {missing}, "
                                   f"final={'set' if rec.final is not None else 'none'}")
        torch.save(
            {
                "name": name, "prompt": prompt, "seed": seed,
                "timesteps": grid_t, "sigmas": grid_s,
                "video": torch.stack([vrec.samples[t] for t in grid_t] + [vrec.final]),
                "audio": torch.stack([arec.samples[t] for t in grid_t] + [arec.final]),
            },
            path,
        )
        save_mp4(out_dir / f"{tag(name, seed)}.mp4", video, audio)
        seconds = round(time.time() - started, 1)
        log.append({"tag": tag(name, seed), "seconds": seconds})
        print(f"teacher {tag(name, seed)}: {seconds}s", flush=True)
    (args.work / "teacher_log.json").write_text(json.dumps(log, indent=2))
    return 0


# --------------------------------------------------------------------------- train


class Trainer:
    def __init__(self, engine, sampler, args):
        import torch

        from wsp.adapter import install_adapters

        self.torch = torch
        self.engine = engine
        self.sampler = sampler
        self.model = engine.model
        self.device = next(self.model.parameters()).device
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.adapters = install_adapters(self.model, args.blocks, args.rank).to(self.device)
        self.audio_weight = args.audio_weight
        self._emb: dict[str, list] = {}

    def embeddings(self, prompt: str):
        torch = self.torch
        if prompt not in self._emb:
            with torch.no_grad():
                emb = self.engine.text_model(
                    [prompt, self.sampler["video_negative_prompt"], self.sampler["audio_negative_prompt"]],
                    self.engine.text_model.device,
                )
            self._emb[prompt] = [e.to(torch.bfloat16).to(self.device) for e in emb]
        return self._emb[prompt]

    def losses(self, traj: dict, k: int):
        """Normalised video and audio losses of the guided velocity on interval k."""
        torch = self.torch
        from wsp.adapter import normalised_mse, target_velocity

        pos, vneg, aneg = self.embeddings(traj["prompt"])
        xv = traj["video"][k].to(self.device, torch.bfloat16)
        xa = traj["audio"][k].to(self.device, torch.bfloat16)
        patch = self.model.video_model.patch_size
        vid_seq = xv.shape[1] * xv.shape[2] * xv.shape[3] // (patch[1] * patch[2])
        aud_seq = xa.shape[0]
        t = torch.full((1,), traj["timesteps"][k], device=self.device)
        common = {"vid_seq_len": vid_seq, "audio_seq_len": aud_seq, "first_frame_is_clean": False}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pv, pa = self.model(vid=[xv], audio=[xa], t=t, vid_context=[pos], audio_context=[pos], **common)
            nv, na = self.model(vid=[xv], audio=[xa], t=t, vid_context=[vneg], audio_context=[aneg],
                                slg_layer=self.sampler["slg_layer"], **common)
        gv = nv[0].float() + self.sampler["video_guidance_scale"] * (pv[0].float() - nv[0].float())
        ga = na[0].float() + self.sampler["audio_guidance_scale"] * (pa[0].float() - na[0].float())
        sk, sn = traj["sigmas"][k], traj["sigmas"][k + 1]
        tv = target_velocity(traj["video"][k], traj["video"][k + 1], sk, sn).to(self.device)
        ta = target_velocity(traj["audio"][k], traj["audio"][k + 1], sk, sn).to(self.device)
        return normalised_mse(gv, tv), normalised_mse(ga, ta)

    def validate(self, trajs: list[dict]) -> dict:
        torch = self.torch
        self.model.train(False)
        per_k_v = [[] for _ in range(STUDENT_STEPS)]
        per_k_a = [[] for _ in range(STUDENT_STEPS)]
        with torch.no_grad():
            for traj in trajs:
                for k in range(STUDENT_STEPS):
                    lv, la = self.losses(traj, k)
                    per_k_v[k].append(float(lv))
                    per_k_a[k].append(float(la))
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        return {
            "video": mean([mean(x) for x in per_k_v]),
            "audio": mean([mean(x) for x in per_k_a]),
            "video_per_k": [round(mean(x), 5) for x in per_k_v],
            "audio_per_k": [round(mean(x), 5) for x in per_k_a],
        }


def stage_train(args) -> int:
    import torch

    engine, sampler = load_engine(args)
    trainer = Trainer(engine, sampler, args)
    tdir = args.work / "teacher"
    train = [torch.load(tdir / f"{tag(n, s)}.pt") for n, _, s in TRAIN]
    val = [torch.load(tdir / f"{tag(n, s)}.pt") for n, _, s in HELDOUT]
    params = list(trainer.adapters.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    samples = [(i, k) for i in range(len(train)) for k in range(STUDENT_STEPS)]
    rng = random.Random(0)
    if SMOKE:
        samples = samples[:2]

    history: dict = {"adapter": trainer.adapters.describe(), "lr": args.lr,
                     "audio_weight": args.audio_weight, "epochs": []}
    started = time.time()
    base = trainer.validate(val)
    history["val_base"] = base
    print(f"val (base == untrained adapter): {json.dumps(base)}", flush=True)
    out = args.work / "train"
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        rng.shuffle(samples)
        trainer.model.train(True)
        trainer.model.gradient_checkpointing = True
        steps = []
        for n, (i, k) in enumerate(samples):
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            lv, la = trainer.losses(train[i], k)
            loss = lv + args.audio_weight * la
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            steps.append({"traj": i, "k": k, "video": round(float(lv), 5), "audio": round(float(la), 5),
                          "grad_norm": round(gnorm, 4), "seconds": round(time.time() - t0, 1)})
            if n % 12 == 0:
                print(f"epoch {epoch} step {n}/{len(samples)} {steps[-1]}", flush=True)
        trainer.model.gradient_checkpointing = False
        val_now = trainer.validate(val)
        history["epochs"].append({"epoch": epoch, "steps": steps, "val": val_now})
        torch.save(trainer.adapters.state_dict(), out / f"adapter_epoch{epoch}.pt")
        print(f"epoch {epoch} val {json.dumps(val_now)}", flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2))
    torch.save(trainer.adapters.state_dict(), out / "adapter_final.pt")
    history["seconds"] = round(time.time() - started, 1)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    return 0


# --------------------------------------------------------------------------- eval


def _quality(mp4: Path) -> dict:
    import imageio.v3 as iio
    import numpy as np

    from wsp.quality import measure_quality

    frames = np.asarray(iio.imread(str(mp4), plugin="FFMPEG"))[..., :3]
    idx = np.linspace(0, len(frames) - 1, 48).astype(int)
    return measure_quality(frames[idx]).as_dict()


def stage_eval(args) -> int:
    import torch

    from wsp.adapter import EulerGridScheduler, RecordingScheduler, install_adapters, patched_schedulers

    engine, sampler = load_engine(args)
    adapters = install_adapters(engine.model, args.blocks, args.rank).to(
        next(engine.model.parameters()).device
    )
    state = args.adapter or args.work / "train" / "adapter_final.pt"
    adapters.load_state_dict(torch.load(state))
    out = args.work / "eval"
    out.mkdir(parents=True, exist_ok=True)

    def euler(stream, s, t):
        inner = EulerGridScheduler([float(x) for x in t], [float(x) for x in s.sigmas])
        return RecordingScheduler(inner=inner, record=[], n_steps=len(t))

    def unipc(n):
        return lambda stream, s, t: RecordingScheduler(inner=s, record=[], n_steps=n)

    variants = [
        ("base_unipc12", False, STUDENT_STEPS, unipc(STUDENT_STEPS)),
        ("base_euler12", False, STUDENT_STEPS, euler),
        ("adapter_euler12", True, STUDENT_STEPS, euler),
        ("base_unipc30", False, 30, unipc(30)),
    ]
    if SMOKE:
        variants = variants[:3]
    rows = []
    for name, prompt, seed in HELDOUT:
        teacher = torch.load(args.work / "teacher" / f"{tag(name, seed)}.pt")
        for label, enabled, steps, factory in variants:
            adapters.enabled = enabled
            t0 = time.time()
            with patched_schedulers(engine, factory) as made:
                result = engine.generate(
                    text_prompt=prompt, seed=seed, sample_steps=steps,
                    video_frame_height_width=[960, 960], **sampler,
                )
            seconds = round(time.time() - t0, 1)
            if result is None:
                rows.append({"seed": seed, "variant": label, "error": "generate failed"})
                continue
            video, audio, _ = result
            mp4 = out / f"{label}_s{seed}.mp4"
            save_mp4(mp4, video, audio)
            row = {"seed": seed, "variant": label, "steps": steps, "seconds": seconds}
            for rec, stream in ((made[0], "video"), (made[1], "audio")):
                ref = teacher[stream][-1].float()
                got = rec.final.float()
                row[f"{stream}_final_rel_err"] = round(float((got - ref).norm() / ref.norm()), 5)
            row["quality"] = _quality(mp4)
            rows.append(row)
            print(json.dumps(row), flush=True)
            (out / "eval.json").write_text(json.dumps(rows, indent=2))
    adapters.enabled = True
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    return {"teacher": stage_teacher, "train": stage_train, "eval": stage_eval}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
