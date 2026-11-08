"""Stage 7 v2: scaled-up adapter training at 5 s, then 10 s. Successor to the pilot.

Same method as ``pod_adapter_pilot.py`` (whose defaults reproduce the pilot); this adds
what scaling needs:

* **Named data sets** with a separate *validation* set, so the best epoch can be
  selected without touching the evaluation seeds.
* **Cosine learning-rate decay**, and ``--init-adapter`` to start from an earlier
  adapter (the 10 s run starts from the 5 s one).
* ``--cfg-grad sequential``: backpropagate the conditional and unconditional passes one
  at a time (see ``wsp.adapter.normalised_mse_grad``). Needed at 10 s, where one pass
  peaks at 67 GB.
* ``--offload-text``: move T5 to the CPU during training, after every prompt embedding
  has been computed (~11 GB back).
* Eval variants are selectable, and the teacher's own video is scored with the same
  tool, so the comparison never mixes tools.

Stages: ``teacher`` -> ``train`` -> ``eval``. Each writes to disk and skips what exists.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

OVI_ROOT = Path(os.environ.get("OVI_ROOT", "/local/Ovi"))
if str(OVI_ROOT) not in sys.path:
    sys.path.insert(0, str(OVI_ROOT))

P = {
    "river": "A river flowing through a forest, camera slowly panning. Audio: gentle water sounds and birdsong.",
    "fan": "A ceiling fan spinning in an empty room. Audio: steady mechanical hum.",
    "coast": "A medium shot of a coastal road at golden hour, waves breaking against rocks. Audio: passing engines and gulls.",
    "fox": "A red fox trotting through fresh snow in a pine forest. Audio: wind and soft footsteps.",
    "city": "A rainy city street at night, neon reflections on wet pavement, people with umbrellas. Audio: rain, distant traffic and footsteps.",
    "waterfall": "A tall waterfall in a misty canyon, camera tilting up slowly. Audio: roaring water and wind.",
    "cafe": "A busy cafe interior, a barista pouring latte art. Audio: espresso machine hiss, cups clinking and soft chatter.",
    "drummer": "A drummer playing a jazz beat on a small stage under warm lights. Audio: brushed snare and ride cymbal.",
    "beach": "Waves rolling onto a sandy beach at sunset, seagulls flying overhead. Audio: surf and gull calls.",
    "train": "A steam train crossing a stone viaduct through green hills. Audio: chugging engine and a long whistle.",
    "market": "A crowded outdoor market at noon, vendors arranging fruit. Audio: haggling voices and rustling bags.",
    "rainforest": "Rain falling on leaves in a dense forest, close-up of dripping ferns. Audio: steady rain and distant thunder.",
    "violin": "A street musician playing violin in a sunny park. Audio: violin melody and distant chatter.",
}

#: The pilot's six trajectories come first so their teacher files are reused.
_PILOT = [("river", 1), ("river", 2), ("fan", 1), ("coast", 1), ("fox", 1), ("city", 1)]
SETS = {
    "v2_5s": {
        "train": _PILOT + [("river", 3), ("fan", 2), ("coast", 2), ("fox", 2), ("city", 2),
                           ("waterfall", 1), ("cafe", 1), ("drummer", 1), ("beach", 1), ("train", 1)],
        "val": [("market", 1), ("rainforest", 1)],
        "eval": [("river", 100), ("river", 200), ("river", 300), ("violin", 100)],
    },
    # A minutes-long plumbing check on the pilot's existing teacher files.
    "check": {"train": [("river", 1)], "val": [("river", 100)], "eval": [("river", 100)]},
    "v2_10s": {
        "train": _PILOT + [("waterfall", 1), ("drummer", 1)],
        "val": [("market", 1), ("rainforest", 1)],
        "eval": [("river", 100), ("river", 200), ("river", 300)],
    },
}
TEACHER_STEPS = 48
STUDENT_STEPS = 12

VARIANTS = {
    # label: (adapter enabled, steps, sampler)
    "base_unipc12": (False, STUDENT_STEPS, "unipc"),
    "base_euler12": (False, STUDENT_STEPS, "euler"),
    "adapter_euler12": (True, STUDENT_STEPS, "euler"),
    "base_unipc30": (False, 30, "unipc"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["teacher", "train", "eval"])
    p.add_argument("--set", dest="data_set", choices=sorted(SETS), required=True)
    p.add_argument("--config", default=str(OVI_ROOT / "ovi/configs/inference/inference_fusion_wsp.yaml"))
    p.add_argument("--model-name", default=None, help="override the config's model_name")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--teacher-dir", type=Path, default=None, help="default: WORK/teacher")
    p.add_argument("--blocks", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-min", type=float, default=1e-4, help="cosine floor; = --lr for constant")
    p.add_argument("--audio-weight", type=float, default=1.0)
    p.add_argument("--cfg-grad", choices=["joint", "sequential"], default="joint")
    p.add_argument("--offload-text", action="store_true")
    p.add_argument("--init-adapter", type=Path, default=None)
    p.add_argument("--adapter", type=Path, default=None, help="eval: adapter state (default best)")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--eval-dir", type=Path, default=None, help="default: WORK/eval")
    p.add_argument("--max-samples", type=int, default=None, help="cap samples per epoch (checks)")
    return p.parse_args(argv)


def tag(name: str, seed: int) -> str:
    return f"{name}_s{seed}"


def load_engine(args):
    from omegaconf import OmegaConf
    from ovi.ovi_fusion_engine import OviFusionEngine

    from wsp.dit import ovi_sampler_kwargs

    config = OmegaConf.load(args.config)
    if args.model_name:
        config.model_name = args.model_name
    engine = OviFusionEngine(config=config, device=0)
    return engine, ovi_sampler_kwargs(config)


def teacher_dir(args) -> Path:
    return args.teacher_dir or args.work / "teacher"


def student_grid(engine) -> tuple[list[float], list[float]]:
    sched, ts = engine.get_scheduler_time_steps(
        sampling_steps=STUDENT_STEPS, solver_name="unipc", device="cpu", shift=5.0
    )
    return [float(t) for t in ts], [float(s) for s in sched.sigmas]


def save_mp4(path: Path, video, audio) -> None:
    from ovi.utils.io_utils import save_video

    save_video(str(path), video, audio, fps=24, sample_rate=16000)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


# --------------------------------------------------------------------------- teacher


def stage_teacher(args) -> int:
    import torch

    from wsp.adapter import RecordingScheduler, patched_schedulers

    engine, sampler = load_engine(args)
    grid_t, grid_s = student_grid(engine)
    out_dir = teacher_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = SETS[args.data_set]
    log_path = args.work / "teacher_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    for name, seed in spec["train"] + spec["val"] + spec["eval"]:
        path = out_dir / f"{tag(name, seed)}.pt"
        if path.exists():
            print(f"skip {path.name} (exists)", flush=True)
            continue
        started = time.time()
        with patched_schedulers(
            engine, lambda stream, s, t: RecordingScheduler(inner=s, record=grid_t, n_steps=TEACHER_STEPS)
        ) as made:
            result = engine.generate(
                text_prompt=P[name], seed=seed, sample_steps=TEACHER_STEPS,
                video_frame_height_width=[960, 960], **sampler,
            )
        if result is None:
            raise RuntimeError(f"generate failed for {tag(name, seed)}; see Ovi's traceback")
        video, audio, _ = result
        for rec, stream in ((made[0], "video"), (made[1], "audio")):
            missing = [t for t in grid_t if t not in rec.samples]
            if missing or rec.final is None:
                raise RuntimeError(f"{stream} recording incomplete: missing {missing}")
        torch.save(
            {
                "name": name, "prompt": P[name], "seed": seed,
                "timesteps": grid_t, "sigmas": grid_s,
                "video": torch.stack([made[0].samples[t] for t in grid_t] + [made[0].final]),
                "audio": torch.stack([made[1].samples[t] for t in grid_t] + [made[1].final]),
            },
            path,
        )
        save_mp4(out_dir / f"{tag(name, seed)}.mp4", video, audio)
        seconds = round(time.time() - started, 1)
        log.append({"tag": tag(name, seed), "seconds": seconds})
        write_json(log_path, log)
        print(f"teacher {tag(name, seed)}: {seconds}s", flush=True)
    return 0


# --------------------------------------------------------------------------- train


class Trainer:
    def __init__(self, engine, sampler, args, prompts: list[str]):
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
        if args.init_adapter:
            self.adapters.load_state_dict(torch.load(args.init_adapter))
        self.audio_weight = args.audio_weight
        self.sequential = args.cfg_grad == "sequential"
        self._emb: dict[str, list] = {}
        for prompt in prompts:
            self._embed(prompt)
        if args.offload_text:
            engine.text_model.model.to("cpu")
            torch.cuda.empty_cache()

    def _embed(self, prompt: str):
        torch = self.torch
        if prompt not in self._emb:
            with torch.no_grad():
                emb = self.engine.text_model(
                    [prompt, self.sampler["video_negative_prompt"], self.sampler["audio_negative_prompt"]],
                    self.engine.text_model.device,
                )
            self._emb[prompt] = [e.to(torch.bfloat16).to(self.device) for e in emb]
        return self._emb[prompt]

    def _inputs(self, traj: dict, k: int):
        torch = self.torch
        from wsp.adapter import target_velocity

        xv = traj["video"][k].to(self.device, torch.bfloat16)
        xa = traj["audio"][k].to(self.device, torch.bfloat16)
        patch = self.model.video_model.patch_size
        common = {
            "vid_seq_len": xv.shape[1] * xv.shape[2] * xv.shape[3] // (patch[1] * patch[2]),
            "audio_seq_len": xa.shape[0],
            "first_frame_is_clean": False,
        }
        t = torch.full((1,), traj["timesteps"][k], device=self.device)
        sk, sn = traj["sigmas"][k], traj["sigmas"][k + 1]
        tv = target_velocity(traj["video"][k], traj["video"][k + 1], sk, sn).to(self.device)
        ta = target_velocity(traj["audio"][k], traj["audio"][k + 1], sk, sn).to(self.device)
        return xv, xa, t, common, tv, ta

    def _pass(self, traj, xv, xa, t, common, positive: bool):
        torch = self.torch
        pos, vneg, aneg = self._emb[traj["prompt"]]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if positive:
                v, a = self.model(vid=[xv], audio=[xa], t=t, vid_context=[pos], audio_context=[pos], **common)
            else:
                v, a = self.model(vid=[xv], audio=[xa], t=t, vid_context=[vneg], audio_context=[aneg],
                                  slg_layer=self.sampler["slg_layer"], **common)
        return v[0].float(), a[0].float()

    def _guided(self, traj, xv, xa, t, common):
        pv, pa = self._pass(traj, xv, xa, t, common, True)
        nv, na = self._pass(traj, xv, xa, t, common, False)
        sv, sa = self.sampler["video_guidance_scale"], self.sampler["audio_guidance_scale"]
        return nv + sv * (pv - nv), na + sa * (pa - na)

    def losses(self, traj: dict, k: int):
        from wsp.adapter import normalised_mse

        xv, xa, t, common, tv, ta = self._inputs(traj, k)
        gv, ga = self._guided(traj, xv, xa, t, common)
        return normalised_mse(gv, tv), normalised_mse(ga, ta)

    def train_step(self, traj: dict, k: int) -> tuple[float, float]:
        """Accumulate gradients for one interval; returns the (video, audio) losses."""
        torch = self.torch
        from wsp.adapter import normalised_mse_grad

        if not self.sequential:
            lv, la = self.losses(traj, k)
            (lv + self.audio_weight * la).backward()
            return float(lv), float(la)

        xv, xa, t, common, tv, ta = self._inputs(traj, k)
        with torch.no_grad():
            gv, ga = self._guided(traj, xv, xa, t, common)
            lv, dv = normalised_mse_grad(gv, tv)
            la, da = normalised_mse_grad(ga, ta)
        da = da * self.audio_weight
        sv, sa = self.sampler["video_guidance_scale"], self.sampler["audio_guidance_scale"]
        for positive, wv, wa in ((True, sv, sa), (False, 1.0 - sv, 1.0 - sa)):
            v, a = self._pass(traj, xv, xa, t, common, positive)
            ((v * dv).sum() * wv + (a * da).sum() * wa).backward()
            del v, a
        return float(lv), float(la)

    def validate(self, trajs: list[dict]) -> dict:
        torch = self.torch
        self.model.train(False)
        self.model.gradient_checkpointing = False
        per_k_v = [[] for _ in range(STUDENT_STEPS)]
        per_k_a = [[] for _ in range(STUDENT_STEPS)]
        with torch.no_grad():
            for traj in trajs:
                for k in range(STUDENT_STEPS):
                    lv, la = self.losses(traj, k)
                    per_k_v[k].append(float(lv))
                    per_k_a[k].append(float(la))
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        video = mean([mean(x) for x in per_k_v])
        audio = mean([mean(x) for x in per_k_a])
        return {
            "video": video, "audio": audio, "total": video + self.audio_weight * audio,
            "video_per_k": [round(mean(x), 5) for x in per_k_v],
            "audio_per_k": [round(mean(x), 5) for x in per_k_a],
        }


def stage_train(args) -> int:
    import torch

    spec = SETS[args.data_set]
    engine, sampler = load_engine(args)
    tdir = teacher_dir(args)
    load = lambda items: [torch.load(tdir / f"{tag(n, s)}.pt") for n, s in items]  # noqa: E731
    train, val = load(spec["train"]), load(spec["val"])
    prompts = sorted({P[n] for n, _ in spec["train"] + spec["val"]})
    trainer = Trainer(engine, sampler, args, prompts)
    params = list(trainer.adapters.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    samples = [(i, k) for i in range(len(train)) for k in range(STUDENT_STEPS)]
    if args.max_samples:
        samples = samples[: args.max_samples]
    total = args.epochs * len(samples)

    def lr_at(step: int) -> float:
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * step / max(total, 1)))

    out = args.work / "train"
    out.mkdir(parents=True, exist_ok=True)
    history: dict = {
        "set": args.data_set, "adapter": trainer.adapters.describe(), "lr": args.lr,
        "lr_min": args.lr_min, "audio_weight": args.audio_weight, "cfg_grad": args.cfg_grad,
        "init_adapter": str(args.init_adapter) if args.init_adapter else None,
        "n_train": len(train), "n_val": len(val), "epochs": [],
    }
    started = time.time()
    best = trainer.validate(val)
    history["val_start"] = best
    torch.save(trainer.adapters.state_dict(), out / "adapter_best.pt")
    history["best_epoch"] = -1
    print(f"val start: {json.dumps(best)}", flush=True)

    rng = random.Random(0)
    step = 0
    for epoch in range(args.epochs):
        rng.shuffle(samples)
        trainer.model.train(True)
        trainer.model.gradient_checkpointing = True
        steps = []
        for n, (i, k) in enumerate(samples):
            for group in opt.param_groups:
                group["lr"] = lr_at(step)
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            lv, la = trainer.train_step(train[i], k)
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            step += 1
            steps.append({"traj": i, "k": k, "video": round(lv, 5), "audio": round(la, 5),
                          "grad_norm": round(gnorm, 4), "lr": round(lr_at(step - 1), 7),
                          "seconds": round(time.time() - t0, 1)})
            if n % 24 == 0:
                print(f"epoch {epoch} step {n}/{len(samples)} {steps[-1]}", flush=True)
        val_now = trainer.validate(val)
        history["epochs"].append({"epoch": epoch, "steps": steps, "val": val_now})
        torch.save(trainer.adapters.state_dict(), out / f"adapter_epoch{epoch}.pt")
        if val_now["total"] < best["total"]:
            best = val_now
            history["best_epoch"] = epoch
            torch.save(trainer.adapters.state_dict(), out / "adapter_best.pt")
        print(f"epoch {epoch} val {json.dumps(val_now)} best_epoch {history['best_epoch']}", flush=True)
        history["seconds"] = round(time.time() - started, 1)
        write_json(out / "history.json", history)
    return 0


# --------------------------------------------------------------------------- eval


def _quality(mp4: Path) -> dict:
    import imageio.v3 as iio
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pod_ovi_colour_check import probe_frames
    from wsp.quality import measure_quality

    frames = np.asarray(iio.imread(str(mp4), plugin="FFMPEG"))[..., :3]
    idx = np.linspace(0, len(frames) - 1, 48).astype(int)
    q = measure_quality(frames[idx]).as_dict()
    colour = probe_frames(frames, mp4.name)
    q.update({"colour_r2": colour["mean_r2"], "colour_drift": colour["mean_abs_drift"]})
    return q


def _audio(audio) -> dict:
    import numpy as np

    audio = np.asarray(audio, dtype=np.float64).ravel()
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(audio.size, 1.0 / 16000)
    return {"rms": round(float(np.sqrt(np.mean(audio ** 2))), 6),
            "centroid_hz": round(float((spectrum * freqs).sum() / max(spectrum.sum(), 1e-12)), 1)}


def stage_eval(args) -> int:
    import torch

    from wsp.adapter import EulerGridScheduler, RecordingScheduler, install_adapters, patched_schedulers

    spec = SETS[args.data_set]
    engine, sampler = load_engine(args)
    adapters = install_adapters(engine.model, args.blocks, args.rank).to(
        next(engine.model.parameters()).device
    )
    state = args.adapter or args.work / "train" / "adapter_best.pt"
    if "adapter_euler12" in args.variants:
        adapters.load_state_dict(torch.load(state))
    out = args.eval_dir or args.work / "eval"
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "eval.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {(r["tag"], r["variant"]) for r in rows}

    def factory(kind: str, n: int):
        if kind == "euler":
            def euler(stream, s, t):
                inner = EulerGridScheduler([float(x) for x in t], [float(x) for x in s.sigmas])
                return RecordingScheduler(inner=inner, record=[], n_steps=len(t))
            return euler
        return lambda stream, s, t: RecordingScheduler(inner=s, record=[], n_steps=n)

    for name, seed in spec["eval"]:
        tg = tag(name, seed)
        teacher = torch.load(teacher_dir(args) / f"{tg}.pt")
        if (tg, "teacher48") not in done:
            rows.append({"tag": tg, "variant": "teacher48", "steps": TEACHER_STEPS,
                         "quality": _quality(teacher_dir(args) / f"{tg}.mp4")})
            write_json(rows_path, rows)
        for label in args.variants:
            if (tg, label) in done:
                continue
            enabled, steps, kind = VARIANTS[label]
            adapters.enabled = enabled
            t0 = time.time()
            with patched_schedulers(engine, factory(kind, steps)) as made:
                result = engine.generate(
                    text_prompt=P[name], seed=seed, sample_steps=steps,
                    video_frame_height_width=[960, 960], **sampler,
                )
            seconds = round(time.time() - t0, 1)
            if result is None:
                rows.append({"tag": tg, "variant": label, "error": "generate failed"})
                write_json(rows_path, rows)
                continue
            video, audio, _ = result
            mp4 = out / f"{label}_{tg}.mp4"
            save_mp4(mp4, video, audio)
            row = {"tag": tg, "variant": label, "steps": steps, "seconds": seconds,
                   "adapter_state": str(state) if enabled else None, "audio": _audio(audio)}
            for rec, stream in ((made[0], "video"), (made[1], "audio")):
                ref = teacher[stream][-1].float()
                row[f"{stream}_final_rel_err"] = round(float((rec.final.float() - ref).norm() / ref.norm()), 5)
            row["quality"] = _quality(mp4)
            rows.append(row)
            write_json(rows_path, rows)
            print(json.dumps(row), flush=True)
    adapters.enabled = True
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    return {"teacher": stage_teacher, "train": stage_train, "eval": stage_eval}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
