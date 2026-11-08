"""Generate past Ovi's trained length and measure whether it degrades over time.

Ovi ships 5 s and 10 s checkpoints only. This runs the 10 s model with a longer latent
(``6 * seconds + 1`` video latent frames, ``31.4 * seconds`` audio tokens), which is
outside anything it was trained on. Long-horizon degradation is what TBC's Oasis
research targeted, so whether and where Ovi degrades decides whether a stability adapter
has a job at 30 s.

What is measured, per 10 s window and for the whole clip: the Stage 6 video metrics,
the colour trend test, the per-frame delta-entropy collapse trend, and audio RMS and
spectral centroid. Windowed numbers are the point: a clip whose last third is worse than
its first is degrading, whatever its average.

The final latents are saved before Ovi decodes them, so a VAE out-of-memory at decode
does not lose the rollout.
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
if str(OVI_ROOT) not in sys.path:
    sys.path.insert(0, str(OVI_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RIVER = ("A river flowing through a forest, camera slowly panning. "
         "Audio: gentle water sounds and birdsong.")
FPS = 24
SAMPLE_RATE = 16000
#: Audio latent tokens per second: 157 for 5 s and 314 for 10 s in Ovi's model specs.
AUDIO_TOKENS_PER_SECOND = 31.4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(OVI_ROOT / "ovi/configs/inference/inference_fusion_wsp.yaml"))
    p.add_argument("--model-name", default="960x960_10s")
    p.add_argument("--seconds", type=int, default=30)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--prompt", default=RIVER)
    p.add_argument("--window", type=int, default=10, help="seconds per analysis window")
    p.add_argument("--out-dir", type=Path, default=Path("/local/long30"))
    return p.parse_args(argv)


def latent_lengths(seconds: int) -> tuple[int, int]:
    """Video latent frames and audio tokens for a clip of ``seconds``.

    Video: ``24 * s + 1`` frames, 4x temporal compression -> ``6 * s + 1`` latents
    (5 s -> 31, 10 s -> 61, matching Ovi's specs). Audio: 31.4 tokens per second
    (5 s -> 157, 10 s -> 314).
    """
    return 6 * seconds + 1, int(round(AUDIO_TOKENS_PER_SECOND * seconds))


def _audio_stats(audio: np.ndarray) -> dict:
    audio = np.asarray(audio, dtype=np.float64).ravel()
    rms = float(np.sqrt(np.mean(audio ** 2)))
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(audio.size, 1.0 / SAMPLE_RATE)
    centroid = float((spectrum * freqs).sum() / max(spectrum.sum(), 1e-12))
    return {"rms": round(rms, 5), "centroid_hz": round(centroid, 1)}


def _delta_entropy(frames: np.ndarray) -> dict:
    """Collapse trend of the delta entropy, on pixel patches, every 4th frame."""
    from pod_ovi_metric import _patches_from_pixels
    from wsp.entropy import frame_delta_entropy

    sub = frames[::4]
    patches = _patches_from_pixels(sub, patch=16, max_patches=200)
    reading = frame_delta_entropy(list(patches), estimator="knn")
    return {
        "slope": reading.slope, "r_squared": reading.r_squared,
        "early_mean": reading.early_mean, "late_mean": reading.late_mean,
        "collapsing": reading.collapsing,
    }


def analyse(frames: np.ndarray, audio: np.ndarray, window: int) -> dict:
    from pod_ovi_colour_check import probe_frames
    from wsp.quality import measure_quality

    def video_metrics(chunk: np.ndarray) -> dict:
        idx = np.linspace(0, len(chunk) - 1, min(48, len(chunk))).astype(int)
        quality = measure_quality(chunk[idx]).as_dict()
        colour = probe_frames(chunk, "chunk")
        quality.update({"colour_r2": colour["mean_r2"], "colour_drift": colour["mean_abs_drift"]})
        return quality

    out: dict = {"whole": video_metrics(frames), "windows": []}
    out["whole"]["delta_entropy"] = _delta_entropy(frames)
    out["whole"]["audio"] = _audio_stats(audio)
    per_v = window * FPS
    per_a = window * SAMPLE_RATE
    n = max(1, int(np.ceil(len(frames) / per_v)))
    for w in range(n):
        chunk = frames[w * per_v:(w + 1) * per_v + 1]
        if len(chunk) < 8:
            continue
        row = {"window": w, "seconds": [w * window, (w + 1) * window]}
        row.update(video_metrics(chunk))
        row["audio"] = _audio_stats(audio[w * per_a:(w + 1) * per_a])
        out["windows"].append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch
    from omegaconf import OmegaConf
    from ovi.ovi_fusion_engine import OviFusionEngine
    from ovi.utils.io_utils import save_video

    from wsp.adapter import RecordingScheduler, patched_schedulers
    from wsp.dit import ovi_sampler_kwargs

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.load(args.config)
    config.model_name = args.model_name
    sampler = ovi_sampler_kwargs(config)
    engine = OviFusionEngine(config=config, device=0)
    native = (engine.video_latent_length, engine.audio_latent_length)
    video_len, audio_len = latent_lengths(args.seconds)
    engine.video_latent_length, engine.audio_latent_length = video_len, audio_len
    tag = f"{args.seconds}s_{args.steps}steps_s{args.seed}"
    report: dict = {
        "model": args.model_name, "native_latents": native,
        "latents": [video_len, audio_len], "seconds": args.seconds,
        "steps": args.steps, "seed": args.seed, "sampler": sampler,
    }
    print(json.dumps(report), flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with patched_schedulers(
        engine, lambda stream, s, t: RecordingScheduler(inner=s, record=[], n_steps=args.steps)
    ) as made:
        result = engine.generate(
            text_prompt=args.prompt, seed=args.seed, sample_steps=args.steps,
            video_frame_height_width=[960, 960], **sampler,
        )
    report["generate_seconds"] = round(time.time() - t0, 1)
    report["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if made and made[0].final is not None:
        torch.save({"video": made[0].final, "audio": made[1].final}, args.out_dir / f"{tag}_latents.pt")
        report["latents_saved"] = True
    if result is None:
        report["error"] = "generate returned None (decode or sampling failed; see Ovi traceback)"
        (args.out_dir / f"{tag}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return 1

    video, audio, _ = result
    mp4 = args.out_dir / f"{tag}.mp4"
    save_video(str(mp4), video, audio, fps=FPS, sample_rate=SAMPLE_RATE)
    frames = np.ascontiguousarray(
        np.clip((np.moveaxis(video, 0, -1) + 1.0) * 127.5, 0, 255).astype(np.uint8)
    )
    del video
    report["frames"] = int(frames.shape[0])
    report["audio_samples"] = int(np.asarray(audio).size)
    report["analysis"] = analyse(frames, np.asarray(audio), args.window)
    (args.out_dir / f"{tag}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "analysis"}), flush=True)
    for row in report["analysis"]["windows"]:
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
