"""How much real detail does a 960x960 Ovi video carry, and does the MP4 encode lose it?

Three readings per video, on 12 evenly spaced grayscale frames:

* ``psnr_down{2,4}``: PSNR between a frame and the same frame downscaled 2x (or 4x)
  and upscaled back. Real 960-line detail is destroyed by the round trip, so a *high*
  value means there was little detail at that scale to destroy.
* ``hf_fraction``: share of spectral power above 0.25 cycles/pixel, i.e. finer than a
  480x480 frame can represent.
* ``effective_px``: the radial frequency below which 99% of the (non-DC) spectral power
  lies, as the frame width that frequency would be the Nyquist limit of
  (``width * f99 / 0.5``). A frame upscaled from 240 px has almost no power above
  0.125 cycles/pixel and reads as ~240.

``--latents`` additionally decodes saved final latents with Ovi's own VAE, so the raw
frames can be compared with the MP4 that was written from the same latents. That
separates what Ovi produces from what the encoder (Ovi's ``save_video``: libx264 at
ffmpeg's default quality, CRF 23) takes away.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

OVI_ROOT = Path(os.environ.get("OVI_ROOT", "/local/Ovi"))


def _gray(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    return frames[..., 0] * 0.299 + frames[..., 1] * 0.587 + frames[..., 2] * 0.114


def _round_trip(gray, factor: float):
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(gray)[:, None]
    h, w = x.shape[-2:]
    small = F.interpolate(x, size=(round(h / factor), round(w / factor)), mode="bilinear",
                          antialias=True, align_corners=False)
    return F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False)[:, 0].numpy()


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def _spectrum(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spec = np.abs(np.fft.fft2(gray - gray.mean(axis=(1, 2), keepdims=True))) ** 2
    h, w = gray.shape[1:]
    radius = np.sqrt(np.fft.fftfreq(w)[None, :] ** 2 + np.fft.fftfreq(h)[:, None] ** 2)
    return spec.sum(axis=0), radius


def _hf_fraction(gray: np.ndarray) -> float:
    power, radius = _spectrum(gray)
    total = power.sum()
    return float(power[radius > 0.25].sum() / total) if total > 0 else 0.0


def _effective_px(gray: np.ndarray, fraction: float = 0.99) -> int:
    power, radius = _spectrum(gray)
    order = np.argsort(radius, axis=None)
    cumulative = np.cumsum(power.ravel()[order])
    if cumulative[-1] <= 0:
        return 0
    f = float(radius.ravel()[order][np.searchsorted(cumulative, fraction * cumulative[-1])])
    return int(round(gray.shape[2] * min(f, 0.5) / 0.5))


def measure(frames: np.ndarray) -> dict:
    idx = np.linspace(0, len(frames) - 1, 12).astype(int)
    gray = _gray(frames[idx])
    out = {"hf_fraction": round(_hf_fraction(gray), 5), "effective_px": _effective_px(gray)}
    for factor in (2.0, 4.0):
        out[f"psnr_down{int(factor)}"] = round(float(_psnr(gray, _round_trip(gray, factor))), 2)
    return out


def read_mp4(path: Path) -> np.ndarray:
    import imageio.v3 as iio

    return np.asarray(iio.imread(str(path), plugin="FFMPEG"))[..., :3]


def decode_latents(path: Path, config: str, model_name: str) -> np.ndarray:
    import torch
    from omegaconf import OmegaConf

    sys.path.insert(0, str(OVI_ROOT))
    from ovi.ovi_fusion_engine import OviFusionEngine

    cfg = OmegaConf.load(config)
    cfg.model_name = model_name
    engine = OviFusionEngine(config=cfg, device=0)
    latents = torch.load(path)["video"].to(engine.device, engine.target_dtype)
    with torch.no_grad():
        video = engine.vae_model_video.wrapped_decode(latents.unsqueeze(0)).squeeze(0).float().cpu().numpy()
    # Exactly save_video's conversion: (C, F, H, W) in [-1, 1] -> uint8 (F, H, W, C).
    video = np.clip(video.transpose(1, 2, 3, 0), -1, 1)
    return ((video + 1) / 2 * 255).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("videos", nargs="*", type=Path)
    p.add_argument("--latents", type=Path, default=None)
    p.add_argument("--latents-mp4", type=Path, default=None, help="the MP4 written from --latents")
    p.add_argument("--config", default=str(OVI_ROOT / "ovi/configs/inference/inference_fusion_wsp.yaml"))
    p.add_argument("--model-name", default="960x960_10s")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    rows = []
    for video in args.videos:
        row = {"file": str(video), **measure(read_mp4(video))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.latents:
        raw = decode_latents(args.latents, args.config, args.model_name)
        row = {"file": f"RAW decode of {args.latents}", **measure(raw)}
        if args.latents_mp4:
            enc = read_mp4(args.latents_mp4)
            n = min(len(raw), len(enc))
            idx = np.linspace(0, n - 1, 12).astype(int)
            row["psnr_raw_vs_mp4"] = round(_psnr(_gray(raw[idx]), _gray(enc[idx])), 2)
            row["mp4"] = {"file": str(args.latents_mp4), **measure(enc)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.out.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
