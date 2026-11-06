"""Does the generated mp4 actually contain an audio track, and does it change with steps?

Ovi is a joint video+audio model, so "quality" includes audio. Every metric in
wsp.quality reads pixels only, which means the speed sweep has said nothing about
half of what the model produces. This checks whether there is audio to measure at all
before building an audio metric for it.

Uses ffprobe for stream detection and ffmpeg to decode to PCM, so it needs neither
torch nor the model — just the ffmpeg binary imageio_ffmpeg already ships.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _ffmpeg() -> str | None:
    """Path to an ffmpeg binary, or None when none is available.

    Returning None rather than raising keeps the "no tool" case distinguishable from
    "no audio" all the way up to the report.
    """
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - any import/binary failure means "unavailable"
        import shutil

        return shutil.which("ffmpeg")


def _ffprobe() -> str | None:
    """ffprobe next to the ffmpeg binary, if the build ships one."""
    ff = _ffmpeg()
    if ff is None:
        return None
    path = Path(ff)
    candidate = path.with_name(path.name.replace("ffmpeg", "ffprobe"))
    return str(candidate) if candidate.exists() else None


def streams(path: Path) -> dict:
    """Stream summary via ffprobe, or ffmpeg's own stderr when ffprobe is absent.

    ``imageio_ffmpeg`` ships ffmpeg but NOT ffprobe, so the ffprobe path is usually
    unavailable. Falling back to "no streams" in that case would report "this video has
    no audio" when the truth is "nothing was inspected" — the exact false-negative this
    project has been bitten by three times. So the fallback parses ffmpeg's banner.
    """
    probe = _ffprobe()
    if probe:
        out = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-show_entries",
                "stream=index,codec_type,codec_name,channels,sample_rate",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if out.returncode == 0:
            try:
                data = json.loads(out.stdout)
                return {"streams": data.get("streams", []), "via": "ffprobe"}
            except json.JSONDecodeError:
                pass

    # ffmpeg -i with no output writes the stream table to stderr and exits non-zero.
    ff = _ffmpeg()
    if ff is None:
        return {"streams": [], "via": "none"}
    out = subprocess.run(
        [ff, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    found: list[dict] = []
    for line in out.stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith("Stream #"):
            kind = "video" if "Video:" in stripped else "audio" if "Audio:" in stripped else "other"
            found.append({"codec_type": kind, "raw": stripped})
    return {"streams": found, "via": "ffmpeg-banner"}


def decode_audio(path: Path, sample_rate: int = 16000) -> np.ndarray | None:
    """Decode the first audio stream to mono float32, or None if there is none."""
    ff = _ffmpeg()
    if ff is None:
        return None
    out = subprocess.run(
        [
            ff,
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        timeout=300,
    )
    if out.returncode != 0 or not out.stdout:
        return None
    return np.frombuffer(out.stdout, dtype=np.float32)


def audio_metrics(samples: np.ndarray, sample_rate: int = 16000) -> dict:
    """Basic audio descriptors: level, dynamics, and spectral centroid drift.

    These are deliberately simple. The point is to detect whether audio *changes*
    with the step budget, not to make a perceptual quality claim.
    """
    if samples.size == 0:
        return {"error": "empty audio"}
    rms = float(np.sqrt(np.mean(samples**2)))
    peak = float(np.max(np.abs(samples)))
    # Frame the signal for a spectral-centroid series, which is the audio analogue of
    # the colour-drift check: a centroid that trends means the timbre is changing.
    win = 2048
    hop = win // 2
    n_frames = max(0, (samples.size - win) // hop + 1)
    centroids: list[float] = []
    for i in range(0, min(n_frames, 400)):
        seg = samples[i * hop : i * hop + win]
        if seg.size < win:
            break
        spec = np.abs(np.fft.rfft(seg * np.hanning(win)))
        freqs = np.fft.rfftfreq(win, 1.0 / sample_rate)
        total = float(spec.sum())
        if total > 1e-12:
            centroids.append(float((spec * freqs).sum() / total))
    centroid_series = np.asarray(centroids, dtype=np.float64)
    if centroid_series.size >= 4:
        x = np.arange(centroid_series.size, dtype=np.float64)
        xm = x.mean()
        ym = centroid_series.mean()
        denom = float(((x - xm) ** 2).sum())
        slope = float(((x - xm) * (centroid_series - ym)).sum() / denom) if denom else 0.0
        pred = ym + slope * (x - xm)
        ss_res = float(((centroid_series - pred) ** 2).sum())
        ss_tot = float(((centroid_series - ym) ** 2).sum())
        r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    else:
        slope, r2 = 0.0, 0.0
    return {
        "seconds": round(samples.size / sample_rate, 3),
        "rms": round(rms, 5),
        "peak": round(peak, 4),
        "centroid_mean_hz": round(float(centroid_series.mean()), 1)
        if centroid_series.size
        else None,
        "centroid_slope_hz_per_frame": round(slope, 3),
        "centroid_r2": round(float(r2), 3),
        "n_spectral_frames": int(centroid_series.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="/workspace/audio_check.json")
    args = ap.parse_args()

    rows = []
    for v in args.videos:
        path = Path(v)
        if not path.exists():
            print(f"MISSING {path}")
            continue
        info = streams(path)
        audio_streams = [s for s in info["streams"] if s.get("codec_type") == "audio"]
        row: dict = {
            "file": path.name,
            "dir": path.parent.name,
            "audio_streams": len(audio_streams),
            "streams": info["streams"],
        }
        if audio_streams:
            samples = decode_audio(path)
            row["audio"] = (
                audio_metrics(samples) if samples is not None else {"error": "decode failed"}
            )
        rows.append(row)
        a = row.get("audio", {})
        print(
            f"{row['dir']:<28} streams={len(info['streams'])} audio={row['audio_streams']} "
            f"rms={a.get('rms')} centroid={a.get('centroid_mean_hz')} r2={a.get('centroid_r2')}"
        )

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
