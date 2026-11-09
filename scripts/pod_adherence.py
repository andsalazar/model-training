"""Prompt adherence for evaluated clips: does the video show, and the audio sound like,
what the prompt asked for?

* **Video:** CLIP ViT-L/14 cosine (x100) between 8 evenly spaced frames and the prompt's
  visual part (the text before "Audio:"), averaged over frames.
* **Audio:** CLAP (``laion/clap-htsat-unfused``) cosine (x100) between the clip's audio
  track, resampled to 48 kHz mono, and the prompt's audio part (the text after "Audio:").

Absolute CLIP/CLAP values depend on the prompt, so each clip is also scored against every
*other* evaluation prompt. ``margin = matched - mean(mismatched)`` is the
prompt-specific part; the comparison that matters is a variant's margin against the
teacher's on the same clip.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pod_adapter_run import P  # noqa: E402
from pod_ovi_audio_check import decode_audio  # noqa: E402


def split_prompt(prompt: str) -> tuple[str, str]:
    visual, sep, audio = prompt.partition("Audio:")
    return visual.strip(), (audio.strip() if sep else "")


def _frames(path: Path, n: int = 8) -> list[np.ndarray]:
    import imageio.v3 as iio

    video = np.asarray(iio.imread(str(path), plugin="FFMPEG"))[..., :3]
    idx = np.linspace(0, len(video) - 1, n).astype(int)
    return [video[i] for i in idx]


class Scorer:
    def __init__(self, device: str = "cuda") -> None:
        import torch
        from transformers import ClapModel, ClapProcessor, CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device).eval()
        self.clap_proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

    def _norm(self, x):
        return x / x.norm(dim=-1, keepdim=True)

    def clip_texts(self, texts: list[str]):
        with self.torch.no_grad():
            inputs = self.clip_proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            return self._norm(self.clip.get_text_features(**inputs))

    def clap_texts(self, texts: list[str]):
        with self.torch.no_grad():
            inputs = self.clap_proc(text=texts, return_tensors="pt", padding=True).to(self.device)
            return self._norm(self.clap.get_text_features(**inputs))

    def clip_video(self, path: Path):
        with self.torch.no_grad():
            inputs = self.clip_proc(images=_frames(path), return_tensors="pt").to(self.device)
            return self._norm(self.clip.get_image_features(**inputs))  # [frames, d]

    def clap_audio(self, path: Path):
        samples = decode_audio(path, sample_rate=48000)
        if samples is None or samples.size == 0:
            return None
        with self.torch.no_grad():
            inputs = self.clap_proc(audios=[samples.astype(np.float32)], sampling_rate=48000,
                                    return_tensors="pt").to(self.device)
            return self._norm(self.clap.get_audio_features(**inputs))  # [1, d]


def score_clips(scorer: Scorer, clips: list[dict]) -> list[dict]:
    """``clips``: dicts with ``tag``, ``variant``, ``path``. Returns them with scores."""
    names = sorted({c["tag"].rsplit("_s", 1)[0] for c in clips})
    visual = {n: split_prompt(P[n])[0] for n in names}
    audio = {n: split_prompt(P[n])[1] for n in names}
    tv = scorer.clip_texts([visual[n] for n in names])
    ta = scorer.clap_texts([audio[n] for n in names])
    out = []
    for clip in clips:
        name = clip["tag"].rsplit("_s", 1)[0]
        i = names.index(name)
        others = [j for j in range(len(names)) if j != i]
        frames = scorer.clip_video(Path(clip["path"]))
        clip_all = (frames @ tv.T).mean(dim=0) * 100  # per prompt
        row = {**clip, "clip": round(float(clip_all[i]), 3),
               "clip_mismatch": round(float(clip_all[others].mean()), 3) if others else None}
        emb = scorer.clap_audio(Path(clip["path"]))
        if emb is not None:
            clap_all = (emb @ ta.T)[0] * 100
            row["clap"] = round(float(clap_all[i]), 3)
            row["clap_mismatch"] = round(float(clap_all[others].mean()), 3) if others else None
        else:
            row["clap"] = None
        for key in ("clip", "clap"):
            if row.get(key) is not None and row.get(f"{key}_mismatch") is not None:
                row[f"{key}_margin"] = round(row[key] - row[f"{key}_mismatch"], 3)
        out.append(row)
        print(json.dumps({k: row.get(k) for k in ("tag", "variant", "clip", "clip_margin", "clap", "clap_margin")}),
              flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, required=True, help="dir with eval.json and the variant MP4s")
    p.add_argument("--teacher-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rows = json.loads((args.eval_dir / "eval.json").read_text())
    clips = []
    for r in rows:
        if "error" in r:
            continue
        path = (args.teacher_dir / f"{r['tag']}.mp4" if r["variant"] == "teacher48"
                else args.eval_dir / f"{r['variant']}_{r['tag']}.mp4")
        if path.exists():
            clips.append({"tag": r["tag"], "variant": r["variant"], "path": str(path)})
    scored = score_clips(Scorer(), clips)
    args.out.write_text(json.dumps(scored, indent=2))
    print(f"wrote {args.out} ({len(scored)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
