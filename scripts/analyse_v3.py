"""Summarise a Stage 7 v3 evaluation: video, flash, audio and prompt adherence per variant.

Everything is relative to the 48-step teacher on the same clip. Video distance is the mean
|relative difference| over sharpness, flicker, motion coherence and colour stability (the
Stage 7 measure). Audio uses one tool for every clip (the MP4 audio track). Adherence is
CLIP (video) and CLAP (audio), as the margin of matched over mismatched prompts.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

M = ["sharpness", "temporal_flicker", "motion_coherence", "colour_stability"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dir", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    rows = [r for r in json.loads((args.dir / "eval.json").read_text()) if "error" not in r]
    adh = {(r["tag"], r["variant"]): r for r in json.loads((args.dir / "adherence.json").read_text())}
    audio = {}
    for r in json.loads((args.dir / "audio_mp4.json").read_text()):
        stem = r["file"][:-4]
        if r["dir"] == "teacher":
            audio[(stem, "teacher48")] = r["audio"]
        else:
            variant, _, tag = stem.rpartition("_")
            tag = stem[len(stem.split("_s")[0].rsplit("_", 1)[0]) + 1:] if False else None
            for v in {x["variant"] for x in rows}:
                if stem.startswith(v + "_"):
                    audio[(stem[len(v) + 1:], v)] = r["audio"]
    by = {(r["tag"], r["variant"]): r for r in rows}
    tags = sorted({r["tag"] for r in rows}, key=lambda t: (not t.startswith("river"), t))
    river = [t for t in tags if t.startswith("river")]
    unseen = [t for t in tags if not t.startswith("river")]
    variants = [v for v in ["teacher48", "base_unipc30", "base_unipc12", "v2_euler12", "v2fix_euler12",
                            "onpol_euler12", "onpol03_euler12"] if any(k[1] == v for k in by)]
    out: dict = {"tags": tags, "variants": {}}

    def dist(t, v):
        a, b = by[(t, v)]["quality"], by[(t, "teacher48")]["quality"]
        return sum(abs(a[m] - b[m]) / abs(b[m]) for m in M) / 4

    print(f"{'variant':<18}{'river':>8}{'unseen':>8}{'all':>8}{'worst':>8} {'flash max':>9}{'>3':>4}"
          f"{'rms err':>9}{'cent err':>9}{'CLIP m':>8}{'CLAP m':>8}")
    for v in variants:
        d = {t: dist(t, v) for t in tags} if v != "teacher48" else {t: 0.0 for t in tags}
        flashes = [by[(t, v)].get("last8_max_jump") for t in tags if by[(t, v)].get("last8_max_jump") is not None]
        rms = [abs(audio[(t, v)]["rms"] / audio[(t, "teacher48")]["rms"] - 1) for t in tags
               if (t, v) in audio and (t, "teacher48") in audio]
        cen = [abs(audio[(t, v)]["centroid_mean_hz"] / audio[(t, "teacher48")]["centroid_mean_hz"] - 1)
               for t in tags if (t, v) in audio and (t, "teacher48") in audio]
        cm = [adh[(t, v)].get("clip_margin") for t in tags if (t, v) in adh]
        am = [adh[(t, v)].get("clap_margin") for t in tags if (t, v) in adh and adh[(t, v)].get("clap_margin") is not None]
        row = {
            "dist_river": st.mean(d[t] for t in river), "dist_unseen": st.mean(d[t] for t in unseen),
            "dist_all": st.mean(d.values()), "dist_worst": max(d.values()), "dist_per_clip": d,
            "flash_max": max(flashes) if flashes else None, "flash_over3": sum(f > 3 for f in flashes),
            "rms_err": st.mean(rms) if rms else None, "centroid_err": st.mean(cen) if cen else None,
            "clip_margin": st.mean(cm) if cm else None, "clap_margin": st.mean(am) if am else None,
            "clip": st.mean(adh[(t, v)]["clip"] for t in tags if (t, v) in adh),
            "clap": st.mean(adh[(t, v)]["clap"] for t in tags if (t, v) in adh and adh[(t, v)].get("clap") is not None),
            "seconds": st.mean(by[(t, v)].get("seconds", 0) or 0 for t in tags),
        }
        out["variants"][v] = row
        f = lambda x, n=3: "-" if x is None else f"{x:.{n}f}"  # noqa: E731
        print(f"{v:<18}{f(row['dist_river']):>8}{f(row['dist_unseen']):>8}{f(row['dist_all']):>8}"
              f"{f(row['dist_worst']):>8} {f(row['flash_max'], 1):>9}{row['flash_over3']:>4}"
              f"{f(row['rms_err']):>9}{f(row['centroid_err'], 4):>9}{f(row['clip_margin'], 2):>8}"
              f"{f(row['clap_margin'], 2):>8}")
    print("\nper-clip video distance:")
    print(f"{'':18}" + "".join(f"{t[:11]:>12}" for t in tags))
    for v in variants[1:]:
        print(f"{v:<18}" + "".join(f"{out['variants'][v]['dist_per_clip'][t]:>12.3f}" for t in tags))
    print("\nper-clip flash (last-8-frame max brightness jump):")
    for v in variants[1:]:
        print(f"{v:<18}" + "".join(f"{by[(t, v)].get('last8_max_jump', float('nan')):>12.2f}" for t in tags))
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
