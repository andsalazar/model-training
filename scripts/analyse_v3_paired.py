"""Paired comparisons for a Stage 7 v3 evaluation, reproducible from the committed JSON.

``analyse_v3.py`` prints per-variant means. This script produces the paired statistics
the write-up quotes, and three checks the first analysis lacked:

* an **exact sign-flip test** next to the percentile bootstrap, because a bootstrap
  interval from 6-9 clips is too narrow to lean on;
* a **symmetric distance**. The Stage 7 measure, mean ``|a - b| / b``, penalises a metric
  that overshoots the teacher more than one that undershoots by the same ratio (2x reads
  1.0, 0.5x reads 0.5). ``|log(a / b)|`` treats both the same;
* a **per-metric decomposition**, so it is visible which of the four metrics a variant's
  gain comes from, and in which direction it misses.

Usage::

    python scripts/analyse_v3_paired.py artifacts/v3_5s --out artifacts/v3_5s/paired.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

METRICS = ["sharpness", "temporal_flicker", "motion_coherence", "colour_stability"]
TEACHER = "teacher48"
#: (first, second): differences are first minus second.
COMPARISONS = [
    ("v2_euler12", "base_unipc12"),
    ("v2_euler12", "base_unipc30"),
    ("base_unipc30", "base_unipc12"),
    ("onpol_euler12", "v2_euler12"),
    ("onpol03_euler12", "v2_euler12"),
    ("v2fix_euler12", "v2_euler12"),
]
#: Above this many clips the exact test would enumerate more than a million sign patterns.
MAX_EXACT = 20


def relative_distance(quality: dict, reference: dict, metrics: list[str] = METRICS) -> float:
    """Mean ``|a - b| / |b|`` over ``metrics``: the Stage 7 measure. Asymmetric."""
    return sum(abs(quality[m] - reference[m]) / abs(reference[m]) for m in metrics) / len(metrics)


def log_distance(quality: dict, reference: dict, metrics: list[str] = METRICS) -> float:
    """Mean ``|log(a / b)|`` over ``metrics``: the same for 2x and 0.5x."""
    total = 0.0
    for m in metrics:
        ratio = quality[m] / reference[m]
        if not ratio > 0:
            raise ValueError(f"{m}: ratio {ratio} has no logarithm")
        total += abs(math.log(ratio))
    return total / len(metrics)


DISTANCES = {"relative": relative_distance, "log": log_distance}


def sign_flip_p(diffs) -> float:
    """Exact two-sided sign-flip test of "the paired differences centre on zero".

    Under that hypothesis each difference is as likely to have either sign, so the
    observed mean is compared with the mean under every one of the ``2**n`` sign patterns.
    """
    x = np.asarray(diffs, dtype=float)
    if x.size == 0:
        return float("nan")
    if x.size > MAX_EXACT:
        raise ValueError(f"{x.size} pairs is too many to enumerate exactly (max {MAX_EXACT})")
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=x.size)))
    means = (signs * x).mean(axis=1)
    return float((np.abs(means) >= abs(x.mean()) - 1e-12).mean())


def bootstrap_interval(diffs, n_boot: int = 20000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean paired difference."""
    x = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size)).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired(diffs) -> dict:
    """Summary of paired differences (first minus second)."""
    x = np.asarray(diffs, dtype=float)
    lo, hi = bootstrap_interval(x)
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "ci95": [lo, hi],
        "negative": int((x < 0).sum()),
        "positive": int((x > 0).sum()),
        "sign_flip_p": sign_flip_p(x),
    }


def load(directory: Path) -> tuple[dict, list[str]]:
    rows = [r for r in json.loads((directory / "eval.json").read_text()) if "error" not in r]
    by = {(r["tag"], r["variant"]): r["quality"] for r in rows}
    tags = sorted({t for t, _ in by}, key=lambda t: (not t.startswith("river"), t))
    return by, tags


def subsets(tags: list[str]) -> dict[str, list[str]]:
    return {"all": tags, "unseen": [t for t in tags if not t.startswith("river")]}


def distance_comparisons(by: dict, tags: list[str]) -> dict:
    variants = {v for _, v in by}
    out: dict = {}
    for name, fn in DISTANCES.items():
        per = {
            v: {t: fn(by[(t, v)], by[(t, TEACHER)]) for t in tags if (t, v) in by}
            for v in sorted(variants - {TEACHER})
        }
        block: dict = {"per_clip": per, "paired": {}}
        for a, b in COMPARISONS:
            if a not in per or b not in per:
                continue
            block["paired"][f"{a} - {b}"] = {
                label: paired([per[a][t] - per[b][t] for t in sel])
                for label, sel in subsets(tags).items()
            }
        out[name] = block
    return out


def decomposition(by: dict, tags: list[str]) -> dict:
    """Per metric: mean absolute and mean signed relative error against the teacher."""
    out: dict = {}
    for v in sorted({v for _, v in by} - {TEACHER}):
        out[v] = {}
        for m in METRICS:
            signed = [by[(t, v)][m] / by[(t, TEACHER)][m] - 1 for t in tags if (t, v) in by]
            out[v][m] = {
                "mean_abs": float(np.mean(np.abs(signed))),
                "mean_signed": float(np.mean(signed)),
                "below_teacher": int(sum(s < 0 for s in signed)),
                "n": len(signed),
            }
    return out


def adherence_comparisons(directory: Path, tags: list[str]) -> dict:
    path = directory / "adherence.json"
    if not path.exists():
        return {}
    rows = {(r["tag"], r["variant"]): r for r in json.loads(path.read_text())}
    out: dict = {}
    for key in ("clip_margin", "clip", "clap_margin"):
        out[key] = {}
        for a, b in COMPARISONS:
            block = {}
            for label, sel in subsets(tags).items():
                diffs = [
                    rows[(t, a)][key] - rows[(t, b)][key]
                    for t in sel
                    if rows.get((t, a), {}).get(key) is not None
                    and rows.get((t, b), {}).get(key) is not None
                ]
                if diffs:
                    block[label] = paired(diffs)
            if block:
                out[key][f"{a} - {b}"] = block
    return out


def _line(name: str, label: str, s: dict, better: str) -> str:
    wins = s["negative"] if better == "negative" else s["positive"]
    return (f"  {name:<34}{label:<8}{s['mean']:+8.3f}  [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"
            f"  {wins}/{s['n']} first better  p={s['sign_flip_p']:.3f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    by, tags = load(args.dir)
    result = {
        "tags": tags,
        "distance": distance_comparisons(by, tags),
        "decomposition": decomposition(by, tags),
        "adherence": adherence_comparisons(args.dir, tags),
    }
    for name, block in result["distance"].items():
        print(f"\nvideo distance to the teacher, {name} (negative = first is closer)")
        for comparison, by_subset in block["paired"].items():
            for label, s in by_subset.items():
                print(_line(comparison, label, s, "negative"))
    print("\nper-metric relative error against the teacher: mean |error| (mean signed), clips below teacher")
    for v, metrics in result["decomposition"].items():
        cells = "  ".join(
            f"{m[:9]} {d['mean_abs']:.3f} ({d['mean_signed']:+.3f}) {d['below_teacher']}/{d['n']}"
            for m, d in metrics.items()
        )
        print(f"  {v:<18}{cells}")
    for key, block in result["adherence"].items():
        print(f"\n{key} (positive = first is higher)")
        for comparison, by_subset in block.items():
            for label, s in by_subset.items():
                print(_line(comparison, label, s, "positive"))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
