"""Analyse the deciding experiment (Stage 7 v4): does the adapter beat running more steps?

Ten unseen prompts at three seeds, every variant generated on one pod. The unit of
analysis is the **prompt**: the three seeds of a prompt are averaged before any test, so
ten prompts give ten independent paired differences.

Per clip, each variant is compared with the 48-step teacher of the same prompt and seed:

* ``log``: mean ``|log(a / b)|`` over sharpness, flicker, motion coherence and colour
  stability. Symmetric; the primary measure.
* ``relative``: mean ``|a - b| / b``, the measure the earlier Stage 7 documents used.

A third measure does not depend on a single teacher sample. ``pooled`` averages each
metric over a prompt's seeds first, for the variant and for the teacher, and then takes
the same ``|log ratio|``: it compares what a variant produces for a prompt on average.

Usage::

    python scripts/analyse_decide.py artifacts/v4_decide --out artifacts/v4_decide/decide.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def _load_paired():
    spec = importlib.util.spec_from_file_location("analyse_v3_paired", _HERE / "analyse_v3_paired.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("analyse_v3_paired", module)
    spec.loader.exec_module(module)
    return module


stats = _load_paired()
METRICS = stats.METRICS
TEACHER = stats.TEACHER
ADAPTER = "v2_euler12"
#: Base-model UniPC at each step count: the curve the adapter is placed on.
CURVE = {12: "base_unipc12", 16: "base_unipc16", 20: "base_unipc20", 30: "base_unipc30"}
#: (first, second); every difference is first minus second, at prompt level.
COMPARISONS = [
    (ADAPTER, "base_unipc12"),   # same cost: does the adapter help at all?
    (ADAPTER, "base_euler12"),   # same sampler: the adapter's own effect
    (ADAPTER, "base_unipc16"),   # does it beat 4 more steps?
    (ADAPTER, "base_unipc20"),   # does it beat 8 more steps?
    (ADAPTER, "base_unipc30"),
    ("base_unipc16", "base_unipc12"),  # can this design resolve the base curve at all?
    ("base_unipc20", "base_unipc16"),
    ("base_unipc30", "base_unipc20"),
]


def split_tag(tag: str) -> tuple[str, int]:
    prompt, _, seed = tag.rpartition("_s")
    return prompt, int(seed)


def load_eval(directory: Path) -> dict:
    """``{(tag, variant): row}`` for every row without an error."""
    rows = json.loads((directory / "eval.json").read_text())
    return {(r["tag"], r["variant"]): r for r in rows if "error" not in r}


def clip_distances(by: dict, kind: str) -> dict[str, dict[str, float]]:
    """``{variant: {tag: distance to the teacher of the same prompt and seed}}``."""
    fn = stats.DISTANCES[kind]
    out: dict[str, dict[str, float]] = {}
    for (tag, variant), row in by.items():
        if variant == TEACHER or (tag, TEACHER) not in by:
            continue
        out.setdefault(variant, {})[tag] = fn(row["quality"], by[(tag, TEACHER)]["quality"])
    return out


def pooled_distances(by: dict) -> dict[str, dict[str, float]]:
    """``{variant: {prompt: |log ratio| of seed-averaged metrics}}``, over shared seeds."""
    out: dict[str, dict[str, float]] = {}
    for variant in sorted({v for _, v in by} - {TEACHER}):
        tags_by_prompt: dict[str, list[str]] = {}
        for tag, v in by:
            if v == variant and (tag, TEACHER) in by:
                tags_by_prompt.setdefault(split_tag(tag)[0], []).append(tag)
        for prompt, tags in tags_by_prompt.items():
            total = 0.0
            for m in METRICS:
                mine = np.mean([by[(t, variant)]["quality"][m] for t in tags])
                ref = np.mean([by[(t, TEACHER)]["quality"][m] for t in tags])
                total += abs(math.log(mine / ref))
            out.setdefault(variant, {})[prompt] = total / len(METRICS)
    return out


def by_prompt(per_clip: dict[str, float], tags: set[str] | None = None) -> dict[str, float]:
    """Average a per-clip value over the seeds of each prompt."""
    groups: dict[str, list[float]] = {}
    for tag, value in per_clip.items():
        if tags is None or tag in tags:
            groups.setdefault(split_tag(tag)[0], []).append(value)
    return {prompt: float(np.mean(values)) for prompt, values in groups.items()}


def compare(a: dict[str, float], b: dict[str, float], *, per_clip: bool) -> dict | None:
    """Paired prompt-level statistics of ``a - b``.

    For per-clip inputs only clips present for both variants are used, so a run that was
    cut short compares like with like.
    """
    if per_clip:
        shared = set(a) & set(b)
        a, b = by_prompt(a, shared), by_prompt(b, shared)
    prompts = sorted(set(a) & set(b))
    if not prompts:
        return None
    result = stats.paired([a[p] - b[p] for p in prompts])
    result["prompts"] = prompts
    return result


def equivalent_steps(curve: dict[int, float], value: float) -> dict:
    """Where ``value`` falls on the base model's distance-against-steps curve.

    Linear interpolation between neighbouring step counts. Returns ``steps`` when the
    value lies on the curve, otherwise which end it falls off. A curve that does not
    fall as steps rise cannot place anything and is reported as such.
    """
    steps = sorted(curve)
    values = [curve[s] for s in steps]
    if any(later >= earlier for earlier, later in zip(values, values[1:])):
        return {"steps": None, "note": "base curve is not monotone; no placement"}
    if value >= values[0]:
        return {"steps": None, "note": f"no better than {steps[0]} steps"}
    if value <= values[-1]:
        return {"steps": None, "note": f"at least as good as {steps[-1]} steps"}
    for (s0, v0), (s1, v1) in zip(zip(steps, values), zip(steps[1:], values[1:])):
        if v1 <= value <= v0:
            return {"steps": float(s0 + (s1 - s0) * (v0 - value) / (v0 - v1)), "note": "interpolated"}
    raise AssertionError("unreachable")


def variant_summary(by: dict, distances: dict) -> dict:
    out = {}
    for variant in sorted({v for _, v in by}):
        rows = [r for (_, v), r in by.items() if v == variant]
        seconds = [r["seconds"] for r in rows if r.get("seconds")]
        entry = {"clips": len(rows), "seconds": float(np.mean(seconds)) if seconds else None}
        for kind, table in distances.items():
            if variant in table:
                per_prompt = by_prompt(table[variant]) if kind != "pooled" else table[variant]
                entry[kind] = float(np.mean(list(per_prompt.values())))
        out[variant] = entry
    return out


def decomposition(by: dict) -> dict:
    """Per metric: mean absolute and signed relative error against the teacher, over clips."""
    out: dict = {}
    for variant in sorted({v for _, v in by} - {TEACHER}):
        out[variant] = {}
        for m in METRICS:
            signed = [row["quality"][m] / by[(tag, TEACHER)]["quality"][m] - 1
                      for (tag, v), row in by.items() if v == variant and (tag, TEACHER) in by]
            out[variant][m] = {"mean_abs": float(np.mean(np.abs(signed))),
                               "mean_signed": float(np.mean(signed)),
                               "below_teacher": int(sum(s < 0 for s in signed)), "n": len(signed)}
    return out


def adherence(directory: Path) -> dict:
    path = directory / "adherence.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    out: dict = {}
    for key in ("clip_margin", "clip", "clap_margin"):
        table: dict[str, dict[str, float]] = {}
        for r in rows:
            if r.get(key) is not None:
                table.setdefault(r["variant"], {})[r["tag"]] = r[key]
        block = {"mean": {v: float(np.mean(list(by_prompt(t).values()))) for v, t in table.items()}, "paired": {}}
        for a, b in [*COMPARISONS, *[(v, TEACHER) for v in table if v != TEACHER]]:
            if a in table and b in table:
                result = compare(table[a], table[b], per_clip=True)
                if result:
                    block["paired"][f"{a} - {b}"] = result
        out[key] = block
    return out


def audio(directory: Path, variants: set[str]) -> dict:
    """Audio level and brightness against the teacher, from each clip's MP4 audio track.

    ``audio_mp4.json`` is written by ``pod_ovi_audio_check.py`` with one tool for every
    clip. The error is ``|log(variant / teacher)|`` for RMS and for the spectral centroid.
    """
    path = directory / "audio_mp4.json"
    if not path.exists():
        return {}
    readings: dict[tuple[str, str], dict] = {}
    for row in json.loads(path.read_text()):
        metrics = row.get("audio") or {}
        if not metrics.get("rms") or not metrics.get("centroid_mean_hz"):
            continue
        stem = row["file"][:-4]
        if row["dir"] == "teacher":
            readings[(stem, TEACHER)] = metrics
            continue
        for variant in variants:
            if stem.startswith(variant + "_"):
                readings[(stem[len(variant) + 1:], variant)] = metrics
    out: dict = {}
    for key, field in (("loudness", "rms"), ("brightness", "centroid_mean_hz")):
        table: dict[str, dict[str, float]] = {}
        for (tag, variant), metrics in readings.items():
            if variant != TEACHER and (tag, TEACHER) in readings:
                table.setdefault(variant, {})[tag] = abs(math.log(metrics[field] / readings[(tag, TEACHER)][field]))
        block = {"mean": {v: float(np.mean(list(by_prompt(t).values()))) for v, t in table.items()}, "paired": {}}
        for a, b in COMPARISONS:
            if a in table and b in table:
                found = compare(table[a], table[b], per_clip=True)
                if found:
                    block["paired"][f"{a} - {b}"] = found
        out[key] = block
    return out


def analyse(directory: Path) -> dict:
    by = load_eval(directory)
    distances = {"log": clip_distances(by, "log"), "relative": clip_distances(by, "relative"),
                 "pooled": pooled_distances(by)}
    result: dict = {
        "clips": sorted({tag for tag, _ in by}),
        "prompts": sorted({split_tag(tag)[0] for tag, _ in by}),
        "variants": variant_summary(by, distances),
        "paired": {}, "placement": {}, "decomposition": decomposition(by),
        "adherence": adherence(directory),
        "audio": audio(directory, {v for _, v in by} - {TEACHER}),
    }
    for kind, table in distances.items():
        block = {}
        for a, b in COMPARISONS:
            if a in table and b in table:
                found = compare(table[a], table[b], per_clip=kind != "pooled")
                if found:
                    block[f"{a} - {b}"] = found
        result["paired"][kind] = block
        curve = {s: result["variants"][v][kind] for s, v in CURVE.items() if kind in result["variants"].get(v, {})}
        if len(curve) == len(CURVE) and kind in result["variants"].get(ADAPTER, {}):
            result["placement"][kind] = {"curve": curve, "adapter": result["variants"][ADAPTER][kind],
                                         **equivalent_steps(curve, result["variants"][ADAPTER][kind])}
    return result


def _row(name: str, s: dict, better: str) -> str:
    wins = s["negative"] if better == "negative" else s["positive"]
    return (f"  {name:<32}{s['mean']:+8.3f}  [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"
            f"  {wins}/{s['n']} prompts first better  p={s['sign_flip_p']:.3f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    result = analyse(args.dir)
    print(f"{len(result['clips'])} clips, {len(result['prompts'])} prompts")
    print(f"\n{'variant':<16}{'clips':>6}{'seconds':>9}{'log':>8}{'relative':>10}{'pooled':>8}")
    for variant, row in result["variants"].items():
        cells = "".join(f"{row[k]:>{w}.3f}" if k in row else f"{'-':>{w}}"
                        for k, w in (("log", 8), ("relative", 10), ("pooled", 8)))
        seconds = f"{row['seconds']:.1f}" if row["seconds"] else "-"
        print(f"{variant:<16}{row['clips']:>6}{seconds:>9}{cells}")
    for kind, block in result["paired"].items():
        print(f"\ndistance to the teacher, {kind} (negative = first is closer), by prompt")
        for name, s in block.items():
            print(_row(name, s, "negative"))
    for kind, place in result["placement"].items():
        steps = f"{place['steps']:.1f} UniPC steps" if place["steps"] else place["note"]
        print(f"\nplacement on the base curve ({kind}): adapter {place['adapter']:.3f} -> {steps}; "
              f"curve {', '.join(f'{s}: {v:.3f}' for s, v in place['curve'].items())}")
    for key, block in result["adherence"].items():
        print(f"\n{key} (positive = first is higher), by prompt; means: "
              + ", ".join(f"{v} {m:.2f}" for v, m in sorted(block["mean"].items())))
        for name, s in block["paired"].items():
            print(_row(name, s, "positive"))
    for key, block in result["audio"].items():
        print(f"\naudio {key} error against the teacher (negative = first is closer), by prompt; means: "
              + ", ".join(f"{v} {m:.3f}" for v, m in sorted(block["mean"].items())))
        for name, s in block["paired"].items():
            print(_row(name, s, "negative"))
    print("\nper-metric relative error against the teacher: mean |error| (mean signed), clips below teacher")
    for variant, metrics in result["decomposition"].items():
        cells = "  ".join(f"{m[:9]} {d['mean_abs']:.3f} ({d['mean_signed']:+.3f}) {d['below_teacher']}/{d['n']}"
                          for m, d in metrics.items())
        print(f"  {variant:<16}{cells}")
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
