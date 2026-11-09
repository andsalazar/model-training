"""Summarise a pod_wsp_adapter.py result: where the adapter helps, and where it writes.

Attention: for each (stream, block), the mean over the 12 timesteps of the largest
per-head effect against the teacher, for each variant. ``ratio`` is adapter / base
Euler-12 (same sampler, adapter off): below 1 the adapter moved that site toward the
teacher, at or above 1 it did not.

Writes: for each (stream, block), the adapter's relative write averaged over steps, per
latent-frame bin, and how much larger the write is in the last two bins than in the rest.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def attention_table(contrasts: dict) -> list[dict]:
    cells: dict = defaultdict(lambda: defaultdict(list))
    for variant, rows in contrasts.items():
        for r in rows:
            if r.get("skipped") or r.get("timestep_a", 1000) >= 999:
                continue
            cells[(r["stream"], r["layer"])][variant].append(r["max_effect"] or 0.0)
    out = []
    for (stream, layer), by in sorted(cells.items()):
        row = {"stream": stream, "block": layer}
        for variant, values in by.items():
            row[variant] = round(st.mean(values), 4)
        base = row.get("base_euler12")
        row["ratio_vs_euler"] = round(row["adapter_euler12"] / base, 3) if base else None
        unipc = row.get("base_unipc12")
        row["ratio_vs_unipc"] = round(row["adapter_euler12"] / unipc, 3) if unipc else None
        out.append(row)
    return out


def write_table(writes: list[dict]) -> list[dict]:
    groups: dict = defaultdict(list)
    for w in writes or []:
        groups[(w["stream"], w["block"])].append(w)
    out = []
    for (stream, block), rows in sorted(groups.items()):
        n = len(rows[0]["rel_per_bin"])
        mean_bins = [st.mean(r["rel_per_bin"][i] for r in rows) for i in range(n)]
        body = st.mean(mean_bins[:-2])
        tail = st.mean(mean_bins[-2:])
        per_step_tail = {r["step"]: round(st.mean(r["rel_per_bin"][-2:]) / max(st.mean(r["rel_per_bin"][:-2]), 1e-12), 2)
                         for r in rows}
        out.append({"stream": stream, "block": block, "mean_write": round(st.mean(mean_bins), 5),
                    "tail_over_body": round(tail / max(body, 1e-12), 2),
                    "first_bin_over_body": round(mean_bins[0] / max(body, 1e-12), 2),
                    "tail_ratio_by_step": per_step_tail})
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    data = json.loads(args.result.read_text())
    summary = {}
    for tag, entry in data["tags"].items():
        att = attention_table(entry["contrast_vs_teacher"])
        wr = write_table(entry["adapter_writes"])
        summary[tag] = {"attention": att, "writes": wr, "reports": entry["reports"]}
        print(f"\n===== {tag} =====")
        for name, rep in entry["reports"].items():
            print(f"  {name:<16} fired {rep['fired']} recorded {rep['recorded']} unlabelled {rep['unlabelled']} ok {rep['generate_ok']}")
        print(f"  {'stream':<6}{'blk':>4}{'euler':>9}{'adapter':>9}{'unipc12':>9}{'a/euler':>9}{'a/unipc':>9}")
        for r in att:
            print(f"  {r['stream']:<6}{r['block']:>4}{r.get('base_euler12', 0):>9.3f}{r.get('adapter_euler12', 0):>9.3f}"
                  f"{r.get('base_unipc12', 0):>9.3f}{r['ratio_vs_euler'] or 0:>9.2f}{r['ratio_vs_unipc'] or 0:>9.2f}")
        for r in wr:
            print(f"  write {r['stream']:<6} block {r['block']}: mean {r['mean_write']:.4f}  last-2-bins/body {r['tail_over_body']}"
                  f"  first/body {r['first_bin_over_body']}  by step {r['tail_ratio_by_step']}")
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
