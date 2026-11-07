#!/bin/bash
# Compare the audio projection against the video one, and show the score spread.
#
# The key question is not the maximum alignment but whether the distribution is a
# plateau: a top score barely above the second means the threshold rule found nothing
# and the min_columns backfill supplied the selection.
set -uo pipefail

source /local/ovi-env/bin/activate

python3 - <<'PY'
import json
from pathlib import Path

for name in ("projection_audio.json", "projection.json"):
    path = Path("/local/captures") / name
    if not path.exists():
        print(f"{name}: not present")
        continue
    data = json.loads(path.read_text())
    # The projection script writes a bare list of site rows; an earlier summary assumed
    # a dict with a "rows" key and crashed on it. Accept both shapes.
    if isinstance(data, dict):
        rows = data.get("rows") or data.get("findings") or []
    else:
        rows = data
    print(f"=== {name} ({len(rows)} rows) ===")
    print(f"{'site':>34} {'best':>8} {'2nd':>8} {'10th':>8} {'spread':>8} {'sel':>4} {'thresh':>8}")
    for r in rows:
        if "skipped" in r:
            print(f"{r.get('key','?'):>34} SKIPPED")
            continue
        scores = r.get("top_scores") or r.get("scores") or []
        if not scores:
            continue
        best = scores[0]
        second = scores[1] if len(scores) > 1 else float("nan")
        tenth = scores[min(9, len(scores) - 1)]
        key = str(r.get("key") or "?").replace("audio_model.", "").replace("video_model.", "")
        print(
            f"{key:>34} {best:>+8.4f} {second:>+8.4f} {tenth:>+8.4f} "
            f"{best - tenth:>8.4f} {r.get('selected', 0):>4} "
            f"{r.get('selection_threshold', 0):>8.4f}"
        )
    print()
    bests = [r["top_scores"][0] for r in rows if r.get("top_scores")]
    seconds = [r["top_scores"][1] for r in rows if len(r.get("top_scores") or []) > 1]
    if bests:
        print(f"best alignment anywhere:  {max(bests):+.4f}")
        print(f"mean best:                {sum(bests)/len(bests):+.4f}")
        if seconds:
            print(
                "mean gap best-to-2nd:     "
                f"{sum(b - s for b, s in zip(bests, seconds))/len(seconds):.4f}"
            )
        print(f"chance level 1/sqrt(3072): {1/3072**0.5:.4f}")
        print()
        print(f"selected counts across sites: {sorted({r.get('selected', 0) for r in rows})}")
        print("(4 is min_columns -- the threshold found nothing and the backfill supplied it)")
        print()
        by_site = {}
        for r in rows:
            attr = str(r.get("key", "")).split(".")[-2]
            by_site.setdefault(attr, []).append(r["top_scores"][0])
        for attr, vals in sorted(by_site.items()):
            print(
                f"  {attr:>12}: best {max(vals):+.4f}  "
                f"mean {sum(vals)/len(vals):+.4f}  n={len(vals)}"
            )
    print()
PY
