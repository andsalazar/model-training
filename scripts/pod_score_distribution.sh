#!/bin/bash
# Examine the alignment score distribution, not just its maximum.
#
# Every site reported selected=4, which is the min_columns backfill: the 0.9-of-best
# threshold selected zero columns. That is a statement about the shape of the score
# distribution -- probably a broad plateau near zero rather than a peak -- and the shape
# is what decides whether an edit is justified.
set -uo pipefail
source /local/ovi-env/bin/activate
REPORT="${1:-/local/captures/projection.json}"
python3 - "$REPORT" <<'PY'
import json
import sys

rows = [r for r in json.load(open(sys.argv[1])) if "top_scores" in r]
print(f"{'key':52s} {'best':>7} {'2nd':>7} {'10th':>7} {'spread':>7} {'sel':>4}")
for row in rows:
    scores = row["top_scores"]
    spread = scores[0] - scores[-1] if len(scores) > 1 else 0.0
    print(
        f"{row['key']:52s} {scores[0]:>+7.4f} {scores[1]:>+7.4f} "
        f"{scores[-1]:>+7.4f} {spread:>7.4f} {row['selected']:>4}"
    )

print()
print("Interpretation inputs:")
best = [r["top_scores"][0] for r in rows]
print(f"  max cosine across all sites: {max(best):.4f}")
print(f"  mean max cosine: {sum(best)/len(best):.4f}")
print(f"  threshold (0.9 * best) is below every score? they selected "
      f"{sorted({r['selected'] for r in rows})}")
PY
