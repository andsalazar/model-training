#!/bin/bash
# Smoke test for the deciding experiment: one prompt, one seed, every variant, through the
# same script as the real run. About 15 minutes on an H100. It
# checks that every path runs and times each step count. It writes into the real work
# directory, so a passing smoke test is also the first clip of the run; if a fix is needed
# afterwards, delete /local/v4_decide before the real run.
set -uo pipefail
bash /local/wsp/scripts/pod_decide.sh /local/v4_decide decide_check "100" || { echo "SMOKE FAILED"; exit 1; }
source /local/ovi-env/bin/activate
python3 - <<'PY'
import json
from pathlib import Path
w = Path("/local/v4_decide")
rows = json.loads((w / "eval" / "eval.json").read_text())
want = {"teacher48", "base_unipc12", "base_unipc16", "base_unipc20", "base_unipc30", "base_euler12", "v2_euler12"}
got = {r["variant"] for r in rows if "error" not in r}
print("variants:", sorted(got))
for r in rows:
    print(f"  {r['variant']:14} steps {r.get('steps')} seconds {r.get('seconds')} error {r.get('error')}")
assert got == want, f"missing {sorted(want - got)}"
for name in ("adherence.json", "audio_mp4.json", "noise.json", "ovi_commit.txt", "gpu.txt", "DONE"):
    assert (w / name).exists(), name
print("adherence rows:", len(json.loads((w / "adherence.json").read_text())))
print("noise:", json.dumps({k: {s: v[s]["redraw_cuda_matches"] for s in ("video", "audio")}
                            for k, v in json.loads((w / "noise.json").read_text())["by_seed"].items()}))
print("SMOKE OK")
PY
