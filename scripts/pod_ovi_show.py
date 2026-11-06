"""Print the key numbers from a metric_real.json report, one block per video."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/ovi_out/metric_real.json")
    data = json.loads(path.read_text())

    for v in data["videos"]:
        pf = v["real_per_frame"]
        dl = v["real_delta"]
        print(f"\n{v['video']}")
        print(f"  frames {v['frames']}  patches {v['patches']}  dims {v['dims']}")
        print(
            f"  per-frame  early {pf['early_mean']:>10.2f}  late {pf['late_mean']:>10.2f}  "
            f"ratio {pf['ratio']:>7.3f}  defined {pf['ratio_defined']}  "
            f"r2 {pf['r_squared']:.3f}  collapsing {pf['collapsing']}"
        )
        print(
            f"  delta      early {dl['early_mean']:>10.2f}  late {dl['late_mean']:>10.2f}  "
            f"ratio {dl['ratio']:>7.3f}  defined {dl['ratio_defined']}  "
            f"r2 {dl['r_squared']:.3f}  collapsing {dl['collapsing']}"
        )
        for c in v["constructed"]:
            print(
                f"    s={c['strength']:.2f}  pf sep={str(c['per_frame']['separates']):<5} "
                f"gap={c['per_frame']['gap']:>+8.4f}  |  dl sep={str(c['delta']['separates']):<5} "
                f"gap={c['delta']['gap']:>+8.4f}"
            )

    print("\nsummary")
    for k, val in data.get("summary", {}).items():
        print(f"  {k}: {val}")


if __name__ == "__main__":
    main()
