#!/bin/bash
# Per-array sizes in a capture, so an unexpectedly large file is explained rather than
# guessed at. A 4-step, 1-layer capture came to 713MB, which is far more than the
# 20.8MB-per-video-capture arithmetic predicted for stride 8.
set -uo pipefail
source /local/ovi-env/bin/activate
CAP="${1:-/local/captures/diag_hooked.npz}"
python3 - "$CAP" <<'PY'
import sys

import numpy as np

with np.load(sys.argv[1]) as data:
    total = 0
    for key in sorted(data.files):
        array = data[key]
        total += array.nbytes
        print(f"{key:36s} {str(array.shape):24s} {str(array.dtype):10s} {array.nbytes/1e6:9.1f} MB")
    print(f"{'TOTAL':36s} {'':24s} {'':10s} {total/1e6:9.1f} MB")
PY
