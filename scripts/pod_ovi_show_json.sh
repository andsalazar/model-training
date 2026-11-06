#!/usr/bin/env bash
# Print a JSON report from the pod. Takes the path as $1.
#
# Exists because nested quotes in a PowerShell -Command string get mangled by the
# remote shell; a script file sidesteps the quoting entirely.
set -u
python3 - "$1" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1]))
print(json.dumps(data, indent=2)[:2500])
PY
