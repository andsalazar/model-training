"""Extract the decisive evidence from an explore run, so it can be checked.

Prints only what a claim rests on: every apply_edit and measure_loop, in order,
with their real results. Run on the pod:

    /workspace/venv2/bin/python /workspace/scripts/inspect_run.py /workspace/explore6.json
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/explore6.json"
    data = json.loads(open(path, encoding="utf-8").read())

    print(f"editor:         {data.get('editor')}")
    print(f"turns:          {data.get('turns')}")
    print(f"forward passes: {data.get('forward_passes')}")
    print(f"fabricated:     {data.get('fabricated')}")
    print(f"audit:          {json.dumps(data.get('audit'))}")
    print()

    print("=== tool calls, in order ===")
    for i, call in enumerate(data.get("tool_calls", [])):
        name = call["name"]
        ok = call.get("ok", True)
        if name in ("apply_edit", "measure_loop", "run_model") or not ok:
            args = json.dumps(call.get("args", {}))
            result = str(call.get("result", ""))
            flag = "" if ok else "  [FAILED]"
            print(f"{i:3d} {name}{flag}  {args}")
            print(f"     -> {result[:400]}")
    print()

    print("=== editor turns that mention edits ===")
    for entry in data.get("transcript", []):
        if entry.get("role") != "editor":
            continue
        text = entry.get("text") or ""
        if "apply_edit" in text or "measure_loop" in text:
            print(f"--- turn {entry.get('turn')} ---")
            print(text[:1200])
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
