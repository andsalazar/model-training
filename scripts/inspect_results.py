"""Inspect a MATH-500 results file: score plus sample-level detail."""

from __future__ import annotations

import json
import sys


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"model:    {data.get('model')}")
    print(f"accuracy: {data.get('accuracy_pct')} ({data['correct']}/{data['total']})")
    print()

    for s in data["samples"][:n]:
        print(f"--- sample {s['index']} correct={s['correct']} ---")
        print(f"gold: {s['gold'][:80]}")
        print(f"pred: {str(s['prediction'])[:80]}")
        tail = s["response"][-260:].replace("\n", " ")
        print(f"tail: ...{tail}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
