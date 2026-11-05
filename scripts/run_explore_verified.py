"""Run the editor, then re-measure its result outside the editor.

The Stage 1 lesson: an editor's conclusion is a claim, not a result. Both runs
that got far enough to matter fabricated numbers -- explore13 inverted a
direction_similarity sign and explore14 reported post-edit token counts without
ever calling measure_loop after its edits. Auditing the transcript catches some
of that, but the reliable check is to measure the checkpoint again.

This script does both in one pass:

1. run the editor loop exactly as ``run_explore.py`` does
2. read the edits it applied from the tool log
3. re-measure the probe set on each edited checkpoint, in a fresh process
4. write a comparison the editor's prose cannot influence

Step 3 is the part that matters. It uses ``wsp.verify`` against the same vLLM
runner, so the numbers come from the model rather than from the transcript.

Usage:
    python run_explore_verified.py --editor deepseek --model Qwen/Qwen3.5-0.8B \
        --checkpoint /workspace/models/Qwen3.5-0.8B/model.safetensors \
        --out /workspace/explore15.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

from wsp.agent import run_agent
from wsp.audit import audit_run
from wsp.editors.registry import build_editor
from wsp.runner import TransformersActivationCapturer, VLLMRunner
from wsp.tools import _default_probes
from wsp.verify import compare, measure_all
from wsp.weights.store import WeightStore

from run_explore import TASK


def _edited_checkpoints(tool_calls: list[dict]) -> list[str]:
    """Checkpoint directories the editor actually wrote, in order, no repeats.

    ``apply_directional_edit`` reports the directory it wrote, not the file, so
    the caller resolves the ``*.safetensors`` inside it. Only successful calls
    count: a rejected edit changed nothing and must not be measured as if it had.
    """
    seen: list[str] = []
    for call in tool_calls:
        if call.get("name") != "apply_directional_edit":
            continue
        if not call.get("ok", True):
            continue
        result = str(call.get("result", ""))
        for token in result.replace("\n", " ").split():
            token = token.rstrip(".,;")
            if token.startswith("/") and "reloaded" not in token:
                if token not in seen:
                    seen.append(token)
                break
    return seen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--editor", default="deepseek")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="/workspace/explore_verified.json")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-forward-passes", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--probes",
        default="",
        help="comma-separated probe names; empty means the default set",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    print(f"editor: {args.editor}", flush=True)
    print(f"model:  {args.model}", flush=True)

    runner = VLLMRunner(args.model)
    store = WeightStore(args.checkpoint)
    capturer = TransformersActivationCapturer(args.model)
    from wsp.tools import WeightTools

    tools = WeightTools(
        store,
        runner,
        max_forward_passes=args.max_forward_passes,
        capturer=capturer,
    )

    async def _run():
        async with build_editor(args.editor, api_key) as client:
            return await run_agent(
                client,
                tools,
                TASK,
                max_turns=args.max_turns,
                max_tokens=args.max_tokens,
            )

    run = asyncio.run(_run())

    print(f"\nturns: {run.turns}")
    print(f"forward passes: {run.forward_passes}")
    print(f"tool calls: {len(run.tool_log.calls)}")
    print(f"\nconclusion:\n{run.conclusion}")

    payload = run.as_dict()
    audit = audit_run(payload)
    payload["audit"] = {
        "clean": audit.clean,
        "findings": [
            {"kind": f.kind, "claim": f.claim, "reason": f.reason}
            for f in audit.findings
        ],
    }
    print(f"\naudit: {audit.summary()}")

    # -- independent re-measurement ---------------------------------------
    probes = _default_probes()
    edited = _edited_checkpoints(payload["tool_calls"])
    print(f"\nedited checkpoints: {edited or 'none'}")

    verification: dict = {"probes": [p.name for p in probes], "edits": []}
    if edited:
        print("\nmeasuring baseline for comparison...", flush=True)
        baseline = measure_all(runner, probes)
        for path in edited:
            print(f"\nre-measuring {path}...", flush=True)
            try:
                runner.reload(path)
            except Exception as exc:  # noqa: BLE001 - report, do not crash
                print(f"  reload failed: {exc}", flush=True)
                verification["edits"].append(
                    {"checkpoint": path, "error": f"reload failed: {exc}"}
                )
                continue
            if not Path(path).exists():
                print(f"  {path} does not exist", flush=True)
                verification["edits"].append(
                    {"checkpoint": path, "error": "checkpoint directory missing"}
                )
                continue
            after = measure_all(runner, probes)
            comparison = compare(baseline, after)
            print(comparison.summary(), flush=True)
            verification["edits"].append(
                {"checkpoint": path, "comparison": comparison.as_dict()}
            )
    else:
        print("the editor applied no edits, so there is nothing to verify")

    payload["verification"] = verification
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
