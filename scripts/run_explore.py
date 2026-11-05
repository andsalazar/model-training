"""Run the editor loop on a pod, with a live vLLM runner.

Usage:
    python run_explore.py --editor deepseek --model Qwen/Qwen3.5-0.8B \
        --checkpoint /workspace/models/Qwen3.5-0.8B/model.safetensors \
        --out /workspace/explore.json

The editor gets the tool surface from wsp.tools, including run_model and
measure_loop, which is what the Stage 0 ablation showed was necessary.
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
from wsp.tools import WeightTools
from wsp.weights.store import WeightStore

TASK = (
    "The model is Qwen3.5-0.8B. In thinking mode it enters reasoning loops on "
    "math problems and runs to the token limit without reaching an answer.\n\n"
    "Find the minimal weight edit that stops the looping. The candidate tensor "
    "is the MLP down-projection of a decoder layer, shape [1024, 3584].\n\n"
    "Use measure_loop to confirm the loop exists before proposing anything, and "
    "again after any edit to confirm it is fixed. Use tensor_stats to find "
    "candidate columns. Do not propose an edit you have not tested.\n\n"
    "Weight statistics alone cannot tell you which column causes the loop. "
    "capture_direction compares the module's activations on a looping prompt "
    "against a control prompt and gives you the direction driving the loop; "
    "aligned_columns then shows which columns write along it. Prefer that over "
    "guessing from column norms.\n\n"
    "The control prompt matters. A prose control against a math prompt conflates "
    "looping with 'reasoning vs prose', which dilutes the direction: explore9 "
    "measured contrast_cosine 0.46-0.72 that way and its edits did not stop the "
    "loop. Use a control that is itself a math problem the model completes "
    "without looping, and keep the two prompts similar in length. A "
    "contrast_cosine near 1.0 means the contrast found nothing; a low one means "
    "the two states genuinely differ. Capture directions at more than one layer "
    "and compare their contrast_cosine before choosing where to edit."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--editor", default="deepseek")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="/workspace/explore.json")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-forward-passes", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    print(f"editor: {args.editor}", flush=True)
    print(f"model:  {args.model}", flush=True)
    print(f"budget: {args.max_turns} turns, {args.max_forward_passes} forward passes", flush=True)

    runner = VLLMRunner(args.model)
    store = WeightStore(args.checkpoint)
    # Capture runs on transformers: vLLM's forward pass is inside compiled CUDA
    # graphs, so there is nowhere to attach a hook. Generation stays on vLLM.
    capturer = TransformersActivationCapturer(args.model)
    tools = WeightTools(
        store,
        runner,
        max_forward_passes=args.max_forward_passes,
        capturer=capturer,
    )

    async def _run():
        async with build_editor(args.editor, api_key) as client:
            return await run_agent(
                client, tools, TASK, max_turns=args.max_turns, max_tokens=args.max_tokens
            )

    run = asyncio.run(_run())

    print(f"\nturns: {run.turns}")
    print(f"forward passes: {run.forward_passes}")
    print(f"tool calls: {len(run.tool_log.calls)}")
    for call in run.tool_log.calls:
        print(f"  - {call.name}({call.arguments}) -> {call.result_summary}")

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

    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
