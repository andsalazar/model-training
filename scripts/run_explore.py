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
    "The model is Qwen3.5-0.8B. In thinking mode it over-thinks trivial math "
    "questions: it reaches the correct answer and then keeps re-verifying it, "
    "sometimes running to the token limit. Measured on this checkpoint, "
    "'What is 12 times 11? Reply with only the digits.' produced 2048 tokens of "
    "self-doubt while the correct answer 132 was already present.\n\n"
    "Find the minimal weight edit that reduces this over-thinking without "
    "breaking correctness. The candidate tensor is the MLP down-projection of a "
    "decoder layer, shape [1024, 3584].\n\n"
    "IMPORTANT: hitting the token cap is NOT the same as looping. measure_loop "
    "reports hit_cap and repeated separately; only repeated means the model is "
    "circling. A verbose answer that reaches the cap and stops is truncation. "
    "Use a large cap (2048) so a terminating answer has room to finish.\n\n"
    "Use measure_loop to establish the baseline before proposing anything, and "
    "again after any edit to confirm the behaviour changed. Use tensor_stats to "
    "find candidate columns. Do not propose an edit you have not tested.\n\n"
    "Weight statistics alone cannot tell you which column drives the behaviour. "
    "capture_direction compares the module's activations on an over-thinking "
    "prompt against a control prompt and gives you the direction driving it; "
    "aligned_columns then shows which columns write along that direction. Prefer "
    "that over guessing from column norms.\n\n"
    "The control prompt matters. Use the SAME question phrased to elicit a short "
    "answer -- 'What is 12 times 11? Answer with just the number.' completes in "
    "471 tokens, while 'Reply with only the digits.' runs to 2048. That is a "
    "genuine contrast between two states of the same question, not between math "
    "and prose.\n\n"
    "A high contrast_cosine is a caution, not a verdict. It compares the two "
    "mean states, and a prefix both prompts share appears in both means and "
    "cancels when they are subtracted -- so the direction can be real even when "
    "the cosine is near 1.0. Do not abandon a direction on the cosine alone. "
    "Test it: capture the same direction at another layer or with a token_window "
    "past the shared prefix, then call direction_similarity. Agreement means the "
    "component is reproducible and worth acting on; disagreement means it is "
    "noise. Only then choose where to edit.\n\n"
    "Capture past truncation. The over-thinking prompt runs to the 2048 cap, so "
    "its mean state is a truncated state: the direction partly encodes where "
    "generation was cut off rather than where it stopped naturally. Pass a "
    "larger max_tokens to capture_direction (4096) so the looping run can reach "
    "its own stop. If the ranked columns change, the truncation was distorting "
    "the direction and the larger cap is the one to use.\n\n"
    "Edit the columns aligned_columns ranked, not a contiguous window around "
    "them. The ranked columns are scattered across the matrix, and a range that "
    "happens to contain a few of them also edits dozens of columns the evidence "
    "did not select. apply_directional_edit takes either form: 'start:stop' or a "
    "comma-separated list like '102,987,914'. Use the list.\n\n"
    "A single prompt improving is not evidence the edit works. Call baseline "
    "before you edit and verify_edit after: verify_edit re-measures the same "
    "probes on the current checkpoint and reports VALIDATED only when every "
    "target improved and nothing else was damaged. A target that improves while "
    "a held-out question regresses is reported as HARMFUL, because shortening "
    "one answer at the cost of pushing another to the token cap is redistributing "
    "the behaviour, not suppressing it. If verify_edit says HARMFUL or NO EFFECT, "
    "the edit did not work -- try a different layer or scale rather than "
    "reporting it as a result."
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
