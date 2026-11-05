"""Minimal vLLM load check. Run as a file, not a heredoc.

vLLM spawns its engine core as a subprocess that re-imports the main module.
Under a heredoc that path is '/root/<stdin>', which does not exist, so the spawn
fails with FileNotFoundError before the model ever loads. Running from a real
file avoids it.
"""

from __future__ import annotations

import sys

MODEL = "/local/models/qwen"
PROMPT = "What is 12*11? Answer with just the number."


def main() -> int:
    from vllm import LLM, SamplingParams

    print(f"loading {MODEL}", flush=True)
    llm = LLM(
        model=MODEL,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        disable_log_stats=True,
    )
    out = llm.generate([PROMPT], SamplingParams(max_tokens=128, temperature=0))
    text = out[0].outputs[0].text
    print(f"GEN: {text!r}", flush=True)
    print("VLLM_LOAD_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
