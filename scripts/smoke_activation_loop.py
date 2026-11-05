"""Smoke test the activation-driven edit loop end to end on CPU.

Builds a tiny Qwen model, then drives the real tool surface through the real
agent loop with a scripted editor. This proves the wiring works without a GPU:
capture a direction, rank columns, apply a directional edit, and confirm the
checkpoint on disk changed.

Usage:
    python scripts/smoke_activation_loop.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wsp.agent import run_agent  # noqa: E402
from wsp.audit import audit_run  # noqa: E402
from wsp.editors.base import Editor, EditorResponse, Usage  # noqa: E402
from wsp.runner import TransformersActivationCapturer  # noqa: E402
from wsp.tools import WeightTools  # noqa: E402
from wsp.weights.store import WeightStore  # noqa: E402

TENSOR = "model.layers.0.mlp.down_proj.weight"


class ScriptedEditor(Editor):
    """Replies with a fixed script, so the loop is deterministic."""

    model_id = "smoke/scripted"

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    async def complete(self, system, user, *, temperature=0.0, max_tokens=8192):
        self.prompts.append(user)
        text = self._replies.pop(0) if self._replies else "DONE\nout of replies"
        return EditorResponse(text=text, usage=Usage(), model=self.model_id)


class _Runner:
    """Records reloads; generation is stubbed since this runs on CPU."""

    def __init__(self) -> None:
        self.reloaded: list[str] = []

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        return "generated"

    def measure(self, prompt: str, max_tokens: int = 2048) -> dict:
        return {"looped": True, "tokens": 100, "hit_cap": False}

    def reload(self, model_path: str) -> None:
        self.reloaded.append(model_path)


def _build_model(root: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    Qwen3ForCausalLM(config).save_pretrained(str(root))

    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    vocab = {f"tok{i}": i for i in range(126)}
    vocab["[UNK]"] = 126
    vocab["[PAD]"] = 127
    tokenizer.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        '{"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]",'
        ' "pad_token": "[PAD]"}',
        encoding="utf-8",
    )
    return root


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = _build_model(Path(tmp) / "model")
        shard = sorted(root.glob("*.safetensors"))[0]

        store = WeightStore(shard)
        runner = _Runner()
        capturer = TransformersActivationCapturer(str(root), device="cpu")
        tools = WeightTools(store, runner, capturer=capturer, max_forward_passes=20)

        original = store.get(TENSOR).copy()

        # Build the argument objects with json.dumps: a Windows path contains
        # backslashes, which are invalid JSON escapes when pasted raw.
        import json

        def call(name: str, args: dict) -> str:
            return f"CALL {name}\n{json.dumps(args)}"

        editor = ScriptedEditor(
            [
                call(
                    "capture_direction",
                    {
                        "name": "loopdir",
                        "tensor": TENSOR,
                        "looping_prompt": "tok1 tok2 tok3",
                        "control_prompt": "tok4 tok5 tok6",
                    },
                ),
                call(
                    "aligned_columns",
                    {"tensor": TENSOR, "direction": "loopdir", "k": 4},
                ),
                call(
                    "apply_directional_edit",
                    {
                        "tensor": TENSOR,
                        "direction": "loopdir",
                        "cols": "0:4",
                        "scale": 0.5,
                        "out_path": str(Path(tmp) / "edited"),
                    },
                ),
                "DONE\ncaptured a direction, ranked columns, applied the edit.",
            ]
        )

        run = asyncio.run(run_agent(editor, tools, "stop the loop", max_turns=8))

        print(f"turns: {run.turns}")
        print(f"forward passes: {run.forward_passes}")
        print("\ntool calls:")
        for call in run.tool_log.calls:
            print(f"  - {call.name} ok={call.ok} -> {call.result_summary}")

        print(f"\nconclusion:\n{run.conclusion}")

        payload = run.as_dict()
        audit = audit_run(payload)
        print(f"\naudit: {audit.summary()}")

        edited = store.get(TENSOR)
        changed = int(np.count_nonzero(~np.isclose(edited, original, rtol=0.0, atol=0.0)))
        print(f"\nweights changed on disk: {changed}")
        print(f"runner reloaded: {bool(runner.reloaded)}")

        failures = []
        if not audit.clean:
            failures.append(f"audit not clean: {audit.summary()}")
        if changed == 0:
            failures.append("edit did not change the checkpoint")
        if not runner.reloaded:
            failures.append("runner was never reloaded")
        names = [c.name for c in run.tool_log.calls]
        for expected in ("capture_direction", "aligned_columns", "apply_directional_edit"):
            if expected not in names:
                failures.append(f"{expected} was never called")

        if failures:
            print("\nFAILED:")
            for failure in failures:
                print(f"  - {failure}")
            return 1

        print("\nOK: capture -> rank -> directional edit -> reload all worked.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
