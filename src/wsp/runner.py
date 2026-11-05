"""Model execution for the editor loop.

The Stage 0 ablation showed an editor needs to *observe* behaviour, not just
read weights. This module is what lets it: a runner that generates from a
checkpoint and reports whether the model looped.

vLLM is imported lazily so the module is usable (and testable) without it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: A response that repeats a block this many times is looping.
_REPEAT_WINDOW = 200
_REPEAT_MIN_HITS = 2


@dataclass
class LoopSignal:
    """What a single generation did."""

    looped: bool
    hit_cap: bool
    repeated: bool
    tokens: int
    text_chars: int

    def as_dict(self) -> dict:
        return {
            "looped": self.looped,
            "hit_cap": self.hit_cap,
            "repeated": self.repeated,
            "tokens": self.tokens,
            "text_chars": self.text_chars,
        }


def detect_repetition(text: str, window: int = _REPEAT_WINDOW) -> bool:
    """Whether the tail of ``text`` repeats a block seen earlier.

    A cheap stand-in for the paper's verbatim-loop detector: a response that
    circles repeats a long suffix, and counting occurrences of that suffix
    catches it without the quadratic scan a full period search needs.
    """
    if len(text) < window * 3:
        return False
    tail = text[-window:]
    return text[:-window].count(tail) >= _REPEAT_MIN_HITS


def classify(text: str, n_tokens: int, max_tokens: int) -> LoopSignal:
    """Turn a generation into a loop verdict."""
    hit_cap = n_tokens >= max_tokens
    repeated = detect_repetition(text)
    return LoopSignal(
        looped=hit_cap or repeated,
        hit_cap=hit_cap,
        repeated=repeated,
        tokens=n_tokens,
        text_chars=len(text),
    )


class NullRunner:
    """A runner with no model attached.

    Used for local runs and tests: the tool surface is exercised, but
    generation reports that nothing is loaded rather than returning nothing.
    """

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        raise RuntimeError(
            "no model attached: run this on a pod, or pass a runner that can execute"
        )

    def measure(self, prompt: str, max_tokens: int = 2048) -> dict:
        raise RuntimeError(
            "no model attached: run this on a pod, or pass a runner that can execute"
        )

    def reload(self, model_path: str) -> None:
        raise RuntimeError("no model attached")


class VLLMRunner:
    """Generation backed by vLLM.

    vLLM is 40x faster than the transformers path on this workload, which is
    what makes a 32k-token measurement affordable inside an editor's loop.
    """

    def __init__(
        self,
        model_path: str,
        *,
        max_model_len: int = 36864,
        gpu_memory_utilization: float = 0.85,
        thinking: bool = True,
    ) -> None:
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.thinking = thinking
        self._llm = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        from transformers import AutoTokenizer
        from vllm import LLM

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._llm = LLM(
            model=self.model_path,
            dtype="bfloat16",
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
            seed=0,
        )

    def _render(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        self._ensure_loaded()
        from vllm import SamplingParams

        params = SamplingParams(
            n=1,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            presence_penalty=1.5,
            seed=0,
        )
        out = self._llm.generate([self._render(prompt)], params)[0]
        return out.outputs[0].text

    def measure(self, prompt: str, max_tokens: int = 2048) -> dict:
        self._ensure_loaded()
        from vllm import SamplingParams

        params = SamplingParams(
            n=1,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            presence_penalty=1.5,
            seed=0,
        )
        out = self._llm.generate([self._render(prompt)], params)[0]
        text = out.outputs[0].text
        n_tokens = len(out.outputs[0].token_ids)
        return classify(text, n_tokens, max_tokens).as_dict()

    def reload(self, model_path: str) -> None:
        """Swap to a different checkpoint, e.g. after an edit."""
        self._llm = None
        self._tokenizer = None
        self.model_path = model_path
        self._ensure_loaded()


def write_edited_checkpoint(
    source: str | Path,
    tensor: str,
    edited: "object",
    out_dir: str | Path,
) -> Path:
    """Write a full checkpoint with one tensor replaced.

    vLLM loads from a directory, so an edit has to be materialised as a
    complete checkpoint rather than a single array. The source is copied
    verbatim apart from the edited tensor.
    """
    import shutil

    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file

    source = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy the non-weight files (config, tokenizer) so the directory loads.
    for item in source.parent.iterdir():
        if item.is_file() and item.suffix in {".json", ".txt"}:
            shutil.copy2(item, out_dir / item.name)

    tensors: dict[str, np.ndarray] = {}
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key == tensor:
                tensors[key] = np.asarray(edited, dtype=np.float32)
            else:
                tensors[key] = handle.get_tensor(key).float().numpy()

    # Write under the filename the index expects. A sharded checkpoint's
    # model.safetensors.index.json names its shards explicitly, so writing
    # "model.safetensors" leaves the index pointing at a file that is not there.
    target_name = _shard_name_for(source, out_dir)
    save_file(tensors, str(out_dir / target_name))

    # Drop any stale index: the single written file now holds every tensor.
    index = out_dir / "model.safetensors.index.json"
    if index.exists():
        index.unlink()

    return out_dir


def _shard_name_for(source: Path, out_dir: Path) -> str:
    """The filename the checkpoint's index expects, or a safe default."""
    index = out_dir / "model.safetensors.index.json"
    if index.exists():
        try:
            import json

            data = json.loads(index.read_text(encoding="utf-8"))
            shards = set((data.get("weight_map") or {}).values())
            if len(shards) == 1:
                return shards.pop()
        except Exception:  # noqa: BLE001 - fall through to the default
            pass
    return source.name
