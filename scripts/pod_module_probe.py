"""Find the real module path for a checkpoint tensor key in the loaded model.

Checkpoint keys and module attribute paths are not always the same string: a
safetensors key carries the full save-time prefix, while the loaded model may
expose the same submodule under a shorter attribute path. This prints both so
the capture script can use the path the model actually has.
"""

from __future__ import annotations

MODEL = "/workspace/qwen"
TENSOR = "model.language_model.layers.23.mlp.down_proj.weight"


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    print("model class:", type(model).__name__)

    # Every module whose name ends in the target suffix.
    hits = [n for n, _ in model.named_modules() if n.endswith("layers.23.mlp.down_proj")]
    print("\nmodules ending in layers.23.mlp.down_proj:")
    for h in hits:
        print("  ", h)

    # The top-level children, to see the prefix structure.
    print("\ntop-level children:")
    for n, _ in model.named_children():
        print("  ", n)

    # Confirm the weight shape at each candidate.
    for h in hits:
        mod = model.get_submodule(h)
        w = getattr(mod, "weight", None)
        print(f"\n{h}: weight {tuple(w.shape) if w is not None else None}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
