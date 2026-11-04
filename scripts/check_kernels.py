"""Report which optimized kernels are importable and whether transformers uses them."""

from __future__ import annotations

import importlib


def check(module: str) -> str:
    try:
        mod = importlib.import_module(module)
        return f"OK  {getattr(mod, '__version__', '?')}"
    except Exception as exc:  # noqa: BLE001 - report any import failure
        return f"MISSING  ({type(exc).__name__}: {exc})"


def main() -> int:
    for name in ("fla", "causal_conv1d", "flash_attn"):
        print(f"{name:20s} {check(name)}")

    print()
    try:
        from transformers.utils import is_flash_linear_attention_available

        print("transformers sees fla:", is_flash_linear_attention_available())
    except Exception as exc:  # noqa: BLE001
        print("transformers fla probe failed:", exc)

    try:
        from transformers.utils import is_causal_conv1d_available

        print("transformers sees causal_conv1d:", is_causal_conv1d_available())
    except Exception as exc:  # noqa: BLE001
        print("transformers causal_conv1d probe failed:", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
