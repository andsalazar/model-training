"""Read-only access to safetensors checkpoints.

Loads tensors lazily by name so a 1.6GB checkpoint does not have to be resident
to inspect one matrix. Writes always go to a new file; the source is never
modified in place.

The torch backend is used rather than numpy because released checkpoints are
stored in bfloat16, which numpy cannot represent. Tensors are converted to
float32 on read, which is lossless for bf16 and adequate for inspection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors import safe_open


@dataclass(frozen=True)
class TensorInfo:
    """Metadata for one tensor in a checkpoint."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def numel(self) -> int:
        n = 1
        for dim in self.shape:
            n *= dim
        return n


class WeightStore:
    """A safetensors checkpoint opened for reading."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.path}")
        self._handle = safe_open(str(self.path), framework="pt", device="cpu")
        self._keys = list(self._handle.keys())

    def __enter__(self) -> "WeightStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._handle = None  # type: ignore[assignment]

    @property
    def keys(self) -> list[str]:
        return list(self._keys)

    def info(self, name: str) -> TensorInfo:
        if name not in self._keys:
            raise KeyError(f"{name!r} not in checkpoint ({len(self._keys)} tensors)")
        # Read the header only; get_slice avoids materialising the tensor.
        shape = tuple(self._handle.get_slice(name).get_shape())
        dtype = self._handle.get_slice(name).get_dtype()
        return TensorInfo(name=name, shape=shape, dtype=str(dtype))

    def get(self, name: str) -> np.ndarray:
        """Load one tensor as a float32 array."""
        if name not in self._keys:
            raise KeyError(f"{name!r} not in checkpoint")
        tensor = self._handle.get_tensor(name)
        return tensor.to("cpu", dtype=None).float().numpy()

    def summary(self) -> dict[str, object]:
        """Shape and dtype for every tensor, without loading the data."""
        tensors = [self.info(k) for k in self._keys]
        return {
            "path": str(self.path),
            "num_tensors": len(tensors),
            "num_params": sum(t.numel for t in tensors),
            "tensors": [
                {"name": t.name, "shape": list(t.shape), "dtype": t.dtype} for t in tensors
            ],
        }

    def write_summary(self, out: str | Path) -> Path:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return out
