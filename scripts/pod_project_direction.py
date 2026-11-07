"""Project measured directions onto Ovi's real weight matrices.

This is the step that makes the work WSP rather than measurement: a captured direction
is scored against every column of the two matrices that write to the residual stream, so
the output is a specific, finite list of weights rather than a description.

Reading weights without loading the model
----------------------------------------
The checkpoint is an 11.66B safetensors file. Loading it onto a GPU to read two matrices
per layer would cost 37GB for a few megabytes of data, so the file is memory-mapped and
only the named tensors are materialised. That also means this can run on a CPU-only box
against a capture produced elsewhere.

Weight naming
-------------
Ovi's module tree is ``model.video_model.blocks[i].self_attn.o`` and
``...blocks[i].ffn.2``. The capture is keyed on (stream, layer), so the lookup is
``f"video_model.blocks.{layer}.self_attn.o"`` inside the checkpoint's prefix. Rather than
assume the prefix, the keys present in the file are inspected once and the prefix is
derived from them; a guessed prefix that matches nothing would report "direction aligned
with no column", which reads as a scientific result rather than a lookup failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RESIDUAL_WRITERS: tuple[tuple[str, str], ...] = (
    ("self_attn", "o"),
    ("ffn", "2"),
)


def open_safetensors(path: Path):
    """Memory-map a safetensors file without loading it.

    Opened with ``framework="torch"``, not ``"numpy"``: Ovi's checkpoint is bfloat16 and
    the numpy framework raises ``TypeError: data type 'bfloat16' not understood`` from
    ``get_tensor`` itself, before any conversion could happen.
    """
    from safetensors.torch import safe_open

    return safe_open(str(path), framework="torch", device="cpu")


def discover_prefix(handle, stream: str, layer: int) -> str:
    """Find the key prefix that addresses ``<stream>_model.blocks.<layer>``.

    Derived from the file rather than assumed. A wrong prefix silently yields no
    weights, and "no aligned column" is indistinguishable from a real negative result.
    """
    needle = f"{stream}_model.blocks.{layer}."
    matches = [key for key in handle.keys() if needle in key]
    if not matches:
        raise KeyError(
            f"no key in the checkpoint contains {needle!r}; "
            f"the checkpoint does not look like an Ovi fusion model"
        )
    # Everything before the needle is the prefix, e.g. "model." or "".
    shortest = min(matches, key=len)
    return shortest[: shortest.index(needle)]


def read_matrix(handle, prefix: str, stream: str, layer: int, attribute: str, path: str):
    """Read one ``[out, in]`` weight matrix, or ``None`` if it is absent.

    Ovi's checkpoint stores bfloat16, which numpy has no dtype for: ``safe_open`` with
    ``framework="numpy"`` raises ``TypeError: data type 'bfloat16' not understood`` from
    ``get_tensor``. The tensor is therefore handed to torch, which widens bfloat16 to
    float32 exactly (bfloat16's exponent range is a subset of float32's).

    Going through torch unconditionally, rather than branching on dtype, keeps one code
    path: a dtype that happened to slip past a check would otherwise only fail later, at
    the matmul.
    """
    key = f"{prefix}{stream}_model.blocks.{layer}.{attribute}.{path}.weight"
    if key not in handle.keys():
        return None, key
    import torch

    tensor = handle.get_tensor(key)
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    return tensor.to(torch.float32).cpu().numpy(), key


def load_directions(path: Path) -> dict[tuple[str, int], np.ndarray]:
    """Reconstruct unit directions from the summarised report.

    The summary deliberately drops the raw vectors to stay small, so directions are
    recomputed here from the means stored alongside. If the report has no vectors, the
    caller must supply the capture instead -- this refuses rather than inventing one.
    """
    data = json.loads(path.read_text())
    directions: dict[tuple[str, int], list[np.ndarray]] = {}
    for row in data.get("rows", []):
        vector = row.get("direction")
        if vector is None:
            continue
        key = (data["stream"], row["layer"])
        directions.setdefault(key, []).append(np.asarray(vector, dtype=np.float32))
    if not directions:
        raise ValueError(
            "the report contains no direction vectors. Derive them with "
            "pod_derive_direction.py's full report rather than the summary."
        )
    # Average the per-step directions at each layer: the question is which columns move
    # the state toward stability in general, not at one particular step.
    return {
        key: (np.mean(vectors, axis=0) / max(np.linalg.norm(np.mean(vectors, axis=0)), 1e-12))
        for key, vectors in directions.items()
    }


def main(argv: list[str] | None = None) -> int:
    from wsp.activations import alignment_spectrum, select_stable_columns

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ratio", type=float, default=0.9)
    parser.add_argument("--depth", type=int, default=10)
    args = parser.parse_args(argv)

    directions = load_directions(args.report)
    print(f"directions for: {sorted(directions)}", flush=True)

    handle = open_safetensors(args.weights)
    findings: list[dict] = []
    for (stream, layer), direction in sorted(directions.items()):
        prefix = discover_prefix(handle, stream, layer)
        for attribute, path in RESIDUAL_WRITERS:
            matrix, key = read_matrix(handle, prefix, stream, layer, attribute, path)
            if matrix is None:
                findings.append(
                    {"stream": stream, "layer": layer, "missing_key": key}
                )
                continue
            if matrix.shape[0] != direction.shape[0]:
                findings.append(
                    {
                        "stream": stream,
                        "layer": layer,
                        "key": key,
                        "shape": list(matrix.shape),
                        "direction_dims": int(direction.shape[0]),
                        "skipped": "width mismatch",
                    }
                )
                continue
            spectrum = alignment_spectrum(matrix, direction, depth=args.depth)
            selection = select_stable_columns(matrix, direction, ratio=args.ratio)
            findings.append(
                {
                    "stream": stream,
                    "layer": layer,
                    "key": key,
                    "shape": list(matrix.shape),
                    "top_columns": [int(i) for i, _ in spectrum],
                    "top_scores": [round(float(s), 6) for _, s in spectrum],
                    "best": round(float(spectrum[0][1]), 6) if spectrum else None,
                    "selected": len(selection.columns),
                    "selection_threshold": round(float(selection.threshold), 6),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(findings, indent=2))
    for row in findings:
        if "best" in row:
            print(
                f"{row['key']:60s} best={row['best']:+.4f} "
                f"selected={row['selected']:3d} {row['shape']}",
                flush=True,
            )
        else:
            print(f"{row}", flush=True)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
