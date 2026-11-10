"""Record the starting noise Ovi actually used for each teacher run, as hashes.

Stage 7 v3 found that the same prompt, seed and settings give a different clip on an A100
than on an H100, and did not test why. One candidate is that the starting noise itself
differs between GPU types: Ovi draws it with a generator on the GPU.

A teacher file holds the state at every student timestep, and the first of those is the
sample entering the first step, which is the starting noise. This hashes it per seed, so
two pods can be compared without moving tensors. It also redraws noise of the same shape
on this GPU and on the CPU, to check that ``Trainer.initial_noise`` reproduces Ovi's draw
and to show how far a CPU draw is from a GPU one.

This records; it does not settle the question. That needs the same file from a second
GPU type.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def describe(tensor: torch.Tensor) -> dict:
    data = tensor.float().cpu().contiguous()
    return {
        "shape": list(data.shape),
        "sha256": hashlib.sha256(data.numpy().tobytes()).hexdigest(),
        "first": [round(float(x), 6) for x in data.flatten()[:6]],
        "std": round(float(data.std()), 6),
    }


def redraw(shape: tuple[int, ...], seed: int, device: str) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    cuda = torch.cuda.is_available()
    out: dict = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if cuda else None,
                 "by_seed": {}}
    for path in sorted(args.teacher_dir.glob("*.pt")):
        traj = torch.load(path, map_location="cpu")
        seed = str(traj["seed"])
        entry = out["by_seed"].setdefault(seed, {"files": []})
        entry["files"].append(path.name)
        for stream in ("video", "audio"):
            used = traj[stream][0]
            record = describe(used)
            previous = entry.get(stream)
            # Ovi seeds the draw only by the seed, so every prompt at one seed must agree.
            record["same_for_every_prompt"] = previous is None or (
                previous["sha256"] == record["sha256"] and previous["same_for_every_prompt"])
            for device in ["cpu"] + (["cuda"] if cuda else []):
                again = redraw(tuple(used.shape), int(seed), device).float().cpu()
                record[f"redraw_{device}_matches"] = bool(torch.equal(again, used.float()))
            entry[stream] = record
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    for seed, entry in out["by_seed"].items():
        v = entry["video"]
        print(f"seed {seed}: {len(entry['files'])} files, video {v['shape']} sha256 {v['sha256'][:16]}, "
              f"same for every prompt {v['same_for_every_prompt']}, "
              f"redraw matches: cpu {v['redraw_cpu_matches']}, cuda {v.get('redraw_cuda_matches')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
