"""Can a TBC-style adapter be trained on Ovi on one A100-80GB? A memory and time probe.

What is measured
----------------
One optimiser step of a small residual adapter inserted before the first fusion blocks of
both streams, with Ovi frozen. The adapter sits early, so the gradient has to flow back
through every later block: the cost is a full forward and backward through the 11.66B
model, which is what any adapter-training objective will pay per sample.

Three configurations, cheapest-to-fit first so an OOM in the last cannot hide the others:

* 5s token count (31 latent frames, 27,900 video tokens, 157 audio) with gradient
  checkpointing,
* 10s token count (61 latent frames, 54,900 video tokens, 314 audio) with checkpointing,
* 5s without checkpointing, which is expected to run out of memory.

The 10s case runs on the 5s checkpoint's weights. The two checkpoints share one
architecture, so memory and time are the same; only the values differ.

The adapter
-----------
TBC describe a ~156k-parameter residual module in early DiT layers. This uses
``x + up(gelu(down(layernorm(x))))`` with ``up`` zero-initialised, so the adapted model
starts identical to the base model. Rank 8 at 3 blocks per stream is ~300k parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

OVI_ROOT = Path(os.environ.get("OVI_ROOT", "/local/Ovi"))
if str(OVI_ROOT) not in sys.path:
    sys.path.insert(0, str(OVI_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapter training memory/time probe.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--blocks", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--timestep", type=float, default=768.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["5s:ckpt", "10s:ckpt", "5s:nockpt"],
        help="horizon:ckpt|nockpt pairs, run in order",
    )
    return parser.parse_args(argv)


#: Latent shapes from ovi_fusion_engine's model specs.
LATENT = {"5s": (31, 157), "10s": (61, 314)}


def build_adapter(dim: int, rank: int):
    import torch
    from torch import nn

    class ResidualAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(dim, elementwise_affine=False)
            self.down = nn.Linear(dim, rank)
            self.up = nn.Linear(rank, dim)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

        def forward(self, x):
            h = self.up(torch.nn.functional.gelu(self.down(self.norm(x.float()))))
            return x + h.to(x.dtype)

    return ResidualAdapter()


def install_adapters(model, blocks: list[int], rank: int):
    """Wrap the fused block forward so adapters act on the residual stream before block i.

    Ovi's fused loop calls ``self.single_fusion_block_forward`` by attribute, through
    ``gradient_checkpointing``, so an instance attribute replaces it for every block.
    """
    from torch import nn

    vdim = model.video_model.blocks[0].dim
    adim = model.audio_model.blocks[0].dim
    adapters = nn.ModuleDict()
    for i in blocks:
        adapters[f"video_{i}"] = build_adapter(vdim, rank)
        adapters[f"audio_{i}"] = build_adapter(adim, rank)
    index = {id(b): i for i, b in enumerate(model.video_model.blocks)}
    original = model.single_fusion_block_forward

    def wrapped(*, vid_block, audio_block, vid, audio, **kwargs):
        i = index[id(vid_block)]
        if f"video_{i}" in adapters:
            vid = adapters[f"video_{i}"](vid)
            audio = adapters[f"audio_{i}"](audio)
        return original(
            vid_block=vid_block, audio_block=audio_block, vid=vid, audio=audio, **kwargs
        )

    model.single_fusion_block_forward = wrapped
    return adapters


def run_case(engine, adapters, optimiser, embeddings, horizon: str, ckpt: bool, args):
    import torch

    model = engine.model
    device = next(model.parameters()).device
    frames, audio_len = LATENT[horizon]
    h = w = 60
    patch = model.video_model.patch_size
    vid_seq = frames * h * w // (patch[1] * patch[2])
    gen = torch.Generator(device=device).manual_seed(0)
    vid = torch.randn(
        (engine.video_latent_channel, frames, h, w),
        device=device, dtype=torch.bfloat16, generator=gen,
    )
    audio = torch.randn(
        (audio_len, engine.audio_latent_channel),
        device=device, dtype=torch.bfloat16, generator=gen,
    )
    t = torch.full((1,), args.timestep, device=device)
    pos = embeddings[0]

    model.gradient_checkpointing = ckpt
    model.train(ckpt)

    def forward():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return model(
                vid=[vid], audio=[audio], t=t,
                vid_context=[pos], audio_context=[pos],
                vid_seq_len=vid_seq, audio_seq_len=audio_len,
                first_frame_is_clean=False,
            )

    result: dict = {"horizon": horizon, "checkpointing": ckpt, "video_tokens": vid_seq,
                    "audio_tokens": audio_len}
    torch.cuda.synchronize()
    with torch.no_grad():
        started = time.time()
        forward()
        torch.cuda.synchronize()
        result["forward_only_s"] = round(time.time() - started, 2)

    steps = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.repeats):
        optimiser.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.time()
        pred_vid, pred_audio = forward()
        loss = pred_vid[0].float().pow(2).mean() + pred_audio[0].float().pow(2).mean()
        loss.backward()
        optimiser.step()
        torch.cuda.synchronize()
        steps.append(round(time.time() - started, 2))
    grads = {
        name: float(p.grad.norm()) if p.grad is not None else None
        for name, p in adapters.named_parameters()
        if name.endswith("up.weight")
    }
    result.update(
        {
            "train_step_s": steps,
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
            "loss": float(loss),
            "up_weight_grad_norms": grads,
            "all_grads_nonzero": all(g and g > 0 for g in grads.values()),
        }
    )
    model.train(False)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch
    from omegaconf import OmegaConf
    from ovi.ovi_fusion_engine import OviFusionEngine

    from wsp.dit import ovi_sampler_kwargs

    config = OmegaConf.load(args.config)
    engine = OviFusionEngine(config=config, device=0)
    model = engine.model
    for p in model.parameters():
        p.requires_grad_(False)
    adapters = install_adapters(model, args.blocks, args.rank).to(
        next(model.parameters()).device
    )
    n_params = sum(p.numel() for p in adapters.parameters())
    optimiser = torch.optim.AdamW(adapters.parameters(), lr=1e-4)

    sampler = ovi_sampler_kwargs(config)
    with torch.no_grad():
        embeddings = engine.text_model(
            ["A river flowing through a forest, camera slowly panning. Audio: gentle "
             "water sounds and birdsong.",
             sampler["video_negative_prompt"], sampler["audio_negative_prompt"]],
            engine.text_model.device,
        )
    embeddings = [e.to(torch.bfloat16).to(next(model.parameters()).device) for e in embeddings]

    payload: dict = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "after_load_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "adapter_blocks": args.blocks,
        "adapter_rank": args.rank,
        "adapter_params": n_params,
        "cases": [],
    }
    print(json.dumps(payload, indent=2), flush=True)

    for case in args.cases:
        horizon, _, mode = case.partition(":")
        print(f"\n=== {horizon} checkpointing={mode == 'ckpt'} ===", flush=True)
        try:
            row = run_case(engine, adapters, optimiser, embeddings, horizon,
                           mode == "ckpt", args)
        except torch.cuda.OutOfMemoryError as exc:
            row = {"horizon": horizon, "checkpointing": mode == "ckpt",
                   "error": f"OOM: {str(exc).splitlines()[0]}",
                   "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - record and continue
            row = {"horizon": horizon, "checkpointing": mode == "ckpt",
                   "error": f"{type(exc).__name__}: {exc}"}
            torch.cuda.empty_cache()
        print(json.dumps(row, indent=2), flush=True)
        payload["cases"].append(row)
        args.out.write_text(json.dumps(payload, indent=2))

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
