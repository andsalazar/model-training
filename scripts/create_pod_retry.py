"""Create the Stage 7 pod as soon as any GPU on the priority list can be placed.

Large-GPU stock is listed as "Low" but repeatedly refused at create time, so this
retries one REST v1 create with a *priority list* of GPU types (``gpuTypePriority:
custom`` tries them in order) until one is placed or the time budget runs out. It stops
after the first success, so it can never create more than one pod.

The key comes from ``RUNPOD_MODEL_TRAINING_API_KEY`` (inject with ``doppler run``) and is
never printed. The created pod is recorded in ``runs/pods.json`` so the ownership guard
in ``wsp.runpod`` treats it as this project's.

    doppler run --project model-training --config dev -- python scripts/create_pod_retry.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

API = "https://rest.runpod.io/v1"
#: The user's order: A100 (planned), H100, H200, then the Blackwell RTX PRO 6000.
GPUS = [
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 PCIe",
    "NVIDIA H200",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
]
#: Hourly Secure Cloud prices from this session's catalog reads, for the log line only.
PRICES = {"NVIDIA A100-SXM4-80GB": 1.79, "NVIDIA A100 80GB PCIe": 1.79, "NVIDIA H100 80GB HBM3": 3.99,
          "NVIDIA H100 PCIe": 2.89, "NVIDIA H200": 5.29, "NVIDIA RTX PRO 6000 Blackwell Server Edition": 2.49}
#: No CUDA list: v1's allowedCudaVersions enum stops at 13.0, but every host in this
#: session's catalog reads reports 13.2, so a list would exclude all of them. A 13.2
#: driver runs this CUDA 12.8 image.


def body(public_key: str, disk: int, name: str = "wsp-stage7-v3", gpus: list[str] | None = None) -> dict:
    return {
        "name": name,
        "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "cloudType": "SECURE",
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": gpus or GPUS,
        "gpuTypePriority": "custom",
        "containerDiskInGb": disk,
        "volumeInGb": 0,
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": public_key},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public-key", required=True)
    p.add_argument("--disk", type=int, default=150)
    p.add_argument("--interval", type=int, default=180, help="seconds between attempts")
    p.add_argument("--hours", type=float, default=8.0)
    p.add_argument("--state", type=Path, default=Path("runs/pods.json"))
    p.add_argument("--name", default="wsp-stage7-v3")
    p.add_argument("--purpose", default="Stage 7 v3: on-policy adapter training, broader eval")
    p.add_argument("--gpus", nargs="+", default=None, help="GPU type ids in priority order (default: GPUS)")
    args = p.parse_args(argv)
    key = os.environ.get("RUNPOD_MODEL_TRAINING_API_KEY")
    if not key:
        print("RUNPOD_MODEL_TRAINING_API_KEY is not set; run via doppler run", flush=True)
        return 2
    deadline = time.time() + args.hours * 3600
    attempt = 0
    with httpx.Client(base_url=API, headers={"Authorization": f"Bearer {key}"}, timeout=60) as client:
        while time.time() < deadline:
            attempt += 1
            stamp = time.strftime("%H:%M:%S")
            try:
                r = client.post("/pods", json=body(args.public_key, args.disk, args.name, args.gpus))
            except httpx.HTTPError as exc:
                print(f"{stamp} attempt {attempt}: network error {type(exc).__name__}", flush=True)
                time.sleep(args.interval)
                continue
            if r.status_code in (200, 201):
                pod = r.json()
                gpu = (pod.get("machine") or {}).get("gpuTypeId") or pod.get("gpu", {}).get("id") or "?"
                print(f"{stamp} attempt {attempt}: CREATED {pod.get('id')} gpu={gpu} "
                      f"price/hr={pod.get('costPerHr', PRICES.get(gpu))}", flush=True)
                print("POD_JSON " + json.dumps({k: pod.get(k) for k in ("id", "name", "costPerHr", "machine",
                                                                        "desiredStatus", "gpu")}), flush=True)
                state = json.loads(args.state.read_text(encoding="utf-8-sig")) if args.state.exists() else {"pods": []}
                state["pods"].append({"id": pod.get("id"), "name": args.name,
                                      "cost_per_hr": pod.get("costPerHr", PRICES.get(gpu)), "gpu": gpu,
                                      "purpose": args.purpose,
                                      "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                args.state.write_text(json.dumps(state, indent=4), encoding="utf-8")
                return 0
            text = r.text[:160].replace("\n", " ")
            print(f"{stamp} attempt {attempt}: {r.status_code} {text}", flush=True)
            if r.status_code in (401, 403):
                return 3
            time.sleep(args.interval)
    print("time budget exhausted; no pod created", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
