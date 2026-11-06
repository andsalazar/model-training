# Getting Ovi running on a Blackwell pod

*Four blockers hit in sequence setting up Ovi on an RTX PRO 4500 Blackwell, and what each
one actually was.*

Companion to [stage6-metric.md](stage6-metric.md). This document records the environment
work needed before the metric can be measured on real video.

## The pod

The persistent pod (`o4uhh2qw28w9zk`, EUR-IS-1, RTX 4000 Ada 53GB) is **EXITED** and its
host has no free GPUs — the recurring blocker from earlier stages. Retrying the start
failed twice with "There are not enough free GPUs on the host machine to start this pod."

So a fresh pod was created: `ia30p9xluq53bk`, **RTX PRO 4500 Blackwell**, EU-RO-1,
$0.72/hr. The API reported `memory: 62` but that is *host* RAM; `nvidia-smi` shows
**32,623 MiB VRAM**. That is Ovi's stated minimum, so CPU offload is required.

| | Value |
|---|---|
| Pod | `ia30p9xluq53bk` |
| GPU | RTX PRO 4500 Blackwell, 32GB VRAM |
| Compute capability | **sm_120** |
| Data centre | EU-RO-1 |
| Cost | $0.72/hr |
| Disk | 80GB overlay, 251GB host RAM, 112 vCPU |

## Blocker 1 — the SSH proxy hangs

The Runpod SSH *proxy* (`ssh.runpod.io`) hangs on this pod. A 75-second hard timeout
returned nothing. The pod's **direct** endpoint works:

```bash
ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 root@213.173.108.44 -p 36586 "nvidia-smi"
```

The proxy also requires a PTY (`Error: Your SSH client doesn't support PTY`), which breaks
non-interactive use. `scripts/ovi_ssh.ps1` targets the direct endpoint with `BatchMode=yes`
so a missing key fails fast instead of prompting.

Note the direct host and port change on pod restart — re-read them from `get-pod`'s
`ssh.direct` block rather than hardcoding.

## Blocker 2 — PEP 668 blocks pip

The pod's system Python is externally managed:

```
error: externally-managed-environment
hint: See PEP 668 for the specification.
```

`--break-system-packages` would work but risks breaking the image's own tooling. A venv at
`/workspace/ovi-env` sidesteps it cleanly. Everything after this runs inside that venv.

## Blocker 3 — FlashAttention built against the wrong torch ABI

The first generation attempt failed at import:

```
RuntimeError: Failed to import diffusers.pipelines.flux.pipeline_flux ...
flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol:
_ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```

That mangled symbol is `c10::Error::Error(c10::SourceLocation, std::string)`. The pod image
ships torch 2.8.0 in system Python; the venv installed 2.6.0 per Ovi's requirements. The
prebuilt `flash_attn` wheel was compiled against 2.8.0's C++ ABI, so importing it under
2.6.0 fails.

**The README calls FlashAttention optional. It is not.** Removing it let the imports
succeed, but generation then failed on a hard assertion:

```
assert FLASH_ATTN_2_AVAILABLE
AssertionError
```

Ovi's attention module asserts on it. So it has to be installed and ABI-matched.

## Blocker 4 — torch 2.6.0 has no Blackwell kernels

With FlashAttention removed, generation got further and failed on the first CUDA kernel:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

The diagnosis:

```
GPU compute capability: (12, 0)   # sm_120
torch 2.6.0+cu124 arch list: sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90
```

Torch 2.6.0's CUDA 12.4 build tops out at **sm_90**. Blackwell is **sm_120**. Every kernel
launch fails. Ovi's `requirements.txt` pins 2.6.0, but that pin predates Blackwell and
cannot run on this GPU at all — the pin is a reproducibility choice, not a hard requirement.

The fix is torch 2.7.1+cu128, whose arch list includes sm_120:

```
torch 2.7.1+cu128
arch list: sm_75 sm_80 sm_86 sm_90 sm_100 sm_120 compute_120
matmul OK
```

Verified with a real 512×512 matmul on the GPU, not just an arch-list check — an arch list
that mentions sm_120 is not proof the build works.

## The lesson worth carrying forward

Three of these four blockers produced error messages that pointed at the *wrong* thing:

- The FlashAttention ABI error surfaced as "Failed to import diffusers.pipelines.flux",
  which reads like a diffusers problem. It was a torch-version mismatch two layers down.
- The Blackwell kernel error surfaced as "CUDA error: no kernel image is available",
  which reads like a driver problem. It was a torch build that predates the GPU.
- The PEP 668 error surfaced as a pip failure, which reads like a permissions problem. It
  was an intentional Python packaging policy.

This is the same pattern as the metric work: the first error message is rarely the real
cause. Reading the traceback to the bottom, and checking the actual version numbers rather
than trusting the pin, is what resolved each one.

## Setup script

`scripts/pod_ovi_setup.sh` does the whole install: clone, venv, torch, requirements,
weights. It is idempotent — re-running skips the clone and the venv.

`scripts/pod_ovi_torch.sh` installs the Blackwell-capable torch. Run it after setup, or
instead of setup's torch step.

`scripts/pod_ovi_flash.sh` / `pod_ovi_flash2.sh` build FlashAttention. The second version
fixes two problems in the first: `nvcc` was not on PATH (the toolkit is at
`/usr/local/cuda-12.8`), and pip's build isolation created an env with no torch, so
`setup.py` failed with "No module named 'torch'". `--no-build-isolation` plus
`CUDA_HOME` and `TORCH_CUDA_ARCH_LIST=12.0` is what works.

## What is installed

| Component | Version / size |
|---|---|
| torch | 2.7.1+cu128 (sm_120 capable) |
| transformers | 4.51.3 |
| Ovi checkpoint | `model_960x960.safetensors`, 23.3GB |
| Wan 2.2 VAE | `Wan2.2_VAE.pth` |
| T5 encoder | `models_t5_umt5-xxl-enc-bf16.pth` |
| MMAudio | `ext_weights/` |
| Disk used | 44GB of 80GB |

## Cost so far

The pod has been running since 08:43. At $0.72/hr, the setup and debugging above cost
roughly **$1.50–2.00** in pod time. The generation runs themselves are the remaining cost:
Ovi's README quotes ~118s per 121-frame 720×720 video at 50 steps with CPU offload, so five
prompts at 30 steps is on the order of 5–8 minutes of GPU time, or about **$0.10**.

That is well inside the "$2–4" estimate in the scoping document, and the metric question is
answered either way.
