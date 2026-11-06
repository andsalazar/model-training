#!/usr/bin/env bash
# Generate several Ovi rollouts and measure each, to test whether "no natural
# collapse at 5 seconds" is prompt-specific or general.
#
# The first real rollout (a river through a forest) did not collapse: per-frame
# ratio 0.86 with R^2 0.30, delta ratio 0.88 with R^2 0.47. Both below the 0.5 fit
# threshold, so neither registered as a collapse. That is one prompt. If every
# prompt behaves the same way, then a 5-second Ovi rollout simply does not collapse
# on its own, and the project has no natural target to measure â€” which is a finding
# worth having before any adapter work.
#
# Prompts are chosen to span the failure surface: a static scene (most likely to
# collapse into a still), sustained motion (least likely), and a repeating element
# (the classic degenerate attractor).
set -uo pipefail

source /workspace/ovi-env/bin/activate
cd /workspace/Ovi

OUT=/workspace/ovi_out/batch
mkdir -p "$OUT"

PROMPTS=(
  "A still photograph of a mountain lake at dawn, no motion. Audio: silence with faint wind."
  "A ceiling fan spinning in an empty room. Audio: steady mechanical hum."
  "A person walking along a beach, waves in the background. Audio: waves and footsteps."
)

for i in "${!PROMPTS[@]}"; do
  prompt="${PROMPTS[$i]}"
  echo "=== [$i] $prompt ==="
  cat > "$OUT/cfg_$i.yaml" <<YAML
ckpt_dir: /workspace/Ovi/ckpts
output_dir: $OUT/run_$i
sample_steps: 30
solver_name: unipc
model_name: '960x960_5s'
shift: 5.0
sp_size: 1
audio_guidance_scale: 3.0
video_guidance_scale: 4.0
mode: 't2v'
fp8: False
cpu_offload: True
seed: 100
video_negative_prompt: 'jitter, bad hands, blur, distortion'
audio_negative_prompt: 'robotic, muffled, echo, distorted'
video_frame_height_width: [960, 960]
text_prompt: "$prompt"
slg_layer: 11
each_example_n_times: 1
YAML
  python3 inference.py --config-file "$OUT/cfg_$i.yaml" > "$OUT/gen_$i.log" 2>&1
  echo "  exit $?"
done

echo "=== generated ==="
find "$OUT" -name '*.mp4' | sort
