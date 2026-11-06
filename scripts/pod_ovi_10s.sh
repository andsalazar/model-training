#!/usr/bin/env bash
# Generate 10-second Ovi rollouts to test whether a longer horizon collapses naturally.
#
# The 5-second results were unanimous: three prompts, no collapse, every ratio within
# 2% of 1.0. TBC's own framing is that information collapse is a long-horizon failure,
# so 5 seconds may simply be too short for the failure mode to appear.
#
# 10 seconds doubles the frame count to 241. If collapse appears anywhere in Ovi's
# supported range, this is where it would be. If it does not, then Ovi at these
# lengths has no stability problem for an adapter to fix, and the project needs a
# different target.
#
# The same three prompts are reused so the 5s and 10s results are directly comparable.
set -uo pipefail

source /workspace/ovi-env/bin/activate
cd /workspace/Ovi

OUT=/workspace/ovi_out/ten
mkdir -p "$OUT"

PROMPTS=(
  "A river flowing through a forest, camera slowly panning. Audio: gentle water sounds and birdsong."
  "A still photograph of a mountain lake at dawn, no motion. Audio: silence with faint wind."
  "A ceiling fan spinning in an empty room. Audio: steady mechanical hum."
)

for i in "${!PROMPTS[@]}"; do
  prompt="${PROMPTS[$i]}"
  echo "=== [$i] $prompt ==="
  cat > "$OUT/cfg_$i.yaml" <<YAML
ckpt_dir: /workspace/Ovi/ckpts
output_dir: $OUT/run_$i
sample_steps: 30
solver_name: unipc
model_name: '960x960_10s'
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
