#!/bin/bash
# Write the 5s capture config the pair script expects.
#
# The repo ships only inference_fusion.yaml, which targets 960x960_10s and i2v mode.
# The capture needs the 5s checkpoint in t2v mode, so a dedicated config is written
# rather than editing the shipped one -- editing it would make the repo's own example
# stop working.
set -uo pipefail

# Inside the repo, which is /local/Ovi with a capital O. Writing to /local/ovi (lowercase)
# creates a separate directory that the capture script does not read.
CFG=/local/Ovi/ovi/configs/inference/inference_fusion_wsp.yaml
mkdir -p "$(dirname "$CFG")"

cat > "$CFG" <<'YAML'
# 5s text-to-video capture config for WSP activation work.
#
# Differs from the shipped inference_fusion.yaml in three ways:
#   * model_name is 960x960_5s, the only checkpoint downloaded (activation capture
#     needs a forward pass, not a long video, so the smaller model is enough).
#   * mode is t2v, so no image conditioning is needed.
#   * sample_steps is overridden per rollout by the capture driver, so the value here
#     is only a default.
ckpt_dir: ./ckpts
output_dir: ./outputs
sample_steps: 30
solver_name: unipc
model_name: "960x960_5s"
shift: 5.0
sp_size: 1
audio_guidance_scale: 3.0
video_guidance_scale: 4.0
mode: "t2v"
fp8: False
cpu_offload: False
seed: 42
video_negative_prompt: "jitter, bad hands, blur, distortion"
audio_negative_prompt: "robotic, muffled, echo, distorted"
video_frame_height_width: [960, 960]
text_prompt: ""
slg_layer: 11
YAML

echo "wrote $CFG"
cat "$CFG"
