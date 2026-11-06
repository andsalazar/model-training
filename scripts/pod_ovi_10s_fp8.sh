#!/usr/bin/env bash
# Generate 5s and 10s rollouts with the fp8 checkpoint.
#
# The bf16 10s run OOM'd: 241 frames at 960x960 needs more than the 31.4 GiB this GPU
# has, and it died in rope_apply_3d trying to allocate another 1.26 GiB with 730 MiB
# free. Ovi documents fp8 as the way to run in 24GB, so this tests whether fp8 plus
# CPU offload fits 10 seconds in 32GB.
#
# fp8 is a quality compromise - Ovi's own README says it "will have quality
# degradation". That matters for the claim: if a 10s fp8 rollout collapses, the
# collapse could be an fp8 artefact rather than a property of the model. So the
# result is only usable as evidence about horizon if the fp8 5s baseline is also
# healthy. Both are generated here.
set -uo pipefail

source /workspace/ovi-env/bin/activate
cd /workspace/Ovi

OUT=/workspace/ovi_out/fp8
mkdir -p "$OUT"

PROMPT="A river flowing through a forest, camera slowly panning. Audio: gentle water sounds and birdsong."

for spec in "960x960_5s:5s" "960x960_10s:10s"; do
  model="${spec%%:*}"
  tag="${spec##*:}"
  echo "=== $tag ($model, fp8) ==="

  cfg="$OUT/cfg_$tag.yaml"
  {
    echo "ckpt_dir: /workspace/Ovi/ckpts"
    echo "output_dir: $OUT/run_$tag"
    echo "sample_steps: 30"
    echo "solver_name: unipc"
    echo "model_name: '$model'"
    echo "shift: 5.0"
    echo "sp_size: 1"
    echo "audio_guidance_scale: 3.0"
    echo "video_guidance_scale: 4.0"
    echo "mode: 't2v'"
    echo "fp8: True"
    echo "cpu_offload: True"
    echo "seed: 100"
    echo "video_negative_prompt: 'jitter, bad hands, blur, distortion'"
    echo "audio_negative_prompt: 'robotic, muffled, echo, distorted'"
    echo "video_frame_height_width: [960, 960]"
    echo "text_prompt: \"$PROMPT\""
    echo "slg_layer: 11"
    echo "each_example_n_times: 1"
  } > "$cfg"
  python3 inference.py --config-file "$cfg" > "$OUT/gen_$tag.log" 2>&1
  echo "  exit $?"
done

echo "=== generated ==="
find "$OUT" -name '*.mp4' | sort
