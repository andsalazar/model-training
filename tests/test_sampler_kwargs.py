"""The capture drivers must sample the same trajectory Ovi's inference.py does."""

from wsp.dit import ovi_sampler_kwargs


def test_reads_every_setting_inference_py_reads():
    config = {
        "solver_name": "unipc",
        "shift": 5.0,
        "video_guidance_scale": 4.0,
        "audio_guidance_scale": 3.0,
        "slg_layer": 11,
        "video_negative_prompt": "jitter, bad hands, blur, distortion",
        "audio_negative_prompt": "robotic, muffled, echo, distorted",
    }
    assert ovi_sampler_kwargs(config) == config


def test_defaults_are_inference_py_not_generate():
    # generate() defaults to guidance 5.0/4.0 and slg_layer 9; inference.py to 4.0/3.0
    # and 11. The capture drivers used generate()'s, so every Stage 6 capture sampled a
    # different trajectory from the speed and quality sweeps.
    kwargs = ovi_sampler_kwargs({})
    assert kwargs["video_guidance_scale"] == 4.0
    assert kwargs["audio_guidance_scale"] == 3.0
    assert kwargs["slg_layer"] == 11


def test_negative_prompt_override_applies_to_both_streams():
    kwargs = ovi_sampler_kwargs({"video_negative_prompt": "x"}, negative_prompt="y")
    assert kwargs["video_negative_prompt"] == kwargs["audio_negative_prompt"] == "y"
