"""Tests for the phase-log parser.

The parser reads timings out of an existing Ovi log so a generation does not have to
be repeated to measure it. It is the difference between a 9-minute answer and a free
one, which makes a silent misparse expensive: it would report a wrong decode share and
redirect the whole speed plan.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from parse_phase_log import _to_seconds, parse  # noqa: E402

#: The real shape of Ovi's output: denoise steps ending at 06:42, then one decode entry
#: whose per-iteration time is two orders of magnitude larger.
REAL_SHAPE = (
    " 48it [06:26,  8.08s/it]\r"
    "49it [06:34,  8.08s/it]\r"
    "50it [06:42,  8.05s/it]\r"
    "50it [06:42,  8.05s/it]\r"
    "1it [07:15, 435.31s/it]\r"
    "1it [07:15, 435.31s/it]\r"
)

#: Elapsed is cumulative from the start of generate, so the decode duration is the
#: difference between the two markers.
DENOISE_SECONDS = 6 * 60 + 42  # 402s
DECODE_SECONDS = 7 * 60 + 15 - DENOISE_SECONDS  # 33s of decoding after 402s of denoise


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "log.txt"
    path.write_text(text)
    return path


class TestToSeconds:
    def test_mm_ss(self):
        assert _to_seconds("06:42") == 6 * 60 + 42

    def test_hh_mm_ss(self):
        assert _to_seconds("01:06:42") == 3600 + 6 * 60 + 42

    def test_unrecognised_raises(self):
        with pytest.raises(ValueError, match="unrecognised"):
            _to_seconds("nonsense")


class TestParse:
    def test_separates_denoise_from_decode(self, tmp_path):
        report = parse(_write(tmp_path, REAL_SHAPE))
        assert report["denoise_seconds"] == DENOISE_SECONDS
        assert report["decode_seconds"] == DECODE_SECONDS
        assert report["total_seconds"] == DENOISE_SECONDS + DECODE_SECONDS

    def test_denoise_dominates_in_the_real_shape(self, tmp_path):
        # Reading the decode marker as per-phase rather than cumulative would make
        # decode look like 52% of the wall clock and suggest steps are not worth
        # cutting. Correctly read, decode is 33s against 402s of denoise, so denoise
        # dominates and halving steps saves close to half.
        report = parse(_write(tmp_path, REAL_SHAPE))
        assert report["denoise_share"] > 0.9
        assert report["saving_from_halving_steps"] > 0.4

    def test_halving_steps_saving_is_computed(self, tmp_path):
        report = parse(_write(tmp_path, REAL_SHAPE))
        expected = round((report["denoise_seconds"] * 0.5) / report["total_seconds"], 3)
        assert report["saving_from_halving_steps"] == pytest.approx(expected)

    def test_steps_for_2x_total_accounts_for_fixed_decode(self, tmp_path):
        # Decode is fixed, so 2x overall needs denoise to fall to total/2 - decode.
        # Here: 402s denoise, 33s decode, total 435s -> target denoise 184.5s, which is
        # 45.9% of the denoise time, so ~23 of 50 steps.
        report = parse(_write(tmp_path, REAL_SHAPE))
        needed = report["steps_for_2x_total"]
        assert isinstance(needed, float)
        assert needed == pytest.approx(22.95, rel=0.05)

    def test_steps_for_2x_total_is_a_number_when_reachable(self, tmp_path):
        # When denoise dominates, a step cut can reach 2x and the count is reported.
        # The decode rate must be well above the 60s threshold or it is classed as a
        # denoise step and the key is never emitted.
        # Denoise marker 50 min, decode marker 53 min -> 50 min denoise, 3 min decode,
        # total 53 min. To halve the total to 26.5 min, denoise must fall to 23.5 min,
        # which is 47% of 50 steps = 23.5 steps. An earlier formula
        # (`denoise - decode` = 47 min) understated the cut needed by 2x.
        text = "50it [50:00, 60.00s/it]\r1it [53:00, 180.00s/it]\r"
        report = parse(_write(tmp_path, text))
        needed = report["steps_for_2x_total"]
        assert isinstance(needed, float)
        assert needed == pytest.approx(23.5, rel=0.05)

    def test_impossible_2x_is_reported(self, tmp_path):
        # When decode alone exceeds half the total, no step cut can reach 2x. The
        # parser must say so rather than return a negative count.
        text = "50it [00:50, 1.00s/it]\r1it [02:00, 120.00s/it]\r"
        report = parse(_write(tmp_path, text))
        assert "impossible" in str(report["steps_for_2x_total"])

    def test_no_entries_returns_error(self, tmp_path):
        report = parse(_write(tmp_path, "just some log text with no progress"))
        assert "error" in report

    def test_denoise_only_has_no_shares(self, tmp_path):
        report = parse(_write(tmp_path, "50it [06:42,  8.05s/it]\r"))
        assert report["decode_seconds"] is None
        assert "decode_share" not in report

    def test_per_step_seconds_is_reported(self, tmp_path):
        # ``per_step_seconds`` divides by the largest step COUNTER seen, not by the
        # number of log entries: tqdm re-renders its final line, so a 50-step run can
        # produce 51 entries. Using the entry count made a step projection wrong by
        # 10x on the real log. Here the counter reaches 50 over 402s = 8.04s/step.
        report = parse(_write(tmp_path, REAL_SHAPE))
        assert report["denoise_entries"] == 4
        assert report["steps"] == 50
        assert report["per_step_seconds"] == pytest.approx(DENOISE_SECONDS / 50, rel=0.01)
