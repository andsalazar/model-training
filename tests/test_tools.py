"""Tests for the editor tool surface, using a fake store and runner."""

from __future__ import annotations

import numpy as np
import pytest

from wsp.tools import ToolError, ToolLog, WeightTools


class FakeInfo:
    def __init__(self, shape, dtype="BF16"):
        self.shape = shape
        self.dtype = dtype


class FakeStore:
    """Minimal stand-in for WeightStore."""

    def __init__(self, tensors: dict[str, np.ndarray]):
        self._tensors = tensors
        self.keys = list(tensors)

    def get(self, name: str) -> np.ndarray:
        return self._tensors[name]

    def info(self, name: str) -> FakeInfo:
        return FakeInfo(self._tensors[name].shape)


class FakeRunner:
    def __init__(self, text: str = "looping text", looped: bool = True):
        self.text = text
        self.looped = looped
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        self.calls.append((prompt, max_tokens))
        return self.text

    def measure(self, prompt: str, max_tokens: int = 2048) -> dict:
        self.calls.append((prompt, max_tokens))
        return {"looped": self.looped, "tokens": 2048, "hit_cap": self.looped}


@pytest.fixture
def tools() -> WeightTools:
    store = FakeStore(
        {
            "model.layers.15.mlp.down_proj.weight": np.ones((8, 128), dtype=np.float32),
            "model.layers.19.mlp.down_proj.weight": np.ones((8, 128), dtype=np.float32),
        }
    )
    return WeightTools(store, FakeRunner())


def test_list_tensors_filters(tools: WeightTools) -> None:
    out = tools.list_tensors("layers.15")
    assert "layers.15" in out
    assert "layers.19" not in out


def test_list_tensors_reports_no_match(tools: WeightTools) -> None:
    with pytest.raises(ToolError, match="no tensors match"):
        tools.list_tensors("nonexistent")


def test_list_tensors_states_truncation() -> None:
    store = FakeStore({f"t{i}": np.ones((2, 2), dtype=np.float32) for i in range(10)})
    tools = WeightTools(store, FakeRunner())

    out = tools.list_tensors("", limit=3)
    assert "showing 3 of 10" in out
    assert "Narrow the pattern" in out


def test_list_tensors_omits_truncation_note_when_complete() -> None:
    store = FakeStore({"a": np.ones((2, 2), dtype=np.float32)})
    tools = WeightTools(store, FakeRunner())

    out = tools.list_tensors("", limit=10)
    assert "showing" not in out


def test_read_slice_returns_text(tools: WeightTools) -> None:
    out = tools.read_slice("model.layers.15.mlp.down_proj.weight", rows="0:2", cols="0:4")
    assert len(out.splitlines()) == 2


def test_read_slice_truncates(tools: WeightTools) -> None:
    out = tools.read_slice("model.layers.15.mlp.down_proj.weight", max_chars=20)
    assert "truncated" in out


def test_tensor_stats_sorts_by_norm() -> None:
    array = np.ones((4, 8), dtype=np.float32)
    array[:, 3] = 10.0  # one dominant column
    store = FakeStore({"w": array})
    tools = WeightTools(store, FakeRunner())

    out = tools.tensor_stats("w")
    lines = out.splitlines()
    # Header, column header, then the highest-norm column first.
    assert lines[2].startswith("3\t")


def test_run_model_returns_text(tools: WeightTools) -> None:
    assert tools.run_model("prompt") == "looping text"


def test_run_model_respects_budget() -> None:
    store = FakeStore({"w": np.ones((2, 2), dtype=np.float32)})
    tools = WeightTools(store, FakeRunner(), max_forward_passes=1)

    tools.run_model("first")
    with pytest.raises(ToolError, match="budget exhausted"):
        tools.run_model("second")


def test_measure_loop_returns_json(tools: WeightTools) -> None:
    out = tools.measure_loop("prompt")
    assert '"looped": true' in out


def _real_checkpoint(tmp_path):
    """A tiny on-disk checkpoint, since apply_edit writes a full directory."""
    from safetensors.numpy import save_file

    from wsp.weights.store import WeightStore

    src = tmp_path / "src"
    src.mkdir()
    save_file(
        {"w": np.ones((4, 8), dtype=np.float32)},
        str(src / "model.safetensors"),
    )
    (src / "config.json").write_text("{}", encoding="utf-8")
    return WeightStore(src / "model.safetensors")


class ReloadableRunner(FakeRunner):
    def __init__(self):
        super().__init__()
        self.reloaded: list[str] = []

    def reload(self, model_path: str) -> None:
        self.reloaded.append(model_path)


def test_apply_edit_writes_checkpoint_and_reloads(tmp_path) -> None:
    store = _real_checkpoint(tmp_path)
    runner = ReloadableRunner()
    tools = WeightTools(store, runner)

    out = tools.apply_edit(
        "w", cols="0:4", direction="0.1,0.2,0.3,0.4", out_path=str(tmp_path / "edited")
    )

    assert "reloaded" in out
    assert (tmp_path / "edited" / "model.safetensors").exists()
    assert runner.reloaded, "runner must be pointed at the edited checkpoint"


def test_apply_edit_actually_changes_weights(tmp_path) -> None:
    store = _real_checkpoint(tmp_path)
    tools = WeightTools(store, ReloadableRunner())

    tools.apply_edit(
        "w", cols="0:2", direction="1.0,1.0", out_path=str(tmp_path / "edited")
    )

    # The store now points at the edited file; column 0 should be 0.0, not 1.0.
    edited = store.get("w")
    assert edited[0, 0] == pytest.approx(0.0)
    assert edited[0, 2] == pytest.approx(1.0)


def test_apply_edit_rejects_wrong_direction_length(tools: WeightTools) -> None:
    with pytest.raises(ToolError, match="direction has"):
        tools.apply_edit(
            "model.layers.15.mlp.down_proj.weight",
            cols="0:4",
            direction="0.1,0.2",
            out_path="x",
        )


def test_apply_edit_rejects_non_numeric(tools: WeightTools) -> None:
    with pytest.raises(ToolError, match="comma-separated"):
        tools.apply_edit(
            "model.layers.15.mlp.down_proj.weight",
            cols="0:2",
            direction="a,b",
            out_path="x",
        )


def test_log_records_every_call(tools: WeightTools) -> None:
    tools.list_tensors("layers")
    tools.run_model("p")
    names = [c["name"] for c in tools.log.summary()]
    assert names == ["list_tensors", "run_model"]


def test_log_writes_json(tmp_path) -> None:
    log = ToolLog()
    path = log.write(tmp_path / "log.json")
    assert path.exists()


def test_forward_passes_left_decrements(tools: WeightTools) -> None:
    before = tools.forward_passes_left
    tools.run_model("p")
    assert tools.forward_passes_left == before - 1
