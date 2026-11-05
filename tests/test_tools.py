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

    def resolve(self, name: str) -> str:
        """Mirror WeightStore.resolve: exact match, else a unique suffix match."""
        if name in self._tensors:
            return name
        suffix = "." + name.lstrip(".")
        matches = [k for k in self._tensors if k.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(f"{name!r} not in checkpoint")

    def get(self, name: str) -> np.ndarray:
        return self._tensors[self.resolve(name)]

    def info(self, name: str) -> FakeInfo:
        return FakeInfo(self._tensors[self.resolve(name)].shape)


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

    def count_tokens(self, text: str) -> int:
        return len(text.split())


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


class ScriptedRunner(FakeRunner):
    """Returns a different answer per prompt, so a before/after is visible."""

    def __init__(self, script: dict[str, str]):
        super().__init__()
        self.script = script

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        self.calls.append((prompt, max_tokens))
        return self.script.get(prompt, "")


def test_baseline_refuses_nothing_and_reports_each_probe() -> None:
    runner = ScriptedRunner({})
    tools = WeightTools(FakeStore({}), runner)
    runner.script = {p.prompt: "132" for p in tools._probes}

    out = tools.baseline()
    assert "baseline captured" in out
    assert "target_12x11" in out
    assert len(tools._baseline) == len(tools._probes)


def test_verify_edit_requires_a_baseline() -> None:
    """A verification with no before has nothing to compare against.

    This is the ordering explore14's conclusion got wrong: it reported post-edit
    behaviour that was never measured after the edit.
    """
    tools = WeightTools(FakeStore({}), ScriptedRunner({}))
    with pytest.raises(ToolError, match="no baseline"):
        tools.verify_edit()


def test_verify_edit_reports_harmful_when_the_target_is_helped_but_a_probe_regresses() -> None:
    runner = ScriptedRunner({})
    tools = WeightTools(FakeStore({}), runner)

    # Baseline: target runs long, held-out stops, guard unchanged.
    runner.script = {
        "What is 12 times 11? Reply with only the digits.": "132 " * 100,
        "What is 17 times 23? Reply with only the digits.": "391 " * 40,
        "What is 9 times 8? Reply with only the digits.": "72 " * 60,
        "Name the capital of France. Reply with only the city.": "Paris " * 20,
    }
    tools.baseline()

    # After the "edit": target much shorter, but held-out grew 3x.
    runner.script = {
        "What is 12 times 11? Reply with only the digits.": "132 " * 8,
        "What is 17 times 23? Reply with only the digits.": "391 " * 120,
        "What is 9 times 8? Reply with only the digits.": "72 " * 60,
        "Name the capital of France. Reply with only the city.": "Paris " * 20,
    }
    out = tools.verify_edit()
    assert "HARMFUL" in out
    assert "heldout_17x23 (regressed)" in out


def test_verify_edit_reports_validated_when_everything_holds() -> None:
    runner = ScriptedRunner({})
    tools = WeightTools(FakeStore({}), runner)

    runner.script = {
        "What is 12 times 11? Reply with only the digits.": "132 " * 100,
        "What is 17 times 23? Reply with only the digits.": "391 " * 40,
        "What is 9 times 8? Reply with only the digits.": "72 " * 60,
        "Name the capital of France. Reply with only the city.": "Paris " * 20,
    }
    tools.baseline()

    runner.script = {
        "What is 12 times 11? Reply with only the digits.": "132 " * 8,
        "What is 17 times 23? Reply with only the digits.": "391 " * 30,
        "What is 9 times 8? Reply with only the digits.": "72 " * 45,
        "Name the capital of France. Reply with only the city.": "Paris " * 20,
    }
    out = tools.verify_edit()
    assert "VALIDATED" in out


def test_verify_edit_remeasures_the_baselined_probes_not_the_default_set() -> None:
    """A custom baseline must be compared against itself, not the default set.

    baseline(probes="target_12x11") stores one reading. If verify_edit re-measured
    the four default probes instead, three verdicts would appear with no baseline
    behind them and the one that mattered would still line up — a silent mismatch
    that costs three extra forward passes and reports results for probes the
    editor never asked about.
    """
    runner = ScriptedRunner({})
    tools = WeightTools(FakeStore({}), runner)
    runner.script = {p.prompt: p.expected for p in tools._probes}

    tools.baseline(probes="target_12x11")
    assert len(tools._baseline) == 1
    assert [p.name for p in tools._probes] == ["target_12x11"]

    before = len(runner.calls)
    tools.verify_edit()
    assert len(runner.calls) - before == 1, "re-measured more than the baseline set"


def test_baseline_rejects_an_unknown_probe_name() -> None:
    tools = WeightTools(FakeStore({}), ScriptedRunner({}))
    with pytest.raises(ToolError, match="unknown probe"):
        tools.baseline(probes="no_such_probe")


def test_baseline_accounts_for_forward_passes() -> None:
    """Verification costs forward passes like any other generation."""
    tools = WeightTools(FakeStore({}), ScriptedRunner({}), max_forward_passes=2)
    with pytest.raises(ToolError, match="forward-pass budget"):
        tools.baseline()


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


# -- activation capture ----------------------------------------------------


class FakeCapturer:
    """Returns a canned activation array per prompt."""

    def __init__(self, by_prompt: dict[str, np.ndarray]):
        self.by_prompt = by_prompt
        self.calls: list[tuple[str, str, int]] = []

    def capture(self, prompt: str, module_path: str, max_tokens: int = 64):
        self.calls.append((prompt, module_path, max_tokens))
        return self.by_prompt[prompt]

    def reload(self, model_path: str) -> None:
        pass


def _capture_tools(tmp_path, by_prompt):
    store = _real_checkpoint(tmp_path)
    capturer = FakeCapturer(by_prompt)
    return WeightTools(store, ReloadableRunner(), capturer=capturer), capturer


def test_capture_activations_reports_shape_and_stats(tmp_path) -> None:
    acts = np.ones((4, 8), dtype=np.float32)
    acts[:, 2] = 5.0
    tools, _ = _capture_tools(tmp_path, {"p": acts})

    out = tools.capture_activations("w", "p")
    assert "activations (4, 8)" in out
    assert "tokens x hidden" in out
    # Highest-norm dimension first.
    assert out.splitlines()[2].startswith("2\t")


def test_capture_activations_hooks_the_tensor_module(tmp_path) -> None:
    tools, capturer = _capture_tools(tmp_path, {"p": np.ones((2, 4), dtype=np.float32)})
    tools.capture_activations("w", "p")
    assert capturer.calls[0][1] == "w", "module path derived from the tensor name"


def test_capture_activations_without_capturer_raises(tools: WeightTools) -> None:
    with pytest.raises(ToolError, match="no activation capturer"):
        tools.capture_activations("model.layers.15.mlp.down_proj.weight", "p")


def test_capture_activations_respects_budget(tmp_path) -> None:
    tools, _ = _capture_tools(tmp_path, {"p": np.ones((2, 4), dtype=np.float32)})
    tools._max_forward_passes = 1
    tools.capture_activations("w", "p")
    with pytest.raises(ToolError, match="budget exhausted"):
        tools.capture_activations("w", "p")


def test_capture_direction_stores_a_named_direction(tmp_path) -> None:
    looping = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    control = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    tools, _ = _capture_tools(tmp_path, {"loop": looping, "calm": control})

    out = tools.capture_direction("d", "w", "loop", "calm")
    assert "direction 'd'" in out
    assert "norm 1.0000" in out
    assert tools.list_directions().count("d") >= 1


def test_capture_direction_reports_contrast_cosine(tmp_path) -> None:
    looping = np.array([[1.0, 0.0]], dtype=np.float32)
    control = np.array([[1.0, 0.0]], dtype=np.float32)
    tools, _ = _capture_tools(tmp_path, {"loop": looping, "calm": control})

    # Identical states: no direction to derive.
    with pytest.raises(ToolError, match="identical"):
        tools.capture_direction("d", "w", "loop", "calm")


def test_capture_direction_flags_a_weak_contrast(tmp_path) -> None:
    looping = np.array([[1.0, 0.001]], dtype=np.float32)
    control = np.array([[1.0, 0.0]], dtype=np.float32)
    tools, _ = _capture_tools(tmp_path, {"loop": looping, "calm": control})

    out = tools.capture_direction("d", "w", "loop", "calm")
    # Nearly parallel states: the cosine rounds to 1.0, which is the warning the
    # editor needs — the contrast found essentially nothing.
    assert "contrast_cosine +1.0000" in out


def test_list_directions_when_empty(tools: WeightTools) -> None:
    assert tools.list_directions() == "no captured directions"


def test_direction_similarity_reports_agreement(tmp_path) -> None:
    """Two captures of the same direction should agree closely."""
    looping = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    control = np.zeros((1, 4), dtype=np.float32)
    tools, _ = _capture_tools(tmp_path, {"loop": looping, "calm": control})

    tools.capture_direction("a", "w", "loop", "calm")
    tools.capture_direction("b", "w", "loop", "calm")

    out = tools.direction_similarity("a", "b")
    assert "cosine('a', 'b') = +1.0000" in out
    assert "reproducible" in out


def test_direction_similarity_rejects_unknown_name(tools: WeightTools) -> None:
    with pytest.raises(ToolError, match="no captured direction"):
        tools.direction_similarity("missing", "also_missing")


def test_capture_direction_accepts_a_token_window(tmp_path) -> None:
    """The window must reach the contrast, not just be accepted and ignored."""
    # A long shared prefix, so averaging over all tokens hides the divergence.
    prefix = np.tile(np.array([1.0, 0.0], dtype=np.float32), (40, 1))
    looping = np.vstack([prefix, np.tile([1.0, 0.0], (4, 1))])
    control = np.vstack([prefix, np.tile([0.0, 1.0], (4, 1))])
    tools, _ = _capture_tools(tmp_path, {"loop": looping, "calm": control})

    whole = tools.capture_direction("whole", "w", "loop", "calm")
    windowed = tools.capture_direction("win", "w", "loop", "calm", token_window="40:44")

    # Over all tokens the states look alike; past the prefix they do not.
    assert "contrast_cosine +0.9" in whole
    assert "contrast_cosine +0.0000" in windowed


def test_aligned_columns_ranks_by_alignment(tmp_path) -> None:
    weight = np.zeros((4, 8), dtype=np.float32)
    weight[0, 5] = 1.0  # column 5 points along +x
    store = FakeStore({"w": weight})
    capturer = FakeCapturer(
        {
            "loop": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            "calm": np.zeros((1, 4), dtype=np.float32),
        }
    )
    tools = WeightTools(store, FakeRunner(), capturer=capturer)
    tools.capture_direction("d", "w", "loop", "calm")

    out = tools.aligned_columns("w", "d", k=3)
    assert out.splitlines()[2].startswith("5\t")


def test_aligned_columns_rejects_unknown_direction(tools: WeightTools) -> None:
    with pytest.raises(ToolError, match="no captured direction"):
        tools.aligned_columns("model.layers.15.mlp.down_proj.weight", "missing")


def test_apply_directional_edit_changes_weights(tmp_path) -> None:
    weight = np.ones((4, 8), dtype=np.float32)
    store = FakeStore({"w": weight})
    store.path = tmp_path / "src" / "model.safetensors"
    store.path.parent.mkdir(parents=True, exist_ok=True)
    from safetensors.numpy import save_file

    save_file({"w": weight}, str(store.path))
    (store.path.parent / "config.json").write_text("{}", encoding="utf-8")

    def reopen(path):
        from safetensors import safe_open

        with safe_open(str(path), framework="np") as handle:
            store._tensors = {"w": handle.get_tensor("w")}

    store.reopen = reopen

    capturer = FakeCapturer(
        {
            "loop": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            "calm": np.zeros((1, 4), dtype=np.float32),
        }
    )
    runner = ReloadableRunner()
    tools = WeightTools(store, runner, capturer=capturer)
    tools.capture_direction("d", "w", "loop", "calm")

    out = tools.apply_directional_edit(
        "w", "d", cols="0:2", scale=1.0, out_path=str(tmp_path / "edited")
    )
    assert "subtracted" in out
    assert runner.reloaded

    edited = store.get("w")
    # Row 0 carried the whole direction, so it drops by 1.0 in the edited columns.
    assert edited[0, 0] == pytest.approx(0.0)
    assert edited[0, 2] == pytest.approx(1.0), "column 2 is outside the range"


def test_apply_directional_edit_accepts_a_column_list(tmp_path) -> None:
    """Editing exactly the ranked columns, not a range that contains them."""
    weight = np.ones((4, 8), dtype=np.float32)
    store = FakeStore({"w": weight})
    store.path = tmp_path / "src" / "model.safetensors"
    store.path.parent.mkdir(parents=True, exist_ok=True)
    from safetensors.numpy import save_file

    save_file({"w": weight}, str(store.path))
    (store.path.parent / "config.json").write_text("{}", encoding="utf-8")

    def reopen(path):
        from safetensors import safe_open

        with safe_open(str(path), framework="np") as handle:
            store._tensors = {"w": handle.get_tensor("w")}

    store.reopen = reopen

    capturer = FakeCapturer(
        {
            "loop": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            "calm": np.zeros((1, 4), dtype=np.float32),
        }
    )
    tools = WeightTools(store, ReloadableRunner(), capturer=capturer)
    tools.capture_direction("d", "w", "loop", "calm")

    out = tools.apply_directional_edit(
        "w", "d", cols="1,4", scale=1.0, out_path=str(tmp_path / "edited")
    )
    assert "subtracted" in out

    edited = store.get("w")
    assert edited[0, 1] == pytest.approx(0.0)
    assert edited[0, 4] == pytest.approx(0.0)
    for col in (0, 2, 3, 5, 6, 7):
        assert edited[0, col] == pytest.approx(1.0), f"column {col} must be untouched"


def test_apply_directional_edit_rejects_unknown_direction(tools: WeightTools) -> None:
    with pytest.raises(ToolError, match="no captured direction"):
        tools.apply_directional_edit(
            "model.layers.15.mlp.down_proj.weight",
            "missing",
            cols="0:2",
            scale=1.0,
            out_path="x",
        )
