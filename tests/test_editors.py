"""Tests for the editor layer, using a mock transport instead of the network."""

from __future__ import annotations

import httpx
import pytest

from wsp.editors.openrouter import OpenRouterEditor, OpenRouterError
from wsp.editors.registry import EDITORS, get_spec


def _editor(handler, **kwargs) -> OpenRouterEditor:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return OpenRouterEditor("test/model", "key", client=http, **kwargs)


def _ok_body(text: str = "ok") -> dict:
    return {
        "model": "test/model",
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }


@pytest.mark.asyncio
async def test_complete_parses_text_and_usage() -> None:
    editor = _editor(lambda request: httpx.Response(200, json=_ok_body("hello")))
    response = await editor.complete("sys", "user")

    assert response.text == "hello"
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 2
    assert response.usage.total_tokens == 12


@pytest.mark.asyncio
async def test_complete_falls_back_to_reasoning_field() -> None:
    body = {
        "model": "test/model",
        "choices": [{"message": {"content": "", "reasoning": "thought"}}],
        "usage": {},
    }
    editor = _editor(lambda request: httpx.Response(200, json=body))
    response = await editor.complete("sys", "user")
    assert response.text == "thought"


@pytest.mark.asyncio
async def test_complete_raises_on_client_error() -> None:
    editor = _editor(lambda request: httpx.Response(400, text="bad request"))
    with pytest.raises(OpenRouterError, match="HTTP 400"):
        await editor.complete("sys", "user")


@pytest.mark.asyncio
async def test_complete_retries_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    responses = [httpx.Response(429, text="slow down"), httpx.Response(200, json=_ok_body())]
    editor = _editor(lambda request: responses.pop(0))

    response = await editor.complete("sys", "user")
    assert response.text == "ok"


@pytest.mark.asyncio
async def test_complete_raises_after_retries_exhausted(monkeypatch) -> None:
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    editor = _editor(lambda request: httpx.Response(503, text="unavailable"), max_retries=2)
    with pytest.raises(OpenRouterError, match="retries exhausted"):
        await editor.complete("sys", "user")


@pytest.mark.asyncio
async def test_complete_raises_on_empty_choices() -> None:
    editor = _editor(lambda request: httpx.Response(200, json={"choices": []}))
    with pytest.raises(OpenRouterError, match="no choices"):
        await editor.complete("sys", "user")


async def _no_sleep(_seconds: float) -> None:
    return None


def test_registry_has_expected_editors() -> None:
    assert set(EDITORS) == {"astra", "deepseek", "kimi", "muse", "glm", "qwen"}


def test_comparison_and_control_arms_are_disjoint() -> None:
    from wsp.editors.registry import COMPARISON_ARMS, CONTROL_ARMS

    assert set(COMPARISON_ARMS).isdisjoint(CONTROL_ARMS)
    assert set(COMPARISON_ARMS) | set(CONTROL_ARMS) <= set(EDITORS)


def test_qwen_is_a_control_arm() -> None:
    from wsp.editors.registry import is_control

    assert is_control("qwen")
    assert not is_control("astra")


def test_glm_is_cheapest_output_arm() -> None:
    glm = get_spec("glm")
    for alias, spec in EDITORS.items():
        if alias == "glm":
            continue
        assert glm.usd_per_mtok_out <= spec.usd_per_mtok_out, f"{alias} is cheaper"


def test_get_spec_rejects_unknown_alias() -> None:
    with pytest.raises(KeyError, match="unknown editor"):
        get_spec("nonexistent")


def test_cost_estimate_scales_with_tokens() -> None:
    spec = get_spec("astra")
    # 1M in + 1M out at $10/$50.
    assert spec.estimate_cost(1_000_000, 1_000_000) == pytest.approx(60.0)


def test_deepseek_is_much_cheaper_than_astra() -> None:
    astra = get_spec("astra").estimate_cost(200_000, 20_000)
    deepseek = get_spec("deepseek").estimate_cost(200_000, 20_000)
    assert astra > deepseek * 100
