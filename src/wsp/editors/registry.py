"""Editor registry.

Candidate editors are declared here once so a run can name them by short alias
and the comparison stays reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

from wsp.editors.base import Editor
from wsp.editors.openrouter import OpenRouterEditor


@dataclass(frozen=True)
class EditorSpec:
    """A named editor candidate."""

    alias: str
    model_id: str
    #: USD per million input / output tokens, for cost estimates.
    usd_per_mtok_in: float
    usd_per_mtok_out: float
    notes: str = ""

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens / 1_000_000 * self.usd_per_mtok_in
            + completion_tokens / 1_000_000 * self.usd_per_mtok_out
        )


#: Verified against the OpenRouter model list on 2026-11-03.
EDITORS: dict[str, EditorSpec] = {
    "astra": EditorSpec(
        alias="astra",
        model_id="openai/gpt-6-astra",
        usd_per_mtok_in=10.00,
        usd_per_mtok_out=50.00,
        notes="The editor used in the paper. Reference arm.",
    ),
    "deepseek": EditorSpec(
        alias="deepseek",
        model_id="deepseek/deepseek-v4.1-flash-1017",
        usd_per_mtok_in=0.03,
        usd_per_mtok_out=0.48,
        notes="MIT weights, 1M context. ~333x cheaper input than Astra.",
    ),
    "kimi": EditorSpec(
        alias="kimi",
        model_id="moonshotai/kimi-k3.1-pro",
        usd_per_mtok_in=2.00,
        usd_per_mtok_out=10.00,
        notes="Modified MIT. Mid-cost frontier arm.",
    ),
    "muse": EditorSpec(
        alias="muse",
        model_id="meta/muse-glimmer-30b",
        usd_per_mtok_in=0.30,
        usd_per_mtok_out=1.20,
        notes="Apache 2.0, 30B. Small enough to self-host later.",
    ),
}


def get_spec(alias: str) -> EditorSpec:
    try:
        return EDITORS[alias]
    except KeyError:
        known = ", ".join(sorted(EDITORS))
        raise KeyError(f"unknown editor {alias!r}; known editors: {known}") from None


def build_editor(alias: str, api_key: str) -> Editor:
    """Construct an editor by alias."""
    return OpenRouterEditor(model_id=get_spec(alias).model_id, api_key=api_key)
