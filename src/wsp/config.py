"""Runtime configuration, read from the environment.

Secrets are injected by `doppler run --` and are never read from files or
hardcoded. See README for the Doppler project layout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class MissingSecret(RuntimeError):
    """Raised when a required secret is absent from the environment."""


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise MissingSecret(
            f"{name} is not set. Run commands via `doppler run --project model-training "
            f"--config dev -- <command>` so secrets are injected."
        )
    return value


@dataclass(frozen=True)
class Settings:
    """Resolved settings for a run."""

    openrouter_api_key: str = field(repr=False)
    hf_token: str | None = field(default=None, repr=False)
    wandb_api_key: str | None = field(default=None, repr=False)
    runpod_api_key: str | None = field(default=None, repr=False)

    openrouter_base_url: str = OPENROUTER_BASE_URL

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openrouter_api_key=_require("OPENROUTER_API_KEY"),
            hf_token=os.environ.get("HF_TOKEN"),
            wandb_api_key=os.environ.get("WANDB_API_KEY"),
            runpod_api_key=os.environ.get("RUNPOD_MODEL_TRAINING_API_KEY"),
        )
