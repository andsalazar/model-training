"""Editor interface.

An *editor* is the model that reads and writes another network's weights. The
paper's central claim is that this is a capability of the editor, not of the
target, so every task must be runnable against any editor unchanged.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class Usage:
    """Token accounting for a single editor call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


@dataclass
class EditorResponse:
    """What an editor returns: text, plus what it cost."""

    text: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    raw: dict | None = None


class Editor(abc.ABC):
    """A model that can read and write neural network weights."""

    #: OpenRouter model id, e.g. ``openai/gpt-6-astra``.
    model_id: str

    @abc.abstractmethod
    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> EditorResponse:
        """Send one prompt and return the response."""

    async def aclose(self) -> None:
        """Release any underlying connections."""
        return None

    async def __aenter__(self) -> "Editor":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model_id={self.model_id!r})"
