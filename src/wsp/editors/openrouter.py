"""OpenRouter-backed editor.

One API key reaches every candidate editor, which is what makes an
apples-to-apples comparison practical: the transport, retry policy, and
accounting are identical, so differences come from the model.
"""

from __future__ import annotations

import asyncio

import httpx

from wsp.editors.base import Editor, EditorResponse, Usage

#: Transient statuses worth retrying. 429 is rate limiting, 5xx is upstream.
RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter returns a non-retryable error."""


class OpenRouterEditor(Editor):
    """Editor backed by the OpenRouter chat completions API."""

    def __init__(
        self,
        model_id: str,
        api_key: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 600.0,
        max_retries: int = 5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model_id = model_id
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = 0.0,
        max_tokens: int = 8192,
        extra: dict | None = None,
    ) -> EditorResponse:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }
        # Some reasoning models reject a temperature; None leaves the provider default.
        if temperature is not None:
            payload["temperature"] = temperature
        if extra:
            payload.update(extra)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: str = ""
        for attempt in range(self._max_retries):
            response = await self._client.post(
                f"{self._base_url}/chat/completions", json=payload, headers=headers
            )
            if response.status_code == 200:
                return self._parse(response.json())

            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            if response.status_code not in RETRY_STATUSES:
                raise OpenRouterError(f"{self.model_id}: {last_error}")

            # Exponential backoff, capped so a long run cannot stall indefinitely.
            await asyncio.sleep(min(2**attempt, 30))

        raise OpenRouterError(f"{self.model_id}: retries exhausted ({last_error})")

    def _parse(self, body: dict) -> EditorResponse:
        choices = body.get("choices") or []
        if not choices:
            raise OpenRouterError(f"{self.model_id}: response contained no choices")

        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        # Reasoning models may return content in a separate field.
        if not text and message.get("reasoning"):
            text = message["reasoning"]

        usage_block = body.get("usage") or {}
        usage = Usage(
            prompt_tokens=usage_block.get("prompt_tokens", 0),
            completion_tokens=usage_block.get("completion_tokens", 0),
        )
        return EditorResponse(
            text=text,
            usage=usage,
            model=body.get("model", self.model_id),
            raw=body,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
