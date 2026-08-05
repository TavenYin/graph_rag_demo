"""OpenAI-compatible asynchronous chat completion client."""

from __future__ import annotations

from typing import Any

import httpx
from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from graph_rag_demo.clients.embedding import ModelResponseError


class LLMClient:
    """Executes arbitrary chat messages without owning business prompts."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        http_client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            transport=transport,
            timeout=30.0,
        )
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/") + "/",
            http_client=http_client,
        )

    async def complete(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        json_mode: bool = False,
    ) -> str:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0,
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}

        try:
            response = await self._client.chat.completions.create(**request)
        except OpenAIError as error:
            raise ModelResponseError("model completion request failed") from error

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as error:
            raise ModelResponseError(
                "model response does not contain assistant content"
            ) from error
        if not isinstance(content, str):
            raise ModelResponseError("model response assistant content must be a string")
        return content

    async def aclose(self) -> None:
        await self._client.close()
