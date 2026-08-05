"""Async client for DashScope's OpenAI-compatible embedding API."""

from __future__ import annotations

import math
from typing import Any

import httpx


class ModelResponseError(ValueError):
    """Raised when a model endpoint returns a response outside this contract."""


class EmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dimensions: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
            timeout=30.0,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = await self._http.post(
            "embeddings",
            json={
                "model": self._model,
                "input": texts,
                "dimensions": self._dimensions,
            },
        )
        response.raise_for_status()
        payload = self._json_object(response)
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise ModelResponseError("embedding response data must match the input count")

        embeddings: list[list[float]] = []
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise ModelResponseError("embedding response contains an invalid vector")
            vector = item["embedding"]
            if len(vector) != self._dimensions:
                raise ModelResponseError(
                    f"embedding response vector must contain exactly {self._dimensions} values"
                )
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in vector):
                raise ModelResponseError("embedding response contains a non-numeric vector value")
            if any(not math.isfinite(value) for value in vector):
                raise ModelResponseError("embedding response contains a non-finite vector value")
            embeddings.append([float(value) for value in vector])
        return embeddings

    async def aclose(self) -> None:
        await self._http.aclose()

    @staticmethod
    def _json_object(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as error:
            raise ModelResponseError("model response is not valid JSON") from error
        if not isinstance(payload, dict):
            raise ModelResponseError("model response JSON must be an object")
        return payload
