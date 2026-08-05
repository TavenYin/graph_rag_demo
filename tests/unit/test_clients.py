import json
import math

import httpx
import pytest

from graph_rag_demo.clients.embedding import EmbeddingClient, ModelResponseError
from graph_rag_demo.clients.llm import LLMClient


def transport_with(payload: dict[str, object]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_embedding_client_parses_openai_compatible_batch_response() -> None:
    client = EmbeddingClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="text-embedding-v4",
        dimensions=1024,
        transport=transport_with(
            {
                "data": [
                    {"embedding": [0.1] * 1024},
                    {"embedding": [0.3] * 1024},
                ]
            }
        ),
    )

    try:
        assert await client.embed(["first", "second"]) == [[0.1] * 1024, [0.3] * 1024]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_embedding_client_rejects_malformed_json_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json", request=request)

    client = EmbeddingClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="text-embedding-v4",
        dimensions=1024,
        transport=httpx.MockTransport(handler),
    )

    try:
        with pytest.raises(ModelResponseError, match="valid JSON"):
            await client.embed(["first"])
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("vector", [[0.1] * 1023, [0.1] * 1025])
async def test_embedding_client_rejects_vector_with_wrong_configured_dimension(
    vector: list[float],
) -> None:
    client = EmbeddingClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="text-embedding-v4",
        dimensions=1024,
        transport=transport_with({"data": [{"embedding": vector}]}),
    )

    try:
        with pytest.raises(ModelResponseError, match="1024 values"):
            await client.embed(["first"])
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
async def test_embedding_client_rejects_non_finite_vector_value(value: float) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps({"data": [{"embedding": [value] * 1024}]}),
            request=request,
        )

    client = EmbeddingClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="text-embedding-v4",
        dimensions=1024,
        transport=httpx.MockTransport(handler),
    )

    try:
        with pytest.raises(ModelResponseError, match="non-finite"):
            await client.embed(["first"])
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_llm_client_accepts_business_defined_messages() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen-plus",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "arbitrary response"},
                        "finish_reason": "stop",
                    }
                ],
            },
            request=request,
        )

    client = LLMClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="qwen-plus",
        transport=httpx.MockTransport(handler),
    )

    try:
        messages = [
            {"role": "system", "content": "Use this business-specific instruction."},
            {"role": "user", "content": "A custom user message."},
        ]
        assert await client.complete(messages, json_mode=True) == "arbitrary response"
        assert requests == [
            {
                "model": "qwen-plus",
                "messages": messages,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
        ]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_llm_client_rejects_response_without_assistant_content() -> None:
    client = LLMClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="qwen-plus",
        transport=transport_with(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen-plus",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": None},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
    )

    try:
        with pytest.raises(ModelResponseError, match="assistant content"):
            await client.complete([{"role": "user", "content": "question"}])
    finally:
        await client.aclose()
