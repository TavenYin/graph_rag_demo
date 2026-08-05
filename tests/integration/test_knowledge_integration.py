"""Integration tests for document ingestion against the local Compose database."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import text


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="set RUN_INTEGRATION_TESTS=1 to use the local PostgreSQL database",
)


class DeterministicEmbeddingClient:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.01] * 1024 for _ in texts]


@pytest_asyncio.fixture
async def database() -> AsyncIterator[object]:
    from graph_rag_demo.config import Settings
    from graph_rag_demo.db import Database

    instance = Database.create(Settings.from_env())
    try:
        yield instance
    finally:
        await instance.close()


@pytest.mark.asyncio
async def test_upload_writes_document_and_actual_token_counts(database) -> None:
    from graph_rag_demo.services.knowledge import KnowledgeService

    service = KnowledgeService(
        database=database,
        embedding_client=DeterministicEmbeddingClient(),
        chunk_size=4,
        chunk_overlap=1,
    )
    document_id = await service.upload("one two three four five six", title="integration")

    try:
        async with database.session() as session:
            rows = (
                await session.execute(
                    text(
                        "SELECT token_count FROM kb_chunk "
                        "WHERE document_id = :document_id ORDER BY chunk_index"
                    ),
                    {"document_id": document_id},
                )
            ).scalars().all()
    finally:
        async with database.session() as session:
            await session.execute(
                text("DELETE FROM kb_document WHERE id = :document_id"),
                {"document_id": document_id},
            )

    assert rows == [4, 3]
