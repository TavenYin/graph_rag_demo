"""Retrieval integration tests for the local PostgreSQL + pgvector service."""

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
async def test_search_all_fuses_real_vector_and_fulltext_rows(database) -> None:
    """Changing retrieval to only use one source makes this test fail."""
    from graph_rag_demo.services.retrieval import RetrievalService
    from graph_rag_demo.tokenize_fts import tokenize_for_fts

    async with database.session() as session:
        document_id = (
            await session.execute(
                text(
                    "INSERT INTO kb_document (title, checksum) "
                    "VALUES ('retrieval test', repeat('a', 64)) RETURNING id"
                )
            )
        ).scalar_one()
        chunk_id = (
            await session.execute(
            text(
                "INSERT INTO kb_chunk "
                "(document_id, chunk_index, content, token_count, content_tsv, embedding) "
                "VALUES (:document_id, 0, :content, 2, "
                "to_tsvector('simple', :fts_tokens), CAST(:embedding AS vector)) "
                "RETURNING id"
            ),
            {
                "document_id": document_id,
                "content": "梦境功能需要完成任务解锁",
                "fts_tokens": tokenize_for_fts("梦境功能需要完成任务解锁"),
                "embedding": "[" + ",".join(["0.01"] * 1024) + "]",
            },
            )
        ).scalar_one()

    try:
        service = RetrievalService(database=database, per_retriever_top_k=5, final_top_n=5)
        results = await service.search_all(
            queries=["梦境功能解锁"], embeddings=[[0.01] * 1024]
        )
    finally:
        async with database.session() as session:
            await session.execute(text("DELETE FROM kb_document WHERE id = :id"), {"id": document_id})

    assert [result.chunk_id for result in results] == [chunk_id]
    assert {match.channel for match in results[0].matches} == {"vector", "fulltext"}
