"""Integration tests for the local PostgreSQL database boundary.

These tests intentionally use the real Docker Compose database. They stay
disabled in ordinary test runs so unit tests do not require Docker.
"""

from __future__ import annotations

import hashlib
import os

import pytest
import pytest_asyncio


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="set RUN_INTEGRATION_TESTS=1 to use the local PostgreSQL database",
)


@pytest_asyncio.fixture
async def database():
    from graph_rag_demo.config import Settings
    from graph_rag_demo.db import Database

    instance = Database.create(Settings.from_env())
    try:
        yield instance
    finally:
        await instance.close()


@pytest.mark.asyncio
async def test_healthcheck_executes_a_database_query(database) -> None:
    """Changing healthcheck to return a constant must make this test fail."""
    from sqlalchemy import event

    statements: list[str] = []

    def record_statement(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(
        database._engine.sync_engine, "before_cursor_execute", record_statement
    )
    try:
        assert await database.healthcheck() is True
    finally:
        event.remove(
            database._engine.sync_engine, "before_cursor_execute", record_statement
        )

    assert any(statement.strip() == "SELECT 1" for statement in statements)


@pytest.mark.asyncio
async def test_session_commits_a_successful_document_write(database) -> None:
    """Removing the transaction boundary must prevent the document from persisting."""
    from sqlalchemy import text

    checksum = hashlib.sha256(b"commit verification").hexdigest()

    async with database.session() as session:
        await session.execute(
            text(
                "INSERT INTO kb_document (title, checksum, metadata) "
                "VALUES (:title, :checksum, CAST(:metadata AS jsonb))"
            ),
            {
                "title": "commit verification",
                "checksum": checksum,
                "metadata": "{}",
            },
        )

    async with database.session() as session:
        row_count = await session.scalar(
            text("SELECT count(*) FROM kb_document WHERE checksum = :checksum"),
            {"checksum": checksum},
        )
        await session.execute(
            text("DELETE FROM kb_document WHERE checksum = :checksum"),
            {"checksum": checksum},
        )

    assert row_count == 1


@pytest.mark.asyncio
async def test_session_rolls_back_document_when_the_body_raises(database) -> None:
    """Removing the transaction boundary must leave this inserted row behind."""
    checksum = hashlib.sha256(b"rollback verification").hexdigest()
    from sqlalchemy import text

    with pytest.raises(RuntimeError, match="force rollback"):
        async with database.session() as session:
            await session.execute(
                text(
                    "INSERT INTO kb_document (title, checksum, metadata) "
                    "VALUES (:title, :checksum, CAST(:metadata AS jsonb))"
                ),
                {
                    "title": "rollback verification",
                    "checksum": checksum,
                    "metadata": "{}",
                },
            )
            raise RuntimeError("force rollback")

    async with database.session() as session:
        row_count = await session.scalar(
            text("SELECT count(*) FROM kb_document WHERE checksum = :checksum"),
            {"checksum": checksum},
        )

    assert row_count == 0
