from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json

import pytest
from sqlalchemy.exc import IntegrityError

from graph_rag_demo.services.knowledge import DuplicateDocumentError, KnowledgeService


@pytest.fixture(autouse=True)
def fts_tokenizer(monkeypatch):
    """Keep ingestion tests independent of the query-side tokenizer implementation."""
    import graph_rag_demo.services.knowledge as knowledge

    monkeypatch.setattr(knowledge, "tokenize_for_fts", lambda chunk: chunk)


class FakeEmbeddingClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.requests.append(texts)
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return [[float(index), 0.5] for index in range(len(texts))]


class FakeResult:
    def __init__(self, value: int) -> None:
        self._value = value

    def scalar_one(self) -> int:
        return self._value


class FakeSession:
    def __init__(self, database: "FakeDatabase") -> None:
        self._database = database
        self._pending_document: dict[str, object] | None = None
        self._pending_chunks: list[dict[str, object]] = []

    async def execute(self, statement, parameters=None):
        sql = str(statement)
        if "INSERT INTO kb_document" in sql:
            assert isinstance(parameters, dict)
            checksum = str(parameters["checksum"])
            if checksum in self._database.checksums:
                raise IntegrityError(sql, parameters, Exception("duplicate checksum"))
            self._pending_document = dict(parameters)
            return FakeResult(self._database.next_document_id)
        if "INSERT INTO kb_chunk" in sql:
            assert isinstance(parameters, list)
            self._pending_chunks = [dict(item) for item in parameters]
            return None
        raise AssertionError(f"unexpected SQL: {sql}")

    def commit(self) -> None:
        assert self._pending_document is not None
        document = dict(self._pending_document)
        document["metadata"] = json.loads(str(document["metadata"]))
        document["id"] = self._database.next_document_id
        self._database.documents.append(document)
        self._database.checksums.add(str(document["checksum"]))
        self._database.chunks.extend(
            {**row, "metadata": json.loads(str(row["metadata"]))}
            for row in self._pending_chunks
        )
        self._database.next_document_id += 1


class FakeDatabase:
    def __init__(self) -> None:
        self.documents: list[dict[str, object]] = []
        self.chunks: list[dict[str, object]] = []
        self.checksums: set[str] = set()
        self.next_document_id = 1
        self.transactions_started = 0

    @asynccontextmanager
    async def session(self) -> AsyncIterator[FakeSession]:
        self.transactions_started += 1
        session = FakeSession(self)
        try:
            yield session
        except Exception:
            raise
        else:
            session.commit()


@pytest.mark.asyncio
async def test_upload_does_not_open_a_database_transaction_when_embedding_fails() -> None:
    database = FakeDatabase()
    embedding_client = FakeEmbeddingClient(fail=True)
    service = KnowledgeService(
        database=database,
        embedding_client=embedding_client,
        chunk_size=4,
        chunk_overlap=1,
    )

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        await service.upload("first document", title="First")

    assert database.documents == []
    assert database.chunks == []
    assert database.transactions_started == 0


@pytest.mark.asyncio
async def test_upload_rejects_equivalent_cleaned_content_as_a_duplicate() -> None:
    database = FakeDatabase()
    embedding_client = FakeEmbeddingClient()
    service = KnowledgeService(
        database=database,
        embedding_client=embedding_client,
        chunk_size=20,
        chunk_overlap=0,
    )

    first_id = await service.upload("  Graph\tRAG\n", title="First")

    with pytest.raises(DuplicateDocumentError):
        await service.upload("Graph RAG", title="Duplicate")

    assert first_id == 1
    assert len(database.documents) == 1
    assert database.transactions_started == 2


@pytest.mark.asyncio
async def test_upload_persists_token_counts_for_all_chunks_in_one_embedding_batch() -> None:
    database = FakeDatabase()
    embedding_client = FakeEmbeddingClient()
    service = KnowledgeService(
        database=database,
        embedding_client=embedding_client,
        chunk_size=3,
        chunk_overlap=1,
    )

    document_id = await service.upload(
        "one two three four five six",
        title="Token document",
        metadata={"source": "unit-test"},
    )

    assert document_id == 1
    assert len(embedding_client.requests) == 1
    assert embedding_client.requests[0] == [chunk["content"] for chunk in database.chunks]
    assert [chunk["token_count"] for chunk in database.chunks] == [3, 3, 2]
    assert [chunk["chunk_index"] for chunk in database.chunks] == [0, 1, 2]
    assert all(chunk["document_id"] == document_id for chunk in database.chunks)
    assert all(chunk["metadata"] == {"source": "unit-test"} for chunk in database.chunks)
    assert database.transactions_started == 1


@pytest.mark.asyncio
async def test_upload_persists_fts_tokens_created_from_each_chunk(monkeypatch) -> None:
    import graph_rag_demo.services.knowledge as knowledge

    monkeypatch.setattr(
        knowledge,
        "tokenize_for_fts",
        lambda chunk: f"tokenized:{chunk}",
    )
    database = FakeDatabase()
    service = KnowledgeService(
        database=database,
        embedding_client=FakeEmbeddingClient(),
        chunk_size=20,
        chunk_overlap=0,
    )

    await service.upload("中文检索", title="FTS")

    assert database.chunks[0]["fts_tokens"] == "tokenized:中文检索"
