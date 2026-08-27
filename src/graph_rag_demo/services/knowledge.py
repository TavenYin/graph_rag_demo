"""Document cleaning, token chunking, embedding, and atomic persistence."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from graph_rag_demo.chunking import split_text
from graph_rag_demo.clients.embedding import EmbeddingClient
from graph_rag_demo.db import Database
from graph_rag_demo.text import clean_text
from graph_rag_demo.tokenize_fts import tokenize_for_fts


class DuplicateDocumentError(ValueError):
    """Raised when another upload already stored the cleaned document text."""


_INSERT_DOCUMENT = text(
    """
    INSERT INTO kb_document (title, checksum, metadata)
    VALUES (:title, :checksum, CAST(:metadata AS jsonb))
    RETURNING id
    """
)

_INSERT_CHUNKS = text(
    """
    INSERT INTO kb_chunk (
        document_id, chunk_index, content, token_count, content_tsv, embedding, metadata
    )
    VALUES (
        :document_id, :chunk_index, :content, :token_count,
        to_tsvector('simple', :fts_tokens), CAST(:embedding AS vector), CAST(:metadata AS jsonb)
    )
    """
)


class KnowledgeService:
    """Persists a fully embedded document in one database transaction."""

    def __init__(
        self,
        *,
        database: Database,
        embedding_client: EmbeddingClient,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._database = database
        self._embedding_client = embedding_client
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def upload(
        self,
        content: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Clean, embed, and atomically save a document and its token chunks."""
        cleaned_content = _normalize_markdown(content)
        if not cleaned_content:
            raise ValueError("content must contain text after cleaning")

        checksum = hashlib.sha256(cleaned_content.encode("utf-8")).hexdigest()
        chunks = split_text(cleaned_content, self._chunk_size, self._chunk_overlap)
        embeddings = await self._embedding_client.embed([chunk.content for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise ValueError("embedding result count must match chunk count")

        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        chunk_rows = [
            {
                "chunk_index": chunk.index,
                "content": chunk.content,
                "token_count": chunk.token_count,
                "fts_tokens": tokenize_for_fts(chunk.content),
                "embedding": _vector_literal(embedding),
                "metadata": json.dumps({**(metadata or {}), "chunk": chunk.metadata}, ensure_ascii=False),
            }
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        try:
            async with self._database.session() as session:
                document_result = await session.execute(
                    _INSERT_DOCUMENT,
                    {"title": title, "checksum": checksum, "metadata": metadata_json},
                )
                document_id = document_result.scalar_one()
                for row in chunk_rows:
                    row["document_id"] = document_id
                await session.execute(_INSERT_CHUNKS, chunk_rows)
        except IntegrityError as error:
            raise DuplicateDocumentError("a document with the same cleaned content already exists") from error

        return document_id


def _vector_literal(vector: list[float]) -> str:
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in vector):
        raise ValueError("embedding values must be numeric")
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def _normalize_markdown(content: str) -> str:
    """Normalize line endings and unsafe characters without changing Markdown indentation."""
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if not _contains_markdown_structure(normalized):
        return clean_text(normalized)
    return normalized.strip()


_MARKDOWN_STRUCTURE = re.compile(r"(?m)^(#{1,6}\s|[-*+]\s|\d+\.\s|>|```|\|.*\|$)")


def _contains_markdown_structure(content: str) -> bool:
    return bool(_MARKDOWN_STRUCTURE.search(content))
