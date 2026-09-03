"""Vector and Chinese full-text retrieval with one global weighted RRF."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from sqlalchemy import text
from sqlalchemy.engine import RowMapping

from graph_rag_demo.db import Database
from graph_rag_demo.models.retrieval import RankedChunk, SearchMatch, SearchResult
from graph_rag_demo.tokenize_fts import tokenize_for_fts


_LOGGER = logging.getLogger(__name__)
RRF_K = 60

_VECTOR_SEARCH = text(
    """
    SELECT
        id,
        content,
        metadata,
        1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
    FROM kb_chunk
    WHERE 1 - (embedding <=> CAST(:embedding AS vector)) > :min_similarity
    ORDER BY similarity DESC, id
    LIMIT :top_k
    """
)

_FULLTEXT_SEARCH = text(
    """
    SELECT
        id,
        content,
        metadata,
        ts_rank_cd(content_tsv, to_tsquery('simple', :query)) AS score
    FROM kb_chunk
    WHERE content_tsv @@ to_tsquery('simple', :query)
    ORDER BY ts_rank_cd(content_tsv, to_tsquery('simple', :query)) DESC, id
    LIMIT :top_k
    """
)


def weighted_rrf(
    rank_lists: Sequence[Sequence[RankedChunk]],
    weights: Sequence[float],
    top_n: int,
) -> list[SearchResult]:
    """Fuse every supplied ranking list once, retaining every score contribution."""
    if len(rank_lists) != len(weights):
        raise ValueError("weights must match rank_lists")
    if top_n < 0:
        raise ValueError("top_n must be at least 0")

    _LOGGER.info("rrf_fusion_started rank_list_count=%d top_n=%d", len(rank_lists), top_n)
    scores: dict[int, float] = defaultdict(float)
    contents: dict[int, str] = {}
    metadata_by_id: dict[int, dict[str, object]] = {}
    matches: dict[int, list[SearchMatch]] = defaultdict(list)

    for rank_list, weight in zip(rank_lists, weights, strict=True):
        if weight < 0:
            raise ValueError("weights must be non-negative")
        for rank, item in enumerate(rank_list, start=1):
            scores[item.chunk_id] += weight / (RRF_K + rank)
            contents.setdefault(item.chunk_id, item.content)
            metadata_by_id.setdefault(item.chunk_id, item.metadata)
            matches[item.chunk_id].append(item.match)

    ranked_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    results = [
        SearchResult(
            chunk_id=chunk_id,
            content=contents[chunk_id],
            metadata=metadata_by_id[chunk_id],
            score=scores[chunk_id],
            matches=tuple(matches[chunk_id]),
        )
        for chunk_id in ranked_ids[:top_n]
    ]
    _LOGGER.info(
        "rrf_fusion_complete rank_list_count=%d candidate_count=%d result_count=%d",
        len(rank_lists),
        len(scores),
        len(results),
    )
    for result in results:
        _LOGGER.debug(
            "rrf_result chunk_id=%d score=%s match_count=%d",
            result.chunk_id,
            result.score,
            len(result.matches),
        )
    return results


class RetrievalService:
    """Runs each query/retriever pair and globally fuses the resulting lists."""

    def __init__(
        self,
        *,
        database: Database,
        per_retriever_top_k: int = 10,
        final_top_n: int = 8,
        original_query_weight: float = 2.0,
        expansion_query_weight: float = 1.0,
        min_vector_similarity: float = 0.6,
    ) -> None:
        if per_retriever_top_k <= 0:
            raise ValueError("per_retriever_top_k must be positive")
        if final_top_n < 0:
            raise ValueError("final_top_n must be at least 0")
        if original_query_weight < 0 or expansion_query_weight < 0:
            raise ValueError("retrieval weights must be non-negative")
        if not -1 <= min_vector_similarity <= 1:
            raise ValueError("min_vector_similarity must be between -1 and 1")
        self._database = database
        self._per_retriever_top_k = per_retriever_top_k
        self._final_top_n = final_top_n
        self._original_query_weight = original_query_weight
        self._expansion_query_weight = expansion_query_weight
        self._min_vector_similarity = min_vector_similarity

    async def search_all(
        self, queries: Sequence[str], embeddings: Sequence[Sequence[float]]
    ) -> list[SearchResult]:
        """Retrieve every query, then make exactly one global fusion call."""
        if len(queries) != len(embeddings):
            raise ValueError("queries and embeddings must have the same length")
        if not queries:
            return []

        _LOGGER.info(
            "retrieval_started query_count=%d min_vector_similarity=%s",
            len(queries),
            self._min_vector_similarity,
        )
        rank_lists: list[list[RankedChunk]] = []
        weights: list[float] = []
        async with self._database.session() as session:
            for query_index, (query, embedding) in enumerate(zip(queries, embeddings, strict=True)):
                weight = (
                    self._original_query_weight
                    if query_index == 0
                    else self._expansion_query_weight
                )
                vector_rows = await session.execute(
                    _VECTOR_SEARCH,
                    {
                        "embedding": _vector_literal(embedding),
                        "min_similarity": self._min_vector_similarity,
                        "top_k": self._per_retriever_top_k,
                    },
                )
                vector_rows = vector_rows.mappings().all()
                _LOGGER.info(
                    "vector_retrieval_complete query_index=%d result_count=%d",
                    query_index,
                    len(vector_rows),
                )
                for rank, row in enumerate(vector_rows, start=1):
                    similarity = float(row["similarity"])
                    _LOGGER.debug(
                        "vector_retrieval_result query_index=%d rank=%d chunk_id=%s distance=%s similarity=%s",
                        query_index,
                        rank,
                        row["id"],
                        1 - similarity,
                        similarity,
                    )
                rank_lists.append(
                    _ranked_chunks(
                        vector_rows,
                        query_index=query_index,
                        query=query,
                        channel="vector",
                    )
                )
                weights.append(weight)

                fts_query = _to_or_tsquery(tokenize_for_fts(query))
                if fts_query:
                    fulltext_rows = await session.execute(
                        _FULLTEXT_SEARCH,
                        {"query": fts_query, "top_k": self._per_retriever_top_k},
                    )
                    fulltext_rows = fulltext_rows.mappings().all()
                    _LOGGER.info(
                        "fulltext_retrieval_complete query_index=%d result_count=%d",
                        query_index,
                        len(fulltext_rows),
                    )
                    for rank, row in enumerate(fulltext_rows, start=1):
                        _LOGGER.debug(
                            "fulltext_retrieval_result query_index=%d rank=%d chunk_id=%s score=%s",
                            query_index,
                            rank,
                            row["id"],
                            row["score"],
                        )
                    rank_lists.append(
                        _ranked_chunks(
                            fulltext_rows,
                            query_index=query_index,
                            query=query,
                            channel="fulltext",
                        )
                    )
                    weights.append(weight)

        return weighted_rrf(rank_lists, weights, self._final_top_n)


def _ranked_chunks(
    rows: Sequence[RowMapping],
    *,
    query_index: int,
    query: str,
    channel: Literal["vector", "fulltext"],
) -> list[RankedChunk]:
    return [
        RankedChunk(
            chunk_id=int(row["id"]),
            content=str(row["content"]),
            metadata=dict(row.get("metadata") or {}),
            match=SearchMatch(
                query_index=query_index,
                query=query,
                channel=channel,
                rank=rank,
            ),
        )
        for rank, row in enumerate(rows, start=1)
    ]


def _vector_literal(vector: Sequence[float]) -> str:
    if not vector:
        raise ValueError("query embedding must not be empty")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in vector):
        raise ValueError("query embedding values must be numeric")
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def _to_or_tsquery(terms: str) -> str:
    """Construct a safe OR tsquery from tokenizer output rather than model text."""
    return " | ".join(term for term in terms.split() if term.isalnum())
