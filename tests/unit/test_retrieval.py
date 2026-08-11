"""Behavior tests for retrieval fusion and its minimal provenance."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from graph_rag_demo.models.retrieval import RankedChunk, SearchMatch
from graph_rag_demo.services.retrieval import RetrievalService, weighted_rrf
from graph_rag_demo.tokenize_fts import tokenize_for_fts


def _hit(
    chunk_id: int,
    content: str,
    *,
    query_index: int,
    query: str,
    channel: str,
    rank: int,
) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        content=content,
        match=SearchMatch(
            query_index=query_index,
            query=query,
            channel=channel,
            rank=rank,
        ),
    )


def test_weighted_rrf_accumulates_each_matching_list_and_deduplicates_results() -> None:
    vector = [
        _hit(1, "shared", query_index=0, query="original", channel="vector", rank=1),
        _hit(2, "single", query_index=0, query="original", channel="vector", rank=2),
    ]
    fulltext = [
        _hit(1, "shared", query_index=0, query="original", channel="fulltext", rank=1),
    ]

    results = weighted_rrf([vector, fulltext], weights=[1.0, 1.0], top_n=10)

    assert [result.chunk_id for result in results] == [1, 2]
    assert results[0].score == pytest.approx(2 / 61)
    assert results[0].matches == (vector[0].match, fulltext[0].match)


def test_weighted_rrf_logs_final_scores(caplog) -> None:
    caplog.set_level("DEBUG", logger="graph_rag_demo.services.retrieval")
    vector = [_hit(1, "shared", query_index=0, query="original", channel="vector", rank=1)]

    weighted_rrf([vector], weights=[2.0], top_n=1)

    assert "rrf_fusion_complete" in caplog.text
    assert "chunk_id=1" in caplog.text
    assert "score=0.03278688524590164" in caplog.text


def test_weighted_rrf_original_weight_outranks_an_equal_expansion_match() -> None:
    original = [_hit(1, "original", query_index=0, query="q", channel="vector", rank=1)]
    expansion = [_hit(2, "expanded", query_index=1, query="q2", channel="vector", rank=1)]

    results = weighted_rrf([original, expansion], weights=[2.0, 1.0], top_n=10)

    assert [result.chunk_id for result in results] == [1, 2]


def test_weighted_rrf_rejects_a_weight_for_only_some_rank_lists() -> None:
    rank_lists = [[_hit(1, "one", query_index=0, query="q", channel="vector", rank=1)]]

    with pytest.raises(ValueError, match="weights must match rank_lists"):
        weighted_rrf(rank_lists, weights=[], top_n=1)


def test_tokenize_for_fts_returns_whitespace_separated_meaningful_terms() -> None:
    terms = tokenize_for_fts("这是一个梦境功能解锁条件的测试")

    assert terms == terms.strip()
    assert "  " not in terms
    assert "的" not in terms.split()
    assert any(term in {"梦境", "功能", "解锁", "条件", "测试"} for term in terms.split())


class _Result:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def mappings(self) -> "_Result":
        return self

    def all(self) -> list[dict[str, object]]:
        return self._rows


class _Session:
    def __init__(
        self,
        responses: list[list[dict[str, object]]],
        executions: list[tuple[object, dict[str, object]]],
    ) -> None:
        self._responses = iter(responses)
        self._executions = executions

    async def execute(self, statement, params):
        self._executions.append((statement, params))
        return _Result(next(self._responses))


class _Database:
    def __init__(self, responses: list[list[dict[str, object]]]) -> None:
        self._responses = responses
        self.executions: list[tuple[object, dict[str, object]]] = []

    @asynccontextmanager
    async def session(self):
        yield _Session(self._responses, self.executions)


@pytest.mark.asyncio
async def test_search_all_collects_all_lists_before_one_global_fusion(monkeypatch) -> None:
    """Replacing global fusion with per-query fusion would call this twice."""
    from graph_rag_demo.services import retrieval

    responses = [
        [{"id": 1, "content": "original vector", "similarity": 0.75}],
        [{"id": 2, "content": "original fulltext", "score": 0.8}],
        [{"id": 3, "content": "expansion vector", "similarity": 0.65}],
        [{"id": 4, "content": "expansion fulltext", "score": 0.6}],
    ]
    service = RetrievalService(database=_Database(responses), per_retriever_top_k=5, final_top_n=5)
    calls: list[tuple[int, list[float]]] = []
    actual_fusion = retrieval.weighted_rrf

    def spy(rank_lists, weights, top_n):
        calls.append((len(rank_lists), list(weights)))
        return actual_fusion(rank_lists, weights, top_n)

    monkeypatch.setattr(retrieval, "weighted_rrf", spy)

    results = await service.search_all(
        queries=["original question", "expansion question"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
    )

    assert len(calls) == 1
    assert calls == [(4, [2.0, 2.0, 1.0, 1.0])]
    assert [result.chunk_id for result in results] == [1, 2, 3, 4]
    assert results[0].matches[0].query == "original question"
    assert results[-1].matches[0].query == "expansion question"


@pytest.mark.asyncio
async def test_search_all_logs_vector_and_fulltext_scores(caplog) -> None:
    from graph_rag_demo.services import retrieval

    caplog.set_level("DEBUG", logger="graph_rag_demo.services.retrieval")
    service = RetrievalService(
        database=_Database(
            [
                [{"id": 1, "content": "vector", "similarity": 0.75}],
                [{"id": 2, "content": "fulltext", "score": 0.8}],
            ]
        ),
        per_retriever_top_k=5,
        final_top_n=5,
    )

    await service.search_all(queries=["question"], embeddings=[[0.1, 0.2]])

    assert "vector_retrieval_result" in caplog.text
    assert "similarity=0.75" in caplog.text
    assert "distance=0.25" in caplog.text
    assert "fulltext_retrieval_result" in caplog.text
    assert "score=0.8" in caplog.text
    assert "rrf_fusion_complete" in caplog.text


@pytest.mark.asyncio
async def test_search_all_filters_vector_results_by_min_similarity() -> None:
    database = _Database(
        [
            [{"id": 1, "content": "vector", "similarity": 0.75}],
            [{"id": 2, "content": "fulltext", "score": 0.8}],
        ]
    )
    service = RetrievalService(
        database=database,
        per_retriever_top_k=5,
        final_top_n=5,
        min_vector_similarity=0.6,
    )

    await service.search_all(queries=["question"], embeddings=[[0.1, 0.2]])

    vector_statement, vector_params = database.executions[0]
    assert ":min_similarity" in vector_statement.text
    assert vector_params["min_similarity"] == 0.6
