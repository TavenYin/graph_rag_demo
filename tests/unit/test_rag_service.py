"""Behavior tests for query expansion and answer orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
import tiktoken

from graph_rag_demo.models.chat import ChatMessage
from graph_rag_demo.models.generation import AnswerPayload
from graph_rag_demo.models.retrieval import SearchResult
from graph_rag_demo.services.rag import RAGService


@dataclass
class FakeLLM:
    expansions: object
    answer_payload: AnswerPayload

    def __post_init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], bool]] = []

    async def complete(
        self, messages: list[dict[str, str]], *, json_mode: bool = False
    ) -> str:
        self.calls.append((messages, json_mode))
        if len(self.calls) == 1:
            if isinstance(self.expansions, Exception):
                raise self.expansions
            return json.dumps({"queries": self.expansions}, ensure_ascii=False)
        return json.dumps(
            {
                "answer": self.answer_payload.answer,
                "used_chunk_ids": self.answer_payload.used_chunk_ids,
            }
        )


@dataclass
class FakeEmbeddingClient:
    vectors: list[list[float]]

    def __post_init__(self) -> None:
        self.requests: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.requests.append(texts)
        return self.vectors


@dataclass
class FakeRetrievalService:
    results: list[SearchResult]

    def __post_init__(self) -> None:
        self.requests: list[tuple[list[str], list[list[float]]]] = []

    async def search_all(
        self, queries: list[str], embeddings: list[list[float]]
    ) -> list[SearchResult]:
        self.requests.append((queries, embeddings))
        return self.results


def _result(chunk_id: int, content: str) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, content=content, score=1.0, matches=())


@pytest.mark.asyncio
async def test_answer_keeps_original_question_first_and_uses_only_three_normalized_expansions() -> None:
    llm = FakeLLM(
        expansions=["  Graph RAG  ", "graph   rag", "", "实体检索", "关系查询", "第四条"],
        answer_payload=AnswerPayload(answer="answer", used_chunk_ids=[]),
    )
    embedding = FakeEmbeddingClient(vectors=[[0.1]] * 4)
    retrieval = FakeRetrievalService(results=[])
    service = RAGService(
        llm_client=llm,
        embedding_client=embedding,
        retrieval_service=retrieval,
    )

    await service.answer(
        "Graph RAG",
        chat_context=[ChatMessage(role="user", content="recent chat")],
    )

    assert embedding.requests == [["Graph RAG", "实体检索", "关系查询", "第四条"]]
    assert retrieval.requests == [
        (["Graph RAG", "实体检索", "关系查询", "第四条"], [[0.1]] * 4)
    ]
    assert llm.calls[0][1] is True
    assert llm.calls[0][0][0]["role"] == "system"
    assert "检索" in llm.calls[0][0][0]["content"]
    assert llm.calls[0][0][1] == {"role": "user", "content": "recent chat"}
    assert llm.calls[0][0][-1] == {"role": "user", "content": "Graph RAG"}


@pytest.mark.asyncio
async def test_answer_falls_back_to_original_question_when_expansion_fails() -> None:
    llm = FakeLLM(
        expansions=ValueError("malformed model response"),
        answer_payload=AnswerPayload(answer="answer", used_chunk_ids=[]),
    )
    embedding = FakeEmbeddingClient(vectors=[[0.1]])
    retrieval = FakeRetrievalService(results=[])
    service = RAGService(
        llm_client=llm,
        embedding_client=embedding,
        retrieval_service=retrieval,
    )

    result = await service.answer("original question")

    assert result.answer == "answer"
    assert embedding.requests == [["original question"]]
    assert retrieval.requests == [(["original question"], [[0.1]])]


@pytest.mark.asyncio
async def test_answer_logs_expansion_and_retrieval_stages(caplog) -> None:
    caplog.set_level("DEBUG", logger="graph_rag_demo.services.rag")
    llm = FakeLLM(
        expansions=["expanded"],
        answer_payload=AnswerPayload(answer="answer", used_chunk_ids=[]),
    )
    service = RAGService(
        llm_client=llm,
        embedding_client=FakeEmbeddingClient(vectors=[[0.1], [0.2]]),
        retrieval_service=FakeRetrievalService(results=[]),
    )

    await service.answer("question")

    assert "rag_expansion_started" in caplog.text
    assert "rag_expansion_completed" in caplog.text
    assert "rag_retrieval_completed" in caplog.text
    assert "rag_answer_completed" in caplog.text


@pytest.mark.asyncio
async def test_answer_builds_context_within_the_token_budget() -> None:
    short = "A short source."
    too_large = "very long evidence " * 40
    llm = FakeLLM(
        expansions=[],
        answer_payload=AnswerPayload(answer="grounded", used_chunk_ids=[1, 2]),
    )
    embedding = FakeEmbeddingClient(vectors=[[0.1]])
    retrieval = FakeRetrievalService(results=[_result(1, short), _result(2, too_large)])
    budget = 40
    service = RAGService(
        llm_client=llm,
        embedding_client=embedding,
        retrieval_service=retrieval,
        context_token_budget=budget,
    )

    result = await service.answer("question")

    context = llm.calls[1][0][1]["content"]
    assert "<knowledge>" in context
    assert '<chunk id="1">' in context
    assert too_large not in context
    assert len(tiktoken.get_encoding("cl100k_base").encode(context)) <= budget
    assert result.used_chunk_ids == [1]


@pytest.mark.asyncio
async def test_answer_escapes_special_characters_in_knowledge_xml() -> None:
    llm = FakeLLM(
        expansions=[],
        answer_payload=AnswerPayload(answer="grounded", used_chunk_ids=[1]),
    )
    service = RAGService(
        llm_client=llm,
        embedding_client=FakeEmbeddingClient(vectors=[[0.1]]),
        retrieval_service=FakeRetrievalService(
            results=[_result(1, "<unsafe>& \"quoted\"")]
        ),
        context_token_budget=100,
    )

    await service.answer("question")

    answer_content = llm.calls[1][0][-1]["content"]
    assert "&lt;unsafe&gt;&amp; \"quoted\"" in answer_content


@pytest.mark.asyncio
async def test_answer_filters_model_citations_that_are_not_in_the_context() -> None:
    llm = FakeLLM(
        expansions=[],
        answer_payload=AnswerPayload(answer="grounded", used_chunk_ids=[2, 99, 2, 1]),
    )
    embedding = FakeEmbeddingClient(vectors=[[0.1]])
    retrieval = FakeRetrievalService(results=[_result(1, "first"), _result(2, "second")])
    service = RAGService(
        llm_client=llm,
        embedding_client=embedding,
        retrieval_service=retrieval,
        context_token_budget=100,
    )

    result = await service.answer("question")

    assert result.answer == "grounded"
    assert result.used_chunk_ids == [2, 1]
