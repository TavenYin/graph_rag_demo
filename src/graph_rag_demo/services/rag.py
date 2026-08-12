"""Query expansion, retrieval, evidence assembly, and answer orchestration."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import tiktoken
from pydantic import ValidationError

from graph_rag_demo.clients.embedding import EmbeddingClient, ModelResponseError
from graph_rag_demo.clients.llm import LLMClient
from graph_rag_demo.models.chat import ChatMessage
from graph_rag_demo.models.generation import (
    AnswerPayload,
    AskResult,
    QueryExpansionPayload,
)
from graph_rag_demo.models.retrieval import SearchResult
from graph_rag_demo.services.prompts import (
    build_answer_messages,
    build_expansion_messages,
    build_knowledge_xml,
)
from graph_rag_demo.services.retrieval import RetrievalService


_LOGGER = logging.getLogger(__name__)
_ENCODING = tiktoken.get_encoding("cl100k_base")


class RAGService:
    """Coordinates the baseline RAG flow without query planning or retries."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        embedding_client: EmbeddingClient,
        retrieval_service: RetrievalService,
        context_token_budget: int = 2_000,
    ) -> None:
        if context_token_budget < 0:
            raise ValueError("context_token_budget must be at least 0")
        self._llm_client = llm_client
        self._embedding_client = embedding_client
        self._retrieval_service = retrieval_service
        self._context_token_budget = context_token_budget

    async def answer(
        self, question: str, chat_context: Sequence[ChatMessage] = ()
    ) -> AskResult:
        """Expand once, retrieve once, then answer from the bounded evidence."""
        original_question = _normalize_query(question)
        if not original_question:
            raise ValueError("question must contain text")
        queries = await self._build_retrieval_queries(original_question, chat_context)
        embeddings = await self._embedding_client.embed(queries)
        results = await self._retrieval_service.search_all(queries, embeddings)
        _LOGGER.info(
            "rag_retrieval_completed query_count=%d result_count=%d",
            len(queries),
            len(results),
        )
        evidence, context_chunk_ids = _build_evidence_context(
            results, self._context_token_budget
        )
        answer = await self._generate_grounded_answer(
            original_question, chat_context, evidence
        )

        return AskResult(
            answer=answer.answer,
            used_chunk_ids=_filter_citations(answer.used_chunk_ids, context_chunk_ids),
        )

    async def _build_retrieval_queries(
        self, original_question: str, chat_context: Sequence[ChatMessage]
    ) -> list[str]:
        _LOGGER.info(
            "rag_expansion_started question_length=%d has_chat_context=%s",
            len(original_question),
            bool(chat_context),
        )
        try:
            response = await self._llm_client.complete(
                build_expansion_messages(original_question, chat_context),
                json_mode=True,
            )
            payload = _parse_query_expansion(response)
            queries = _normalize_expansions(original_question, payload.queries)
            _LOGGER.info("rag_expansion_completed query_count=%d", len(queries))
            _LOGGER.debug("rag_expanded_queries queries=%s", queries)
            return queries
        except Exception as error:
            _LOGGER.warning(
                "rag_expansion_fallback reason=%s query_count=1",
                type(error).__name__,
            )
            return [original_question]

    async def _generate_grounded_answer(
        self,
        question: str,
        chat_context: Sequence[ChatMessage],
        evidence: Sequence[SearchResult],
    ) -> AnswerPayload:
        _LOGGER.info("rag_answer_started question_length=%d", len(question))
        response = await self._llm_client.complete(
            build_answer_messages(question, chat_context, evidence),
            json_mode=True,
        )
        result = _parse_grounded_answer(response)
        _LOGGER.info(
            "rag_answer_completed citation_count=%d",
            len(result.used_chunk_ids),
        )
        return result


def _parse_query_expansion(content: str) -> QueryExpansionPayload:
    try:
        return QueryExpansionPayload.model_validate_json(content)
    except ValidationError as error:
        raise ModelResponseError("query expansion response is invalid") from error


def _parse_grounded_answer(content: str) -> AnswerPayload:
    try:
        return AnswerPayload.model_validate_json(content)
    except ValidationError as error:
        raise ModelResponseError("answer response is invalid") from error


def _normalize_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("question must be a string")
    return " ".join(query.split())


def _normalize_expansions(original_question: str, expansions: object) -> list[str]:
    if not isinstance(expansions, list) or any(not isinstance(query, str) for query in expansions):
        raise ValueError("expansions must be a list of strings")

    queries = [original_question]
    normalized_seen = {original_question.casefold()}
    for expansion in expansions:
        query = _normalize_query(expansion)
        normalized_key = query.casefold()
        if not query or normalized_key in normalized_seen:
            continue
        queries.append(query)
        normalized_seen.add(normalized_key)
        if len(queries) == 4:
            break
    return queries


def _build_evidence_context(
    results: list[SearchResult], token_budget: int
) -> tuple[list[SearchResult], set[int]]:
    selected: list[SearchResult] = []
    included_chunk_ids: set[int] = set()
    for result in results:
        candidate = build_knowledge_xml([*selected, result])
        if len(_ENCODING.encode(candidate)) > token_budget:
            continue
        selected.append(result)
        included_chunk_ids.add(result.chunk_id)
    return selected, included_chunk_ids


def _filter_citations(citations: object, allowed_chunk_ids: set[int]) -> list[int]:
    if not isinstance(citations, list):
        return []
    used_chunk_ids: list[int] = []
    for chunk_id in citations:
        if (
            isinstance(chunk_id, int)
            and not isinstance(chunk_id, bool)
            and chunk_id in allowed_chunk_ids
            and chunk_id not in used_chunk_ids
        ):
            used_chunk_ids.append(chunk_id)
    return used_chunk_ids
