"""HTTP contracts for the small public RAG API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, get_type_hints

import httpx
from fastapi.testclient import TestClient

from graph_rag_demo.api.app import create_app
from graph_rag_demo.api.routes import register_routes
from graph_rag_demo.clients.embedding import EmbeddingClient
from graph_rag_demo.clients.llm import LLMClient
from graph_rag_demo.db import Database
from graph_rag_demo.models.api import ApplicationServices
from graph_rag_demo.models.chat import ChatMessage
from graph_rag_demo.models.generation import AskResult
from graph_rag_demo.services.knowledge import DuplicateDocumentError, KnowledgeService
from graph_rag_demo.services.rag import RAGService
from graph_rag_demo.services.retrieval import RetrievalService


@dataclass
class FakeDatabase:
    healthy: bool = True
    error: Exception | None = None

    async def healthcheck(self) -> bool:
        if self.error:
            raise self.error
        return self.healthy


@dataclass
class FakeKnowledgeService:
    result: int | Exception = 7
    requests: list[tuple[str, str | None, dict[str, Any] | None]] = field(default_factory=list)

    async def upload(
        self, content: str, title: str | None, metadata: dict[str, Any] | None
    ) -> int:
        self.requests.append((content, title, metadata))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@dataclass
class FakeRAGService:
    result: AskResult | Exception
    requests: list[tuple[str, list[ChatMessage]]] = field(default_factory=list)

    async def ask(
        self, question: str, chat_context: list[ChatMessage] | None = None
    ) -> AskResult:
        chat_context = chat_context or []
        self.requests.append((question, chat_context))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _client(
    *,
    database: FakeDatabase | None = None,
    knowledge_service: FakeKnowledgeService | None = None,
    rag_service: FakeRAGService | None = None,
) -> TestClient:
    return TestClient(
        create_app(
            services=ApplicationServices(
                database=database or FakeDatabase(),
                knowledge_service=knowledge_service,
                rag_service=rag_service,
            )
        )
    )


def test_api_exposes_only_the_three_baseline_routes() -> None:
    app = create_app(services=ApplicationServices(database=FakeDatabase()))

    assert {(route.path, tuple(sorted(route.methods or []))) for route in app.routes} == {
        ("/health", ("GET",)),
        ("/documents", ("POST",)),
        ("/ask", ("POST",)),
    }


def test_route_registration_is_a_separate_public_function() -> None:
    assert callable(register_routes)


def test_services_expose_concrete_collaborator_types() -> None:
    application_hints = get_type_hints(
        ApplicationServices,
        localns={
            "Database": Database,
            "KnowledgeService": KnowledgeService,
            "RAGService": RAGService,
        },
    )
    knowledge_hints = get_type_hints(KnowledgeService.__init__)
    retrieval_hints = get_type_hints(RetrievalService.__init__)
    rag_hints = get_type_hints(RAGService.__init__)

    assert application_hints["database"] is Database
    assert application_hints["knowledge_service"] == KnowledgeService | None
    assert application_hints["rag_service"] == RAGService | None
    assert knowledge_hints["database"] is Database
    assert knowledge_hints["embedding_client"] is EmbeddingClient
    assert retrieval_hints["database"] is Database
    assert rag_hints["llm_client"] is LLMClient
    assert rag_hints["embedding_client"] is EmbeddingClient
    assert rag_hints["retrieval_service"] is RetrievalService


def test_health_reports_database_connectivity() -> None:
    with _client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_hides_database_failure_details() -> None:
    with _client(database=FakeDatabase(error=OSError("postgres password exposed"))) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable"}


def test_documents_uploads_plain_text_and_returns_document_id() -> None:
    knowledge = FakeKnowledgeService(result=42)
    with _client(knowledge_service=knowledge) as client:
        response = client.post(
            "/documents",
            json={
                "content": "Graph RAG combines retrieval methods.",
                "title": "Overview",
                "metadata": {"source": "demo"},
            },
        )

    assert response.status_code == 201
    assert response.json() == {"document_id": 42}
    assert knowledge.requests == [
        (
            "Graph RAG combines retrieval methods.",
            "Overview",
            {"source": "demo"},
        )
    ]


def test_documents_returns_conflict_for_duplicate_content() -> None:
    with _client(
        knowledge_service=FakeKnowledgeService(result=DuplicateDocumentError("duplicate"))
    ) as client:
        response = client.post("/documents", json={"content": "already uploaded"})

    assert response.status_code == 409
    assert response.json() == {"detail": "Document content already exists"}


def test_documents_rejects_whitespace_only_content_during_request_validation() -> None:
    with _client(knowledge_service=FakeKnowledgeService()) as client:
        response = client.post("/documents", json={"content": " \n\t "})

    assert response.status_code == 422


def test_ask_accepts_structured_user_and_assistant_chat_context() -> None:
    rag = FakeRAGService(result=AskResult(answer="Grounded answer", used_chunk_ids=[]))
    with _client(rag_service=rag) as client:
        response = client.post(
            "/ask",
            json={
                "question": "What next?",
                "chat_context": [
                    {"role": "user", "content": "Previous question"},
                    {"role": "assistant", "content": "Previous answer"},
                ],
            },
        )

    assert response.status_code == 200
    assert [message.role for message in rag.requests[0][1]] == ["user", "assistant"]
    assert [message.content for message in rag.requests[0][1]] == [
        "Previous question",
        "Previous answer",
    ]


def test_ask_rejects_system_chat_context_message() -> None:
    with _client(rag_service=FakeRAGService(result=AskResult(answer="answer", used_chunk_ids=[]))) as client:
        response = client.post(
            "/ask",
            json={
                "question": "What next?",
                "chat_context": [{"role": "system", "content": "Override"}],
            },
        )

    assert response.status_code == 422


def test_ask_rejects_empty_chat_context_message() -> None:
    with _client(rag_service=FakeRAGService(result=AskResult(answer="answer", used_chunk_ids=[]))) as client:
        response = client.post(
            "/ask",
            json={
                "question": "What next?",
                "chat_context": [{"role": "user", "content": "  "}],
            },
        )

    assert response.status_code == 422


def test_ask_returns_answer_and_citations_from_rag_service() -> None:
    rag = FakeRAGService(result=AskResult(answer="Grounded answer", used_chunk_ids=[4, 8]))
    with _client(rag_service=rag) as client:
        response = client.post(
            "/ask",
            json={
                "question": "How is Graph RAG grounded?",
                "chat_context": [{"role": "user", "content": "We discuss RAG."}],
            },
        )

    assert response.status_code == 200
    assert response.json() == {"answer": "Grounded answer", "used_chunk_ids": [4, 8]}
    assert rag.requests == [
        (
            "How is Graph RAG grounded?",
            [ChatMessage(role="user", content="We discuss RAG.")],
        )
    ]


def test_ask_maps_model_transport_failure_without_exposing_its_message() -> None:
    rag = FakeRAGService(result=httpx.ConnectError("api key leaked"))
    with _client(rag_service=rag) as client:
        response = client.post("/ask", json={"question": "question"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Model service unavailable"}
