"""FastAPI application wiring and lifecycle ownership."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from graph_rag_demo.clients.embedding import EmbeddingClient
from graph_rag_demo.clients.llm import LLMClient
from graph_rag_demo.config import Settings
from graph_rag_demo.db import Database
from graph_rag_demo.models.api import ApplicationServices
from graph_rag_demo.services.knowledge import KnowledgeService
from graph_rag_demo.services.rag import RAGService
from graph_rag_demo.services.retrieval import RetrievalService

from .routes import register_routes


def create_app(*, services: ApplicationServices | None = None) -> FastAPI:
    """Create an application without a mutable module-level service singleton."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active_services = services or _build_services(Settings.from_env())
        app.state.services = active_services
        try:
            yield
        finally:
            await active_services.aclose()

    app = FastAPI(
        title="Graph RAG Demo",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    register_routes(app)
    return app


def _build_services(settings: Settings) -> ApplicationServices:
    settings.validate()
    database = Database.create(settings)
    if not settings.use_real_clients:
        return ApplicationServices(database=database, closers=(database.close,))

    api_key = settings.dashscope_api_key
    if api_key is None:
        raise RuntimeError("DASHSCOPE_API_KEY is required for real clients")

    embedding_client = EmbeddingClient(
        api_key=api_key,
        base_url=settings.dashscope_base_url,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    llm_client = LLMClient(
        api_key=api_key,
        base_url=settings.dashscope_base_url,
        model=settings.llm_model,
    )
    retrieval_service = RetrievalService(
        database=database,
        min_vector_similarity=settings.vector_min_similarity,
    )
    return ApplicationServices(
        database=database,
        knowledge_service=KnowledgeService(
            database=database,
            embedding_client=embedding_client,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        rag_service=RAGService(
            llm_client=llm_client,
            embedding_client=embedding_client,
            retrieval_service=retrieval_service,
        ),
        closers=(database.close, embedding_client.aclose, llm_client.aclose),
    )
