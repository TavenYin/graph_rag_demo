"""HTTP endpoint registration and request-to-service translation."""

from __future__ import annotations

import logging

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from sqlalchemy.exc import SQLAlchemyError

from graph_rag_demo.clients.embedding import ModelResponseError
from graph_rag_demo.chunking import split_text
from graph_rag_demo.config import Settings
from graph_rag_demo.models.api import (
    ApplicationServices,
    AskRequest,
    AskResponse,
    ChunkDebugRequest,
    ChunkDebugResponse,
    ChunkItem,
    DocumentRequest,
    DocumentResponse,
    HealthResponse,
)
from graph_rag_demo.services.knowledge import DuplicateDocumentError, _normalize_markdown

_LOGGER = logging.getLogger(__name__)


def register_routes(app: FastAPI) -> None:
    """Register this demo's HTTP routes on an application instance."""

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        try:
            is_healthy = await _services(request).database.healthcheck()
        except (OSError, SQLAlchemyError):
            _LOGGER.warning("Database health check failed")
            raise HTTPException(status_code=503, detail="Database unavailable") from None
        if not is_healthy:
            raise HTTPException(status_code=503, detail="Database unavailable")
        return HealthResponse(status="ok")

    @app.post("/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
    async def upload_document(payload: DocumentRequest, request: Request) -> DocumentResponse:
        knowledge_service = _services(request).knowledge_service
        if knowledge_service is None:
            raise HTTPException(status_code=503, detail="Model service is not configured")
        try:
            document_id = await knowledge_service.upload(
                payload.content,
                title=payload.title,
                metadata=payload.metadata,
            )
        except DuplicateDocumentError:
            raise HTTPException(status_code=409, detail="Document content already exists") from None
        except (httpx.HTTPError, ModelResponseError, OSError, SQLAlchemyError):
            _LOGGER.warning("Document upload dependency failed")
            raise HTTPException(status_code=503, detail="Model service unavailable") from None
        except Exception:
            _LOGGER.exception("Document upload failed")
            raise HTTPException(status_code=500, detail="Document upload failed") from None
        return DocumentResponse(document_id=document_id)

    @app.post("/debug/chunk", response_model=ChunkDebugResponse)
    async def debug_chunk(payload: ChunkDebugRequest) -> ChunkDebugResponse:
        settings = Settings.from_env()
        chunk_size = payload.chunk_size or settings.chunk_size
        chunk_overlap = (
            payload.chunk_overlap if payload.chunk_overlap is not None else settings.chunk_overlap
        )
        if not 0 <= chunk_overlap < chunk_size:
            raise HTTPException(
                status_code=422,
                detail="chunk_overlap must be at least 0 and less than chunk_size",
            )
        cleaned_content = _normalize_markdown(payload.content)
        if not cleaned_content:
            raise HTTPException(status_code=422, detail="content must contain text after cleaning")
        try:
            chunks = split_text(cleaned_content, chunk_size, chunk_overlap)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        return ChunkDebugResponse(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunks=[
                ChunkItem(
                    index=chunk.index,
                    content=chunk.content,
                    token_count=chunk.token_count,
                    metadata=dict(chunk.metadata),
                )
                for chunk in chunks
            ],
        )

    @app.post("/ask", response_model=AskResponse)
    async def ask(payload: AskRequest, request: Request) -> AskResponse:
        rag_service = _services(request).rag_service
        if rag_service is None:
            raise HTTPException(status_code=503, detail="Model service is not configured")
        try:
            result = await rag_service.answer(payload.question, payload.chat_context)
        except (httpx.HTTPError, ModelResponseError, OSError, SQLAlchemyError):
            _LOGGER.warning("RAG dependency failed")
            raise HTTPException(status_code=503, detail="Model service unavailable") from None
        except Exception:
            _LOGGER.exception("RAG request failed")
            raise HTTPException(status_code=500, detail="Answer generation failed") from None
        return AskResponse(answer=result.answer, used_chunk_ids=result.used_chunk_ids)


def _services(request: Request) -> ApplicationServices:
    return request.app.state.services
