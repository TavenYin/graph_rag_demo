from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from graph_rag_demo.models.chat import ChatMessage
from graph_rag_demo.text import clean_text

if TYPE_CHECKING:
    from graph_rag_demo.db import Database
    from graph_rag_demo.services.knowledge import KnowledgeService
    from graph_rag_demo.services.rag import RAGService


@dataclass
class ApplicationServices:
    database: "Database"
    knowledge_service: "KnowledgeService | None" = None
    rag_service: "RAGService | None" = None
    closers: tuple[Callable[[], Awaitable[None]], ...] = ()

    async def aclose(self) -> None:
        for close in reversed(self.closers):
            await close()


class HealthResponse(BaseModel):
    status: str


class DocumentRequest(BaseModel):
    content: str = Field(min_length=1)
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def content_must_contain_text(cls, value: str) -> str:
        if not clean_text(value):
            raise ValueError("content must contain text")
        return value


class DocumentResponse(BaseModel):
    document_id: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    chat_context: list[ChatMessage] = Field(default_factory=list, max_length=20)

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        if not clean_text(value):
            raise ValueError("question must contain text")
        return value


class AskResponse(BaseModel):
    answer: str
    used_chunk_ids: list[int]
