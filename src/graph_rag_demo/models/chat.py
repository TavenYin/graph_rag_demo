from typing import Literal

from pydantic import BaseModel, Field, field_validator

from graph_rag_demo.text import clean_text


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4_000)

    @field_validator("content")
    @classmethod
    def content_must_contain_text(cls, value: str) -> str:
        cleaned = clean_text(value)
        if not cleaned:
            raise ValueError("content must contain text")
        return cleaned
