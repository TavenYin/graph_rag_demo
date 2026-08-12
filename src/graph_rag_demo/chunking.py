"""Chinese-aware recursive text chunking measured with tiktoken."""

from __future__ import annotations

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from graph_rag_demo.models.chunk import TokenChunk


_ENCODING = tiktoken.get_encoding("cl100k_base")
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[TokenChunk]:
    """Split text on preferred Chinese boundaries within a token budget."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be at least 0 and less than chunk_size")
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_token_count,
        keep_separator="end",
    )
    return [
        TokenChunk(
            content=content,
            token_count=_token_count(content),
            index=index,
        )
        for index, content in enumerate(splitter.split_text(text))
    ]


def _token_count(text: str) -> int:
    return len(_ENCODING.encode(text))
