"""Deterministic token-based text chunking."""

from bisect import bisect_right

import tiktoken

from graph_rag_demo.models.chunk import TokenChunk


_ENCODING = tiktoken.get_encoding("cl100k_base")


def _unicode_boundaries(tokens: list[int]) -> list[int]:
    _, offsets = _ENCODING.decode_with_offsets(tokens)
    boundaries = [0]
    boundaries.extend(
        index
        for index in range(1, len(tokens))
        if offsets[index] != offsets[index - 1]
    )
    boundaries.append(len(tokens))
    return boundaries


def _boundary_at_or_before(boundaries: list[int], token_index: int) -> int:
    return boundaries[bisect_right(boundaries, token_index) - 1]


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[TokenChunk]:
    """Split text into fixed-size tiktoken chunks with token overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be at least 0 and less than chunk_size")

    tokens = _ENCODING.encode(text)
    if not tokens:
        return []

    boundaries = _unicode_boundaries(tokens)
    chunks: list[TokenChunk] = []
    start = 0
    while start < len(tokens):
        end = _boundary_at_or_before(boundaries, start + chunk_size)
        if end == start:
            end = boundaries[bisect_right(boundaries, start)]

        chunk_tokens = tokens[start:end]
        content = _ENCODING.decode(chunk_tokens)
        chunks.append(
            TokenChunk(
                content=content,
                token_count=len(_ENCODING.encode(content)),
                index=len(chunks),
            )
        )
        if end == len(tokens):
            break

        next_start = _boundary_at_or_before(boundaries, end - chunk_overlap)
        if next_start <= start:
            next_start = boundaries[bisect_right(boundaries, start)]
        start = next_start

    return chunks
