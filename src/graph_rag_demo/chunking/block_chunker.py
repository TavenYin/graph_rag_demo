"""Chunk Markdown section blocks using a body-only token budget."""

from __future__ import annotations

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from graph_rag_demo.chunking.models import DocumentChunk, MarkdownBlock, Section


_ENCODING = tiktoken.get_encoding("cl100k_base")
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def chunk_blocks(sections: list[Section], chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]:
    """Create chunks while keeping non-paragraph blocks indivisible."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be at least 0 and less than chunk_size")

    chunks: list[DocumentChunk] = []
    for section in sections:
        pending: list[str] = []
        pending_tokens = 0

        def flush() -> None:
            nonlocal pending_tokens
            if pending:
                chunks.append(
                    _document_chunk(
                        section.header_path, section.heading_levels, "\n\n".join(pending), len(chunks)
                    )
                )
                pending.clear()
                pending_tokens = 0

        for block in section.blocks:
            parts = _split_paragraph(block, chunk_size, chunk_overlap)
            for part in parts:
                part_tokens = _token_count(part.content)
                if part_tokens > chunk_size:
                    flush()
                    chunks.append(
                        _document_chunk(
                            section.header_path, section.heading_levels, part.content, len(chunks), oversized=True
                        )
                    )
                elif pending and pending_tokens + part_tokens > chunk_size:
                    flush()
                    pending.append(part.content)
                    pending_tokens = part_tokens
                else:
                    pending.append(part.content)
                    pending_tokens += part_tokens
        flush()
    return chunks


def _split_paragraph(block: MarkdownBlock, chunk_size: int, chunk_overlap: int) -> list[MarkdownBlock]:
    if not block.splittable or _token_count(block.content) <= chunk_size:
        return [block]
    splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_token_count,
        keep_separator="end",
    )
    return [MarkdownBlock(content=content, splittable=True) for content in splitter.split_text(block.content)]


def _document_chunk(
    header_path: tuple[str, ...],
    heading_levels: tuple[int, ...],
    body: str,
    index: int,
    oversized: bool = False,
) -> DocumentChunk:
    headers = "\n".join(
        f"{'#' * level} {title}" for level, title in zip(heading_levels, header_path, strict=True)
    )
    content = f"{headers}\n\n{body}" if headers else body
    metadata: dict[str, object] = {
        "header_path": list(header_path),
        "h1": _header_at_level(header_path, heading_levels, 1),
        "h2": _header_at_level(header_path, heading_levels, 2),
        "h3": _header_at_level(header_path, heading_levels, 3),
    }
    if oversized:
        metadata["oversized_block"] = True
    return DocumentChunk(content=content, token_count=_token_count(body), index=index, metadata=metadata)


def _token_count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _header_at_level(header_path: tuple[str, ...], heading_levels: tuple[int, ...], level: int) -> str | None:
    for heading_level, title in zip(heading_levels, header_path, strict=True):
        if heading_level == level:
            return title
    return None
