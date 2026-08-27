"""Data structures for structure-aware Markdown chunking."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MarkdownBlock:
    content: str
    splittable: bool


@dataclass(frozen=True)
class Section:
    header_path: tuple[str, ...]
    heading_levels: tuple[int, ...]
    blocks: tuple[MarkdownBlock, ...]


@dataclass(frozen=True)
class DocumentChunk:
    content: str
    token_count: int
    index: int
    metadata: dict[str, object] = field(default_factory=dict)
