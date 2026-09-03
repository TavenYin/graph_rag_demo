"""Data structures for structure-aware Markdown chunking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BlockKind = Literal["paragraph", "list", "quote", "table", "code"]
ReferenceKind = Literal["link", "image"]


@dataclass(frozen=True)
class MarkdownReference:
    """One link or image occurrence extracted from the source Markdown."""

    key: str
    kind: ReferenceKind
    url: str
    label: str
    title: str | None

    @property
    def placeholder(self) -> str:
        prefix = "LINK" if self.kind == "link" else "IMG"
        return f"@@{prefix}:{self.key}@@"

    def to_metadata(self) -> dict[str, str | None]:
        metadata = {
            "key": self.key,
            "type": self.kind,
            "url": self.url,
            "title": self.title,
        }
        metadata["text" if self.kind == "link" else "alt"] = self.label
        return metadata


@dataclass(frozen=True)
class MarkdownBlock:
    content: str
    kind: BlockKind
    references: tuple[MarkdownReference, ...] = ()


@dataclass(frozen=True)
class Section:
    header_path: tuple[str, ...]
    heading_levels: tuple[int, ...]
    blocks: tuple[MarkdownBlock, ...]
    references: tuple[MarkdownReference, ...] = ()


@dataclass(frozen=True)
class DocumentChunk:
    content: str
    search_text: str
    token_count: int
    index: int
    metadata: dict[str, object] = field(default_factory=dict)
