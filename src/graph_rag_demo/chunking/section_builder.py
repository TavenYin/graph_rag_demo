"""Build sections from heading-aware Markdown AST nodes."""

from __future__ import annotations

from typing import Any

from graph_rag_demo.chunking.markdown_parser import plain_text, render_block
from graph_rag_demo.chunking.models import MarkdownBlock, Section


def build_sections(ast: list[dict[str, Any]]) -> list[Section]:
    headers: list[str | None] = [None] * 6
    header_path: tuple[str, ...] = ()
    heading_levels: tuple[int, ...] = ()
    blocks: list[MarkdownBlock] = []
    sections: list[Section] = []

    def flush() -> None:
        if blocks:
            sections.append(
                Section(
                    header_path=header_path,
                    heading_levels=heading_levels,
                    blocks=tuple(blocks),
                )
            )
            blocks.clear()

    for node in ast:
        if node["type"] == "heading":
            flush()
            level = int(node["attrs"]["level"])
            title = plain_text(node).strip()
            if not title:
                continue
            headers[level - 1] = title
            for position in range(level, len(headers)):
                headers[position] = None
            heading_levels = tuple(
                position + 1 for position, header in enumerate(headers) if header is not None
            )
            header_path = tuple(header for header in headers if header is not None)
            continue
        block = render_block(node)
        if block and block.content.strip():
            blocks.append(block)
    flush()
    return sections
