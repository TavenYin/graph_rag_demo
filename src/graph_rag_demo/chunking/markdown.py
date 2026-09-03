"""Parse Markdown AST nodes into the block model consumed by the chunker."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import mistune

from graph_rag_demo.chunking.models import (
    MarkdownBlock,
    MarkdownReference,
    ReferenceKind,
    Section,
)


_MARKDOWN = mistune.create_markdown(renderer="ast", plugins=["table"])
_RESERVED_PLACEHOLDER = re.compile(r"@@(?:LINK:link_|IMG:img_)\d+@@")


@dataclass
class _ReferenceCollector:
    """Assign readable document-local keys while rendering inline AST nodes."""

    counts: dict[ReferenceKind, int] = field(default_factory=lambda: {"link": 0, "image": 0})

    def create(
        self,
        kind: ReferenceKind,
        *,
        url: str,
        label: str,
        title: str | None,
    ) -> MarkdownReference:
        self.counts[kind] += 1
        prefix = "link" if kind == "link" else "img"
        return MarkdownReference(
            key=f"{prefix}_{self.counts[kind]}",
            kind=kind,
            url=url,
            label=label,
            title=title,
        )


def parse_sections(markdown: str) -> list[Section]:
    """Parse one Markdown document directly into heading-aware sections."""
    if _RESERVED_PLACEHOLDER.search(markdown):
        raise ValueError("Markdown contains a reserved reference placeholder")

    ast: list[dict[str, Any]] = _MARKDOWN(markdown)
    collector = _ReferenceCollector()
    headers: list[str | None] = [None] * 6
    header_references: list[tuple[MarkdownReference, ...]] = [()] * 6
    header_path: tuple[str, ...] = ()
    heading_levels: tuple[int, ...] = ()
    blocks: list[MarkdownBlock] = []
    sections: list[Section] = []

    def flush() -> None:
        if not blocks:
            return
        references = tuple(reference for group in header_references for reference in group)
        sections.append(Section(header_path, heading_levels, tuple(blocks), references))
        blocks.clear()

    for node in ast:
        if node["type"] == "heading":
            flush()
            level = int(node["attrs"]["level"])
            references: list[MarkdownReference] = []
            title = _render_inline(node.get("children", []), collector, references).strip()
            if not title:
                continue
            headers[level - 1] = title
            header_references[level - 1] = tuple(references)
            for position in range(level, len(headers)):
                headers[position] = None
                header_references[position] = ()
            heading_levels = tuple(
                position + 1 for position, header in enumerate(headers) if header is not None
            )
            header_path = tuple(header for header in headers if header is not None)
            continue

        block = _render_block(node, collector)
        if block is not None and block.content.strip():
            blocks.append(block)

    flush()
    return sections


def _render_block(node: dict[str, Any], collector: _ReferenceCollector) -> MarkdownBlock | None:
    node_type = node["type"]
    if node_type in {"blank_line", "heading"}:
        return None

    references: list[MarkdownReference] = []
    if node_type == "paragraph":
        content = _render_inline(node.get("children", []), collector, references)
        return MarkdownBlock(content, "paragraph", tuple(references))
    if node_type == "list":
        content = _render_list(node, collector, references)
        return MarkdownBlock(content, "list", tuple(references))
    if node_type == "table":
        content = _render_table(node, collector, references)
        return MarkdownBlock(content, "table", tuple(references))
    if node_type == "block_quote":
        content, nested_references = _render_blocks(node.get("children", []), collector)
        quoted = "\n".join(f"> {line}" if line else ">" for line in content.splitlines())
        return MarkdownBlock(quoted, "quote", nested_references)
    if node_type == "block_code":
        info = str(node.get("attrs", {}).get("info") or "")
        raw = str(node.get("raw", ""))
        if raw and not raw.endswith("\n"):
            raw += "\n"
        return MarkdownBlock(f"```{info}\n{raw}```", "code")

    # Unsupported block nodes are intentionally outside the current contract.
    return None


def _render_blocks(
    nodes: list[dict[str, Any]], collector: _ReferenceCollector
) -> tuple[str, tuple[MarkdownReference, ...]]:
    blocks = [block for node in nodes if (block := _render_block(node, collector)) is not None]
    return (
        "\n\n".join(block.content for block in blocks if block.content),
        tuple(reference for block in blocks for reference in block.references),
    )


def _render_inline(
    nodes: list[dict[str, Any]],
    collector: _ReferenceCollector,
    references: list[MarkdownReference],
) -> str:
    parts: list[str] = []
    for node in nodes:
        node_type = node["type"]
        if node_type == "text":
            parts.append(str(node.get("raw", "")))
        elif node_type == "codespan":
            parts.append(f"`{node.get('raw', '')}`")
        elif node_type == "strong":
            parts.append(_render_inline(node.get("children", []), collector, references))
        elif node_type == "emphasis":
            parts.append(_render_inline(node.get("children", []), collector, references))
        elif node_type == "softbreak":
            parts.append("\n")
        elif node_type == "linebreak":
            parts.append("  \n")
        elif node_type in {"link", "image"}:
            kind: ReferenceKind = "link" if node_type == "link" else "image"
            attrs = node.get("attrs", {})
            reference = collector.create(
                kind,
                url=str(attrs.get("url", "")),
                label=_plain_inline(node.get("children", [])),
                title=str(attrs["title"]) if attrs.get("title") is not None else None,
            )
            references.append(reference)
            parts.append(reference.placeholder)
        elif node_type == "inline_html":
            parts.append(str(node.get("raw", "")))
        else:
            parts.append(_render_inline(node.get("children", []), collector, references))
    return "".join(parts)


def _plain_inline(nodes: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for node in nodes:
        node_type = node["type"]
        if node_type in {"text", "codespan"}:
            parts.append(str(node.get("raw", "")))
        elif node_type in {"softbreak", "linebreak"}:
            parts.append("\n")
        else:
            parts.append(_plain_inline(node.get("children", [])))
    return "".join(parts)


def _render_list(
    node: dict[str, Any],
    collector: _ReferenceCollector,
    references: list[MarkdownReference],
    depth: int = 0,
) -> str:
    ordered = bool(node.get("attrs", {}).get("ordered"))
    lines: list[str] = []
    for number, item in enumerate(node.get("children", []), start=1):
        children = item.get("children", [])
        text_nodes = [child for child in children if child["type"] != "list"]
        nested = [child for child in children if child["type"] == "list"]
        text = "\n".join(
            _render_inline(child.get("children", []), collector, references) for child in text_nodes
        ).strip()
        marker = f"{number}." if ordered else "-"
        lines.append(f"{'  ' * depth}{marker} {text}".rstrip())
        for child in nested:
            lines.extend(_render_list(child, collector, references, depth + 1).splitlines())
    return "\n".join(lines)


def _render_table(
    node: dict[str, Any],
    collector: _ReferenceCollector,
    references: list[MarkdownReference],
) -> str:
    rows: list[list[str]] = []
    for part in node.get("children", []):
        table_rows = [part] if part["type"] == "table_head" else part.get("children", [])
        for row in table_rows:
            rows.append(
                [
                    _render_inline(cell.get("children", []), collector, references)
                    for cell in row.get("children", [])
                ]
            )
    if not rows:
        return ""
    header, *body = rows
    separator = ["---"] * len(header)
    return "\n".join("| " + " | ".join(row) + " |" for row in [header, separator, *body])
