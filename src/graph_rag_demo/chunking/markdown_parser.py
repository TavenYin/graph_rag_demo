"""Mistune AST parsing and Markdown block rendering."""

from __future__ import annotations

from typing import Any

import mistune

from graph_rag_demo.chunking.models import MarkdownBlock


_MARKDOWN = mistune.create_markdown(renderer="ast", plugins=["table"])
_ATOMIC_BLOCKS = {"list", "table", "block_quote", "block_code"}


def parse_markdown(markdown: str) -> list[dict[str, Any]]:
    """Parse Markdown to an AST without destructive whitespace cleaning."""
    return _MARKDOWN(markdown)


def render_block(node: dict[str, Any]) -> MarkdownBlock | None:
    """Render an AST block node into Markdown that remains readable to the LLM."""
    node_type = node["type"]
    if node_type in {"blank_line", "heading"}:
        return None
    if node_type == "paragraph":
        return MarkdownBlock(_render_inline(node.get("children", [])), splittable=True)
    if node_type == "list":
        return MarkdownBlock(_render_list(node), splittable=False)
    if node_type == "table":
        return MarkdownBlock(_render_table(node), splittable=False)
    if node_type == "block_quote":
        text = _render_blocks(node.get("children", []))
        return MarkdownBlock("\n".join(f"> {line}" if line else ">" for line in text.splitlines()), False)
    if node_type == "block_code":
        info = str(node.get("attrs", {}).get("info") or "")
        return MarkdownBlock(f"```{info}\n{node.get('raw', '')}```", False)
    return None


def plain_text(node: dict[str, Any]) -> str:
    return _render_inline(node.get("children", []))


def _render_blocks(nodes: list[dict[str, Any]]) -> str:
    blocks = [render_block(node) for node in nodes]
    return "\n\n".join(block.content for block in blocks if block and block.content)


def _render_inline(nodes: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for node in nodes:
        node_type = node["type"]
        if node_type in {"text", "codespan"}:
            parts.append(str(node.get("raw", "")))
        elif node_type == "softbreak":
            parts.append("\n")
        elif node_type == "linebreak":
            parts.append("  \n")
        elif node_type == "image":
            alt = _render_inline(node.get("children", []))
            url = str(node.get("attrs", {}).get("url", ""))
            parts.append(f"![{alt}]({url})")
        elif node_type == "link":
            label = _render_inline(node.get("children", []))
            url = str(node.get("attrs", {}).get("url", ""))
            parts.append(f"[{label}]({url})")
        else:
            parts.append(_render_inline(node.get("children", [])))
    return "".join(parts)


def _render_list(node: dict[str, Any], depth: int = 0) -> str:
    ordered = bool(node.get("attrs", {}).get("ordered"))
    lines: list[str] = []
    for number, item in enumerate(node.get("children", []), start=1):
        children = item.get("children", [])
        text_nodes = [child for child in children if child["type"] != "list"]
        nested = [child for child in children if child["type"] == "list"]
        text = "\n".join(
            _render_inline(child.get("children", [])) for child in text_nodes
        ).strip()
        marker = f"{number}." if ordered else "-"
        lines.append(f"{'  ' * depth}{marker} {text}".rstrip())
        lines.extend(_render_list(child, depth + 1).splitlines() for child in nested)
    return "\n".join(line for item in lines for line in (item if isinstance(item, list) else [item]))


def _render_table(node: dict[str, Any]) -> str:
    rows: list[list[str]] = []
    for part in node.get("children", []):
        table_rows = [part] if part["type"] == "table_head" else part.get("children", [])
        for row in table_rows:
            rows.append([_render_inline(cell.get("children", [])) for cell in row.get("children", [])])
    if not rows:
        return ""
    header, *body = rows
    separator = ["---"] * len(header)
    return "\n".join("| " + " | ".join(row) + " |" for row in [header, separator, *body])
