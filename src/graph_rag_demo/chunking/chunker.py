"""Turn parsed Markdown sections into final token-bounded chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from graph_rag_demo.chunking.markdown import parse_sections
from graph_rag_demo.chunking.models import (
    DocumentChunk,
    MarkdownBlock,
    MarkdownReference,
    Section,
)
from graph_rag_demo.chunking.table import split_table


_ENCODING = tiktoken.get_encoding("cl100k_base")
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
_LIST_PREFIX = re.compile(r"^(\s*(?:[-*+]|\d+\.)\s+)(.*)$")
_QUOTE_PREFIX = re.compile(r"^(>\s?)(.*)$")
_CODE_SPAN = re.compile(r"`[^`\n]*`")


@dataclass(frozen=True)
class _ChunkBody:
    content: str
    references: tuple[MarkdownReference, ...] = ()


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]:
    """Split Markdown into heading-enriched chunks and associated references."""
    _validate_config(chunk_size, chunk_overlap)
    if not text:
        return []
    return _chunk_sections(parse_sections(text), chunk_size, chunk_overlap)


def _chunk_sections(
    sections: list[Section], chunk_size: int, chunk_overlap: int
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for section in sections:
        bodies: list[_ChunkBody] = []
        for run in _group_runs(section.blocks):
            kind = run[0].kind
            references = tuple(reference for block in run for reference in block.references)
            if kind == "paragraph":
                merged = "\n\n".join(block.content for block in run)
                bodies.extend(
                    _ChunkBody(part, _references_in_content(part, references))
                    for part in _split_prose(merged, references, chunk_size, chunk_overlap)
                )
            elif kind == "list":
                _append_structured_parts(
                    bodies,
                    _split_prefixed_lines(
                        run[0].content,
                        references,
                        _LIST_PREFIX,
                        chunk_size,
                        chunk_overlap,
                    ),
                    references,
                    chunk_size,
                )
            elif kind == "quote":
                _append_structured_parts(
                    bodies,
                    _split_prefixed_lines(
                        run[0].content,
                        references,
                        _QUOTE_PREFIX,
                        chunk_size,
                        chunk_overlap,
                    ),
                    references,
                    chunk_size,
                )
            elif kind == "table":
                leading = _take_short_leading_body(bodies, chunk_size)
                combined_references = (*leading.references, *references)
                bodies.extend(
                    _ChunkBody(part, _references_in_content(part, combined_references))
                    for part in split_table(
                        run[0].content,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=_token_count,
                        pre_text=leading.content,
                    )
                )
            else:
                bodies.append(_ChunkBody(run[0].content, references))

        for body in bodies:
            if not body.content.strip():
                continue
            chunks.append(_document_chunk(section, body, len(chunks), chunk_size))
    return chunks


def _group_runs(blocks: tuple[MarkdownBlock, ...]) -> list[list[MarkdownBlock]]:
    runs: list[list[MarkdownBlock]] = []
    for block in blocks:
        if block.kind == "paragraph" and runs and runs[-1][0].kind == "paragraph":
            runs[-1].append(block)
        else:
            runs.append([block])
    return runs


def _split_prose(
    text: str,
    references: tuple[MarkdownReference, ...],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if not text.strip():
        return []
    if _token_count(text) <= chunk_size:
        return [text]

    protected, atoms = _protect_inline_atoms(text, references)

    def protected_token_count(value: str) -> int:
        return _token_count(_restore_atoms(value, atoms))

    splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=protected_token_count,
        keep_separator="end",
    )
    return [_restore_atoms(part, atoms) for part in splitter.split_text(protected)]


def _protect_inline_atoms(
    text: str, references: tuple[MarkdownReference, ...]
) -> tuple[str, dict[str, str]]:
    protected = text
    atoms: dict[str, str] = {}
    for reference in references:
        marker = chr(0xF0000 + len(atoms))
        atoms[marker] = reference.placeholder
        protected = protected.replace(reference.placeholder, marker)

    def protect_code(match: re.Match[str]) -> str:
        marker = chr(0xF0000 + len(atoms))
        atoms[marker] = match.group(0)
        return marker

    protected = _CODE_SPAN.sub(protect_code, protected)
    return protected, atoms


def _restore_atoms(text: str, atoms: dict[str, str]) -> str:
    restored = text
    for marker, value in atoms.items():
        restored = restored.replace(marker, value)
    return restored


def _take_short_leading_body(bodies: list[_ChunkBody], chunk_size: int) -> _ChunkBody:
    if bodies and _token_count(bodies[-1].content) <= chunk_size // 2:
        return bodies.pop()
    return _ChunkBody("")


def _split_prefixed_lines(
    text: str,
    references: tuple[MarkdownReference, ...],
    prefix_pattern: re.Pattern[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split long list/quote lines and repeat their structural prefix."""
    parts: list[str] = []
    for line in text.splitlines():
        match = prefix_pattern.match(line)
        if match is None or _token_count(line) <= chunk_size:
            parts.append(line)
            continue

        prefix, body = match.groups()
        body_budget = max(1, chunk_size - _token_count(prefix))
        body_overlap = min(chunk_overlap, body_budget - 1)
        line_references = _references_in_content(line, references)
        body_parts = _split_prose(body, line_references, body_budget, body_overlap)
        parts.extend(f"{prefix}{part}" for part in body_parts)
    return parts


def _append_structured_parts(
    bodies: list[_ChunkBody],
    parts: list[str],
    references: tuple[MarkdownReference, ...],
    chunk_size: int,
) -> None:
    """Pack complete list/quote continuations without losing their prefixes."""
    for part in parts:
        part_references = _references_in_content(part, references)
        if bodies:
            combined = f"{bodies[-1].content}\n\n{part}"
            if _token_count(combined) <= chunk_size:
                combined_references = (*bodies[-1].references, *part_references)
                bodies[-1] = _ChunkBody(
                    combined,
                    _references_in_content(combined, combined_references),
                )
                continue
        bodies.append(_ChunkBody(part, part_references))


def _references_in_content(
    content: str, references: tuple[MarkdownReference, ...]
) -> tuple[MarkdownReference, ...]:
    present = [reference for reference in references if reference.placeholder in content]
    return tuple(sorted(present, key=lambda reference: content.index(reference.placeholder)))


def _document_chunk(
    section: Section,
    body: _ChunkBody,
    index: int,
    chunk_size: int,
) -> DocumentChunk:
    headers = "\n".join(
        f"{'#' * level} {title}"
        for level, title in zip(section.heading_levels, section.header_path, strict=True)
    )
    content = f"{headers}\n\n{body.content}" if headers else body.content
    references = _references_in_content(content, (*section.references, *body.references))
    search_text = content
    for reference in references:
        search_text = search_text.replace(reference.placeholder, reference.label)

    metadata: dict[str, object] = {
        "header_path": list(section.header_path),
        "h1": _header_at_level(section, 1),
        "h2": _header_at_level(section, 2),
        "h3": _header_at_level(section, 3),
    }
    if references:
        metadata["references"] = [reference.to_metadata() for reference in references]
    if _token_count(body.content) > chunk_size:
        metadata["oversized_block"] = True
    return DocumentChunk(
        content=content,
        search_text=search_text,
        token_count=_token_count(body.content),
        index=index,
        metadata=metadata,
    )


def _validate_config(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be at least 0 and less than chunk_size")


def _token_count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _header_at_level(section: Section, level: int) -> str | None:
    for heading_level, title in zip(section.heading_levels, section.header_path, strict=True):
        if heading_level == level:
            return title
    return None
