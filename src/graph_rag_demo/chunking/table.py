"""Split rendered Markdown tables without breaking or duplicating rows."""

from __future__ import annotations

from collections.abc import Callable


def split_table(
    table_markdown: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    length_function: Callable[[str], int],
    pre_text: str = "",
) -> list[str]:
    """Repeat table headers and keep every body row as one indivisible unit."""
    header_rows, content_rows = _parse_table_markdown(table_markdown)
    header = "\n".join(header_rows).strip()
    if not content_rows:
        return [_join_table_chunk(pre_text, header, [])]

    full = _join_table_chunk(pre_text, header, content_rows)
    if length_function(full) <= chunk_size:
        return [full]
    return _split_rows(
        pre_text,
        header,
        content_rows,
        chunk_size,
        chunk_overlap,
        length_function,
    )


def _parse_table_markdown(table_markdown: str) -> tuple[list[str], list[str]]:
    rows = [row for row in table_markdown.strip().splitlines() if row]
    if len(rows) >= 2:
        return rows[:2], rows[2:]
    return rows, []


def _split_rows(
    pre_text: str,
    header: str,
    rows: list[str],
    chunk_size: int,
    chunk_overlap: int,
    length_function: Callable[[str], int],
) -> list[str]:
    header_length = length_function(_join_table_chunk(pre_text, header, []))
    row_lengths = [length_function(row) + 1 for row in rows]
    max_overlap = min(chunk_overlap, chunk_size // 2)
    chunks: list[str] = []
    current: list[tuple[str, int]] = []
    current_length = header_length

    for row, row_length in zip(rows, row_lengths, strict=True):
        # A row that cannot fit with the repeated header is an atomic oversized chunk.
        if header_length + row_length > chunk_size:
            if current:
                chunks.append(_join_table_chunk(pre_text, header, [item[0] for item in current]))
                current = []
                current_length = header_length
            chunks.append(_join_table_chunk(pre_text, header, [row]))
            continue

        if current and current_length + row_length > chunk_size:
            chunks.append(_join_table_chunk(pre_text, header, [item[0] for item in current]))
            current = _rows_that_fit_overlap(
                current,
                max_overlap=max_overlap,
                available_length=chunk_size - header_length - row_length,
            )
            current_length = header_length + sum(item[1] for item in current)

        current.append((row, row_length))
        current_length += row_length

    if current:
        chunks.append(_join_table_chunk(pre_text, header, [item[0] for item in current]))
    return chunks


def _rows_that_fit_overlap(
    rows: list[tuple[str, int]],
    *,
    max_overlap: int,
    available_length: int,
) -> list[tuple[str, int]]:
    allowed_length = min(max_overlap, max(0, available_length))
    overlap: list[tuple[str, int]] = []
    overlap_length = 0
    for row in reversed(rows):
        if overlap_length + row[1] > allowed_length:
            break
        overlap.insert(0, row)
        overlap_length += row[1]
    return overlap


def _join_table_chunk(pre_text: str, header: str, rows: list[str]) -> str:
    table = "\n".join([header, *rows]) if rows else header
    return f"{pre_text}\n\n{table}" if pre_text else table
