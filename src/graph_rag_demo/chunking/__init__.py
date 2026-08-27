"""Structure-aware Markdown chunking."""

from graph_rag_demo.chunking.block_chunker import chunk_blocks
from graph_rag_demo.chunking.markdown_parser import parse_markdown
from graph_rag_demo.chunking.models import DocumentChunk
from graph_rag_demo.chunking.section_builder import build_sections


TokenChunk = DocumentChunk


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]:
    """Split Markdown into heading-enriched, token-bounded document chunks."""
    if not text:
        return []
    return chunk_blocks(build_sections(parse_markdown(text)), chunk_size, chunk_overlap)


__all__ = ["DocumentChunk", "TokenChunk", "split_text"]
