"""Public interface for structure-aware Markdown chunking."""

from graph_rag_demo.chunking.chunker import split_text
from graph_rag_demo.chunking.models import DocumentChunk


TokenChunk = DocumentChunk
__all__ = ["DocumentChunk", "TokenChunk", "split_text"]
