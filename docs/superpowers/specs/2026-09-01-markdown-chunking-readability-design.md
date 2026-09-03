# Markdown Chunking Readability Design

## Goal

Keep the Markdown chunking flow easy to read while preserving semantic structure, extracting links and images into chunk metadata, and enforcing predictable token boundaries.

## Public contract

- `split_text(text, chunk_size, chunk_overlap)` remains the public entry point.
- Persisted chunk content contains `@@LINK:link_n@@` and `@@IMG:img_n@@` placeholders instead of URLs.
- Each occurrence receives its own document-local key, even when URLs repeat.
- `metadata["references"]` contains only references used by that final chunk, in appearance order.
- `search_text` replaces link placeholders with labels and image placeholders with alt text for embedding and full-text indexing; URLs are excluded.
- Heading prefixes remain in persisted content while `token_count` continues to count the body only.
- Only indivisible code blocks, inline code spans, and table rows may exceed `chunk_size`; oversized content must be isolated and marked.

## Readable module flow

```text
chunking/__init__.py
  -> markdown.py: Mistune AST -> typed Section/MarkdownBlock values
  -> chunker.py: Section blocks -> final DocumentChunk values
  -> table.py: row-aware Markdown table slicing
  -> models.py: shared data contracts
```

The implementation uses functions and small dataclasses only. It does not add strategy classes, factories, registries, dependency injection, or a second general-purpose AST model.

## Markdown behavior

- Paragraphs are recursively split on Chinese-aware boundaries.
- Strong and emphasis markers are removed while their text is retained.
- Inline code is indivisible; an oversized code span is emitted alone.
- Lists split first at item boundaries. An oversized item repeats its list marker on every continuation.
- Quotes repeat `> ` on every continuation.
- Fenced code blocks remain indivisible.
- Tables split by complete rows, repeat the header, and overlap only rows that fit both overlap and chunk budgets.
- Unsupported AST block nodes are ignored for now.

## Normalization

All input uses one Markdown-safe normalization path: Unicode NFC, normalized line endings, removal of unsafe control and zero-width characters, and outer trimming. Per-line trimming and the partial Markdown-detection regular expression are removed so indentation remains meaningful.

## Metadata and downstream use

References are stored under the existing JSONB chunk metadata, so no migration is required. Embedding and FTS use `DocumentChunk.search_text`. Retrieval carries chunk metadata so callers can resolve placeholders without reparsing Markdown.

## Verification

Tests must cover reference extraction and association, repeated URLs, overlap, long atomic references, semantic search text, list and quote continuations, oversized table rows, Markdown-safe normalization, parameter validation, and the existing heading/table/code contracts.
