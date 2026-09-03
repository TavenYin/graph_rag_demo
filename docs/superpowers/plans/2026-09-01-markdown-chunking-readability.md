# Markdown Chunking Readability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Markdown chunking pipeline readable while extracting link/image references into metadata and fixing structural and token-boundary errors.

**Architecture:** Replace the parser-to-renderer-to-regex round trip with a readable four-module flow: `markdown.py` parses AST nodes into typed blocks, `chunker.py` applies block-specific rules, `table.py` owns row slicing, and `models.py` owns data contracts. Keep functions and small dataclasses; do not introduce extensibility frameworks.

**Tech Stack:** Python 3.12, Mistune 3, LangChain text splitters, tiktoken, pytest, SQLAlchemy/PostgreSQL JSONB.

**Spec:** `docs/superpowers/specs/2026-09-01-markdown-chunking-readability-design.md`

## Global Constraints

- Do not run `git commit`, `git push`, or create a pull request before human review.
- Preserve existing user changes and the public `split_text` entry point.
- Use TDD for every behavior change: focused test must fail for the expected reason before production edits.
- Do not run write-formatters or automatic fix commands.
- Keep unsupported AST block nodes ignored for now.
- Do not add a database migration.

---

### Task 1: Reference extraction and searchable text

**Files:**
- Modify: `src/graph_rag_demo/chunking/models.py`
- Create: `src/graph_rag_demo/chunking/markdown.py`
- Modify: `src/graph_rag_demo/chunking/chunker.py`
- Test: `tests/unit/test_chunking.py`
- Test: `tests/unit/test_knowledge.py`

**Interfaces:**
- Produces: `MarkdownReference`, `MarkdownBlock.references`, `DocumentChunk.search_text`, `parse_sections(markdown: str) -> list[Section]`.
- Preserves: `split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]`.

- [ ] Add failing tests proving links/images become placeholders, metadata contains only local references, repeated URLs receive distinct keys, and search text uses label/alt without URLs.
- [ ] Run the focused tests and confirm failures show that raw Markdown URLs remain in content or references/search text are absent.
- [ ] Add the minimal reference data contract and AST rendering needed to pass the tests.
- [ ] Run the focused tests and confirm they pass without changing unrelated behavior.

### Task 2: Structural Markdown boundaries

**Files:**
- Modify: `src/graph_rag_demo/chunking/markdown.py`
- Modify: `src/graph_rag_demo/chunking/chunker.py`
- Test: `tests/unit/test_chunking.py`

**Interfaces:**
- Consumes: `MarkdownBlock.kind` values `paragraph`, `list`, `quote`, `table`, and `code`.
- Produces: final chunks with readable list/quote continuations and isolated inline/block code.

- [ ] Add failing tests for long list items, long quotes, strong/emphasis text, inline code, and invalid parameters with empty input.
- [ ] Run the focused tests and confirm each fails for the intended current behavior.
- [ ] Implement paragraph, list, quote, and atomic-code splitting with no strategy classes.
- [ ] Run the focused tests and confirm the structural contracts pass.

### Task 3: Table budget and overlap correctness

**Files:**
- Rename: `src/graph_rag_demo/chunking/table_splitter.py` to `src/graph_rag_demo/chunking/table.py`
- Modify: `src/graph_rag_demo/chunking/chunker.py`
- Test: `tests/unit/test_chunking.py`

**Interfaces:**
- Produces: `split_table(...)` slices that repeat headers, isolate oversized rows, and overlap only rows that fit the configured limits.

- [ ] Add a failing test proving an oversized row appears once and is not copied into overlap.
- [ ] Run the test and confirm it fails because the current overlap fallback repeats the oversized row.
- [ ] Change row packing so oversized rows are standalone and overlap is selected against both the overlap limit and the next chunk budget.
- [ ] Run table and chunking tests and confirm they pass.

### Task 4: Markdown-safe normalization

**Files:**
- Modify: `src/graph_rag_demo/services/knowledge.py`
- Test: `tests/unit/test_knowledge.py`

**Interfaces:**
- Produces: `_normalize_markdown(content: str) -> str` that preserves indentation for every document.

- [ ] Add a failing test proving indented code remains indented and unsafe characters are still removed.
- [ ] Run it and confirm failure is caused by the current plain-text fallback stripping indentation.
- [ ] Replace the structure-detection branch with one safe normalization path.
- [ ] Run knowledge and chunking tests and confirm they pass.

### Task 5: Persistence and retrieval metadata plumbing

**Files:**
- Modify: `src/graph_rag_demo/services/knowledge.py`
- Modify: `src/graph_rag_demo/models/retrieval.py`
- Modify: `src/graph_rag_demo/services/retrieval.py`
- Modify: `src/graph_rag_demo/services/prompts.py`
- Test: `tests/unit/test_knowledge.py`
- Test: `tests/unit/test_retrieval.py`
- Test: `tests/unit/test_rag_service.py`

**Interfaces:**
- Consumes: `DocumentChunk.search_text` and persisted JSONB metadata.
- Produces: embeddings/FTS based on search text and `SearchResult.metadata` available when constructing answer evidence.

- [ ] Add failing tests proving embedding and FTS use search text and retrieval preserves chunk metadata.
- [ ] Run focused tests and confirm failures identify use of raw placeholder content or missing metadata.
- [ ] Update persistence, SQL selection, retrieval models, and knowledge XML with the smallest compatible changes.
- [ ] Run focused tests and confirm the end-to-end reference association is retained.

### Task 6: Remove obsolete modules and verify

**Files:**
- Modify: `src/graph_rag_demo/chunking/__init__.py`
- Delete: `src/graph_rag_demo/chunking/markdown_parser.py`
- Delete: `src/graph_rag_demo/chunking/section_builder.py`
- Delete: `src/graph_rag_demo/chunking/block_chunker.py`
- Delete: `src/graph_rag_demo/chunking/atomic_spans.py`

**Interfaces:**
- Preserves: imports of `DocumentChunk`, `TokenChunk`, and `split_text` from `graph_rag_demo.chunking`.

- [ ] Repoint the package exports to the readable module flow and verify no obsolete imports remain with `rtk rg`.
- [ ] Run the complete pytest suite to a full log, preserve the exit code, and search the complete log for failure/error summaries.
- [ ] Run read-only Ruff checks if configured and `rtk git diff --check`.
- [ ] Review `rtk git diff` to confirm only the approved chunking, normalization, indexing, retrieval, tests, and design documents changed.
