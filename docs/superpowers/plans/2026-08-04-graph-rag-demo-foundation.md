# Graph RAG Demo Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, reproducible traditional RAG baseline that uses token chunking, query expansion, global weighted RRF, PostgreSQL + pgvector, and no intent classification.

**Architecture:** FastAPI delegates document ingestion to `KnowledgeService` and question answering to `RAGService`. RAG expansion produces one original plus up to three deduplicated variants; each contributes vector and FTS ranking lists, all fused once by weighted RRF. PostgreSQL persists only documents and chunks; clients are injected for deterministic tests.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy async + asyncpg, PostgreSQL pgvector, httpx, tiktoken, jieba, pytest, Docker Compose.

## Global Constraints

- Project root is `/Users/taven/WorkSpace/mine/graph-rag-demo`; use an independent Git repository.
- Never commit, push, or create a PR; the user performs final review.
- Store no real secret in source; use `.env.example` and ignored `.env`.
- Use fully local PostgreSQL + pgvector through Docker Compose.
- Keep the design small: no Neo4j, entity tables, intent recognition, business category, retries, DI framework, task queue, or repository layer.
- Use `tiktoken` as the project-wide deterministic token estimate.
- Use one global weighted RRF over every query/retriever ranking list; no intermediate RRF.

---

### Task 1: Project scaffold and typed configuration

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `src/graph_rag_demo/__init__.py`, `src/graph_rag_demo/config.py`, `tests/unit/test_config.py`

**Interfaces:**
- Produces `Settings.from_env()` and `Settings.validate()`.

- [ ] Write tests proving a missing API key is rejected only for real-client configuration and that `embedding_dimensions != 1024` is rejected.
- [ ] Run `pytest tests/unit/test_config.py -v`; expect collection failure because the package is absent.
- [ ] Add the minimal package metadata, dependencies and `Settings` dataclass with environment parsing and validation.
- [ ] Re-run the focused test and then `pytest tests/unit/test_config.py -v` with exit code 0.

### Task 2: Token chunking and shared result models

**Files:**
- Create: `src/graph_rag_demo/models.py`, `src/graph_rag_demo/chunking.py`, `tests/unit/test_chunking.py`

**Interfaces:**
- Produces `TokenChunk(content: str, token_count: int, index: int)` and `split_text(text, chunk_size, chunk_overlap)`.

- [ ] Write tests for exact token-size splitting, token overlap, empty input and storing actual token count.
- [ ] Run the focused test and observe failure because `chunking` is unavailable.
- [ ] Implement only the tiktoken-based splitter and immutable models needed by later services.
- [ ] Re-run the focused test with exit code 0.

### Task 3: Database schema and async database boundary

**Files:**
- Create: `compose.yaml`, `scripts/init_db.sql`, `src/graph_rag_demo/db.py`, `tests/integration/test_database.py`

**Interfaces:**
- Produces `Database.create(settings)`, `Database.session()` and `Database.healthcheck()`.

- [ ] Write integration tests for a database health query and transaction rollback after a forced exception.
- [ ] Run them with `RUN_INTEGRATION_TESTS=1`; expect skip or connection failure before Compose is started.
- [ ] Define only `kb_document` and `kb_chunk`, indexes, async engine/session lifecycle and the health/transaction boundary.
- [ ] Start the local service when Docker is available, run schema initialization and re-run integration tests.

### Task 4: DashScope clients and fake-compatible contracts

**Files:**
- Create: `src/graph_rag_demo/llm_client.py`, `src/graph_rag_demo/embedding_client.py`, `tests/unit/test_clients.py`

**Interfaces:**
- Produces async `expand(question, context) -> list[str]`, `answer(question, context) -> AnswerPayload`, and `embed(texts) -> list[list[float]]`.

- [ ] Write HTTP-transport tests for successful payload parsing and malformed JSON response handling.
- [ ] Run the focused test and observe import failure.
- [ ] Implement minimal async httpx clients, strict JSON parsing and injected transports; do not implement retry.
- [ ] Re-run focused tests with exit code 0.

### Task 5: Atomic document ingestion

**Files:**
- Create: `src/graph_rag_demo/knowledge.py`, `tests/unit/test_knowledge.py`, `tests/integration/test_knowledge.py`

**Interfaces:**
- Produces `KnowledgeService.upload(content, title, metadata) -> int` and `DuplicateDocumentError`.

- [ ] Write tests showing embedding failure writes no document, duplicate cleaned content is rejected, and successful upload persists token counts.
- [ ] Run focused unit tests and observe failure because service is missing.
- [ ] Implement clean text, SHA-256, token splitting, single batch embedding call and one database transaction for document plus chunks.
- [ ] Re-run unit tests; run integration test against Compose when available.

### Task 6: Vector/FTS retrieval and one global weighted RRF

**Files:**
- Create: `src/graph_rag_demo/tokenize_fts.py`, `src/graph_rag_demo/retrieval.py`, `tests/unit/test_retrieval.py`, `tests/integration/test_retrieval.py`

**Interfaces:**
- Produces `RetrievalService.search_all(queries, embeddings) -> list[SearchResult]` and `weighted_rrf(rank_lists, weights, top_n)`.

- [ ] Write tests proving multiple ranking lists are fused once, repeated chunk IDs accumulate, original-query weights dominate equal expansion matches, and no duplicate result survives.
- [ ] Run focused unit tests and observe failure.
- [ ] Implement parameterized pgvector/FTS queries, result provenance, and only one final weighted RRF call.
- [ ] Re-run unit tests; run integration retrieval test against Compose when available.

### Task 7: Query expansion and answer orchestration

**Files:**
- Create: `src/graph_rag_demo/rag_service.py`, `tests/unit/test_rag_service.py`

**Interfaces:**
- Produces `RAGService.ask(question, chat_context) -> AskResult`.

- [ ] Write tests proving original question stays first, expansion deduplicates and caps at three, expansion failure falls back, embeddings are batched once, and invalid model citations are filtered.
- [ ] Run focused tests and observe failure.
- [ ] Implement the minimal orchestration over injected clients/retrieval, token-budget context assembly and citation filtering.
- [ ] Re-run focused tests with exit code 0.

### Task 8: HTTP surface and learning documentation

**Files:**
- Create: `src/graph_rag_demo/api.py`, `scripts/run_server.py`, `README.md`, `tests/unit/test_api.py`
- Modify: `.env.example`, `pyproject.toml`

**Interfaces:**
- Exposes `GET /health`, `POST /documents`, `POST /ask`.

- [ ] Write API tests for health, duplicate document conflict and ask response citation contract using real route handlers with injected fake services.
- [ ] Run focused tests and observe failure.
- [ ] Implement the three endpoints, lifespan resource wiring, concise error mapping and README commands/architecture/next-step explanation.
- [ ] Re-run focused tests and run the complete non-integration suite with exit code 0.

### Task 9: Full verification and specification cross-check

**Files:**
- Modify: `README.md` only if verification exposes an incorrect command.

- [ ] Start Compose, run schema initialization, then execute all unit and integration tests with full logs preserved.
- [ ] Run `python -m compileall src` and verify every README command against the project files.
- [ ] Compare the result with `docs/superpowers/specs/2026-08-04-graph-rag-demo-foundation-design.md`; record any unavailable external prerequisite rather than claiming it passed.
- [ ] Do not commit; present the working tree for the user's unified review.
