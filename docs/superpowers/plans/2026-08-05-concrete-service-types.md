# Concrete Service Types Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make navigation from API and services reach the actual database, client, and service implementations.

**Architecture:** Replace dependency-boundary `Any` annotations with the existing concrete classes. Keep `Any` only for unconstrained JSON metadata and external JSON payloads; do not introduce protocols, a dependency-injection framework, or new layers.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy, pytest.

## Global Constraints

- Do not commit, push, or create a pull request.
- Keep the existing dependency injection used by tests.
- Do not change runtime behavior.

---

### Task 1: Verify concrete collaborator annotations

**Files:**
- Modify: `tests/unit/test_api.py`
- Modify: `src/graph_rag_demo/models/api.py`, `src/graph_rag_demo/services/knowledge.py`, `src/graph_rag_demo/services/retrieval.py`, `src/graph_rag_demo/services/rag.py`

**Interfaces:**
- `ApplicationServices.database: Database`
- `ApplicationServices.knowledge_service: KnowledgeService | None`
- `ApplicationServices.rag_service: RAGService | None`
- `KnowledgeService(database: Database, embedding_client: EmbeddingClient, ...)`
- `RetrievalService(database: Database, ...)`
- `RAGService(llm_client: LLMClient, embedding_client: EmbeddingClient, retrieval_service: RetrievalService, ...)`

- [x] Add a failing test using `typing.get_type_hints` to assert each dependency annotation resolves to its concrete implementation.
- [x] Run the focused test and confirm it fails because current annotations resolve to `Any`.
- [x] Replace only collaborator `Any` annotations with the listed concrete classes; type SQL mapping rows as `RowMapping`.
- [x] Run the focused test, then the complete unit and integration suites.
