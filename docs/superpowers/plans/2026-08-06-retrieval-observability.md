# Retrieval Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended; do not commit without human approval). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make retrieval scores and the key RAG stages observable through structured logs without changing retrieval or answer behavior.

**Architecture:** Add module loggers to RAG and retrieval services. Log stage summaries at INFO and per-result scores at DEBUG. Keep document content, prompts, credentials, and model answers out of logs. Vector retrieval uses configurable cosine distance filtering with a default maximum distance of 0.4.

**Tech Stack:** Python logging, pytest, PostgreSQL/pgvector integration tests.

## Global Constraints

- Preserve all existing retrieval, RRF, expansion, and answer behavior.
- Log identifiers, counts, ranks, and numeric scores; do not log full document content.
- Do not add a logging framework or telemetry dependency.
- Do not commit, push, or create a pull request.

---

### Task 1: Add retrieval score and RRF log contracts

**Files:**
- Modify: `tests/unit/test_retrieval.py`
- Modify: `src/graph_rag_demo/services/retrieval.py`

- [x] Add caplog tests for vector distance, full-text score, and RRF final scores.
- [x] Run the focused tests and observe failures because retrieval currently discards database scores and has no logs.
- [x] Select score columns in both SQL queries, preserve them in log-only processing, and log per-query retrieval and final RRF summaries.
- [x] Filter vector rows with configurable cosine distance, defaulting to a maximum distance of 0.4.

### Task 2: Add expansion and RAG stage logs

**Files:**
- Modify: `tests/unit/test_rag_service.py`
- Modify: `src/graph_rag_demo/services/rag.py`

- [x] Add caplog tests for expansion start/result/fallback and retrieval completion.
- [x] Run focused tests and observe failures.
- [x] Add INFO stage logs with counts and DEBUG logs for expanded queries; keep prompts, evidence text, and answer text out of logs.

### Task 3: Verify and document

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-08-06-retrieval-observability.md`

- [x] Document the observable log fields and levels.
- [x] Search for missing stage logging.
- [x] Run unit tests, local PostgreSQL integration tests, and source compilation.
