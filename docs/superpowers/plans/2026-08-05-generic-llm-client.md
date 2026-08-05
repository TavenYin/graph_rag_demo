# Generic LLM Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate business prompts from the model transport client and use the OpenAI-compatible Python SDK for chat requests.

**Architecture:** `LLMClient` exposes one generic `complete(messages, json_mode)` method. `RAGService` builds the expansion and answer messages and validates their business-specific JSON payloads. The SDK remains behind `LLMClient` so the rest of the application does not depend on SDK response objects.

**Tech Stack:** Python 3.11+, OpenAI-compatible Chat Completions, httpx test transport, pytest.

## Global Constraints

- Preserve the existing RAG behavior and public API.
- Keep local deterministic tests; do not call a remote model during tests.
- Do not add prompt templates or abstractions beyond the two current RAG operations.
- Do not commit, push, or create a pull request.

---

### Task 1: Define the generic client contract

**Files:**
- Modify: `tests/unit/test_clients.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- `LLMClient.complete(messages: list[ChatCompletionMessageParam], json_mode: bool = False) -> str`
- `LLMClient.aclose() -> None`

- [x] Replace business-specific client tests with a test that sends arbitrary system and user messages and verifies the raw assistant text.
- [x] Run the focused test and observe the expected failure because `complete` does not exist.
- [x] Add the OpenAI Python SDK dependency and implement the generic client using `AsyncOpenAI`, preserving transport injection for tests.
- [x] Run the focused client tests.

### Task 2: Move prompt construction and payload parsing into RAG

**Files:**
- Modify: `tests/unit/test_rag_service.py`
- Modify: `src/graph_rag_demo/services/rag.py`

**Interfaces:**
- RAG expansion builds its own system/user messages and parses a `queries` JSON array.
- RAG answer builds its own system/user messages and parses `answer` plus `used_chunk_ids`.

- [x] Update the Fake LLM to implement only `complete`.
- [x] Run the RAG tests and observe the expected failure against the old `expand/answer` calls.
- [x] Implement the two business-specific message builders and JSON validation in `RAGService`.
- [x] Run all unit tests and the local integration suite.

### Task 3: Verify wiring and documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-08-05-generic-llm-client.md`

- [x] Document that prompts belong to RAG orchestration and the client only performs generic completion.
- [x] Search for stale `_chat_json`, `LLMClient.expand`, and `LLMClient.answer` references.
- [x] Run the complete verification command and inspect its exit code and full summary.
