# Lightweight Layout Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Organize the Graph RAG Demo into small API, client, model, and service packages without changing its public HTTP behavior.

**Architecture:** `api.app` owns FastAPI creation and lifespan wiring; `api.routes` owns route registration and handlers; `models.api` owns API schemas and lifecycle dependency data. Clients, services, and shared data models each receive a focused package, while configuration, database access, token chunking, text cleaning, and FTS tokenization remain simple root modules.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, SQLAlchemy async, pytest.

## Global Constraints

- Keep the project a small Graph RAG Demo; do not add a DI framework, repository layer, or new runtime behavior.
- `create_app` remains importable as `graph_rag_demo.api.create_app`.
- API schemas and `ApplicationServices` live outside route handlers.
- API construction and route registration are separate modules.
- Do not commit, push, or create a PR.

---

### Task 1: Add import-contract tests for the new layout

**Files:**
- Modify: `tests/unit/test_api.py`, `tests/unit/test_clients.py`, `tests/unit/test_chunking.py`, `tests/unit/test_rag_service.py`, `tests/unit/test_retrieval.py`

**Interfaces:**
- Requires `graph_rag_demo.api.app.create_app`, `graph_rag_demo.api.routes.register_routes`, and models under `graph_rag_demo.models` submodules.

- [x] Write import assertions and update test imports to the desired package locations.
- [x] Run `uv run --group dev python -m pytest tests/unit -v`; expect collection errors because the destination packages do not exist.

### Task 2: Move model, client, and service modules

**Files:**
- Create: `src/graph_rag_demo/models/__init__.py`, `models/api.py`, `models/chunk.py`, `models/generation.py`, `models/retrieval.py`, `clients/__init__.py`, `clients/embedding.py`, `clients/llm.py`, `services/__init__.py`, `services/knowledge.py`, `services/retrieval.py`, `services/rag.py`, `text.py`
- Remove after moves: root `models.py`, `embedding_client.py`, `llm_client.py`, `knowledge.py`, `retrieval.py`, `rag_service.py`

- [x] Move each class without changing its behavior; move `clean_text` to `text.py` so API schemas do not depend on a service module.
- [x] Update all production imports and keep `models/__init__.py` as a small re-export surface.
- [x] Run the complete unit suite; expect all behavior tests to pass.

### Task 3: Split FastAPI app wiring from route handlers

**Files:**
- Create: `src/graph_rag_demo/api/__init__.py`, `api/app.py`, `api/routes.py`
- Remove after move: root `api.py`
- Modify: `scripts/run_server.py`, `README.md`, `docs/superpowers/specs/2026-08-04-graph-rag-demo-foundation-design.md`

**Interfaces:**
- `api.app.create_app(services=None) -> FastAPI` constructs the app and lifespan.
- `api.routes.register_routes(app) -> None` registers exactly `/health`, `/documents`, and `/ask`.

- [x] Write or update the API layout test so `create_app` and `register_routes` are independently importable while public route behavior remains unchanged.
- [x] Move lifecycle/client construction into `app.py`; move request handling and error mapping into `routes.py`; put Pydantic models in `models/api.py`.
- [x] Update documented source paths, run all unit and integration tests, compile sources, and check the diff.
