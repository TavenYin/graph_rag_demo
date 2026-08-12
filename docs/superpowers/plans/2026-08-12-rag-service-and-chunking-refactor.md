# RAG Service 与递归分块优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 提升 RAG 服务命名和模型输出校验的可读性，并使用中文递归 token 分块改善语义边界。

**Architecture:** `RAGService.answer` 保持现有编排职责，私有方法按检索查询构造和有证据回答生成命名。LLM JSON 使用 Pydantic 模型统一解析，业务去重和证据引用过滤仍由服务负责。仅使用 `langchain-text-splitters` 的递归文本分块器，以 tiktoken 计数和中文分隔符生成不超过 token 上限的 chunk。

**Tech Stack:** Python 3.11+, Pydantic v2, tiktoken, langchain-text-splitters, pytest。

## Global Constraints

- 不引入 `langchain`、Agent、Chain、Retriever 或模型客户端封装。
- `kb_chunk.token_count` 必须继续表示实际 `cl100k_base` token 数。
- 继续保证中文和 emoji 不产生 `U+FFFD`。
- 现有全局 RRF 与 API 契约不改变。

---

### Task 1: 使用 Pydantic 解析 LLM JSON

**Files:**
- Modify: `src/graph_rag_demo/models/generation.py`
- Modify: `src/graph_rag_demo/services/rag.py`
- Modify: `tests/unit/test_rag_service.py`

**Interfaces:**
- Produces: `QueryExpansionPayload.model_validate_json(content)`。
- Produces: `AnswerPayload.model_validate_json(content)`。
- Produces: `RAGService.answer(question, chat_context) -> AskResult`。

- [x] 写失败测试，覆盖无效 JSON、缺少字段和非整数 `used_chunk_ids`。
- [x] 运行失败测试，确认当前手工字段解析不能满足 Pydantic 模型接口。
- [x] 定义严格 Pydantic 输出模型；将 JSON 解析异常包装为 `ModelResponseError`，保留扩写去重和引用过滤。
- [x] 运行 RAG 服务单元测试。

### Task 2: 重命名 RAG 编排方法

**Files:**
- Modify: `src/graph_rag_demo/services/rag.py`
- Modify: `src/graph_rag_demo/api/routes.py`
- Modify: `tests/unit/test_api.py`
- Modify: `tests/unit/test_rag_service.py`

**Interfaces:**
- Produces: `RAGService.answer(question, chat_context)`。
- Produces: `_build_retrieval_queries(question, chat_context)`。
- Produces: `_generate_grounded_answer(question, chat_context, evidence)`。

- [x] 写失败测试，验证路由调用 `answer`。
- [x] 运行失败测试，确认旧的 `ask` 接口不再满足调用。
- [x] 重命名方法和调用点，不修改检索、RRF、提示词或响应行为。
- [x] 运行 API 与 RAG 服务单元测试。

### Task 3: 中文递归 token 分块

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `src/graph_rag_demo/chunking.py`
- Modify: `tests/unit/test_chunking.py`

**Interfaces:**
- Produces: `split_text(text, chunk_size, chunk_overlap) -> list[TokenChunk]`。
- Consumes: `RecursiveCharacterTextSplitter`，长度由 `cl100k_base` token 数决定。

- [x] 写失败测试，验证优先按中文句末标点分割、chunk token 上限、中文/emoji 无损。
- [x] 运行失败测试，确认固定 token 切分不满足语义边界。
- [x] 添加 `langchain-text-splitters`，使用中文分隔符和 tiktoken 长度函数实现递归切分；重新计算每个 chunk 的实际 token 数。
- [x] 运行分块、知识上传和完整测试套件。
