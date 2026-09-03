"""Chinese prompts and model-message builders for the RAG workflow."""

from __future__ import annotations

from collections.abc import Sequence
import json
from xml.sax.saxutils import escape

from graph_rag_demo.models.chat import ChatMessage
from graph_rag_demo.models.retrieval import SearchResult


EXPANSION_SYSTEM_PROMPT = """你是一个检索问题改写助手。
请根据当前问题和已有对话，生成有助于知识库检索的补充问题。
不要回答问题，不要解释原因。
只返回 JSON 对象，格式为：{"queries": ["补充问题1", "补充问题2"]}。"""

ANSWER_SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。
请仅依据用户问题和 <knowledge> 中提供的知识回答。
如果知识不足以支持结论，请明确说明信息不足，不要自行编造。
回答必须简洁、准确，并保留必要的条件和限制。
只返回 JSON 对象，格式为：{"answer": "回答内容", "used_chunk_ids": [1, 2]}。"""


def build_expansion_messages(
    question: str,
    chat_context: Sequence[ChatMessage],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
        *(_message_to_dict(message) for message in chat_context),
        {"role": "user", "content": question},
    ]


def build_answer_messages(
    question: str,
    chat_context: Sequence[ChatMessage],
    evidence: Sequence[SearchResult],
) -> list[dict[str, str]]:
    knowledge_xml = build_knowledge_xml(evidence)
    user_content = f"当前问题：\n{question}\n\n参考知识：\n{knowledge_xml}"
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        *(_message_to_dict(message) for message in chat_context),
        {"role": "user", "content": user_content},
    ]


def build_knowledge_xml(results: Sequence[SearchResult]) -> str:
    if not results:
        return "<knowledge />"

    chunks = [_knowledge_chunk_xml(result) for result in results]
    return "<knowledge>\n" + "\n".join(chunks) + "\n</knowledge>"


def _knowledge_chunk_xml(result: SearchResult) -> str:
    references = result.metadata.get("chunk", {})
    if isinstance(references, dict):
        references = references.get("references", [])
    if not isinstance(references, list) or not references:
        return f'  <chunk id="{result.chunk_id}">{escape(result.content)}</chunk>'

    references_json = escape(json.dumps(references, ensure_ascii=False))
    return (
        f'  <chunk id="{result.chunk_id}">\n'
        f"    <content>{escape(result.content)}</content>\n"
        f"    <references>{references_json}</references>\n"
        "  </chunk>"
    )


def _message_to_dict(message: ChatMessage) -> dict[str, str]:
    return {"role": message.role, "content": message.content}
