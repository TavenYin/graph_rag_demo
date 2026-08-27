import tiktoken
import pytest

from graph_rag_demo.chunking import split_text
from graph_rag_demo.models.chunk import TokenChunk


def test_split_text_keeps_heading_path_in_content_and_chunk_metadata():
    chunks = split_text(
        "# 游戏\n## 第二章\n\n获得暗夜之矛。",
        chunk_size=100,
        chunk_overlap=0,
    )

    assert [chunk.content for chunk in chunks] == [
        "# 游戏\n## 第二章\n\n获得暗夜之矛。"
    ]
    assert chunks[0].token_count == len(tiktoken.get_encoding("cl100k_base").encode("获得暗夜之矛。"))
    assert chunks[0].metadata == {
        "header_path": ["游戏", "第二章"],
        "h1": "游戏",
        "h2": "第二章",
        "h3": None,
    }


def test_split_text_keeps_actual_heading_levels_after_a_jump():
    chunks = split_text("### Deep section\n\n正文", chunk_size=100, chunk_overlap=0)

    assert chunks[0].content == "### Deep section\n\n正文"
    assert chunks[0].metadata == {
        "header_path": ["Deep section"],
        "h1": None,
        "h2": None,
        "h3": "Deep section",
    }


def test_split_text_keeps_table_and_code_block_atomic_and_preserves_markdown():
    text = """# API

| 字段 | 说明 |
| --- | --- |
| id | 标识 |

```python
  print(\"keep indent\")
```
"""

    chunks = split_text(text, chunk_size=1, chunk_overlap=0)

    assert [chunk.content for chunk in chunks] == [
        "# API\n\n| 字段 | 说明 |\n| --- | --- |\n| id | 标识 |",
        '# API\n\n```python\n  print("keep indent")\n```',
    ]
    assert all(chunk.metadata["oversized_block"] for chunk in chunks)


def test_split_text_prefers_chinese_sentence_boundaries_within_token_budget():
    text = "第一句。第二句。第三句。第四句。"

    chunks = split_text(text, chunk_size=8, chunk_overlap=0)

    assert [chunk.index for chunk in chunks] == [0, 1, 2, 3]
    assert [chunk.content for chunk in chunks] == ["第一句。", "第二句。", "第三句。", "第四句。"]
    assert all(chunk.token_count <= 8 for chunk in chunks)
    assert all(isinstance(chunk, TokenChunk) for chunk in chunks)


def test_split_text_records_token_counts_under_the_requested_budget():
    encoding = tiktoken.get_encoding("cl100k_base")
    text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"

    chunks = split_text(text, chunk_size=10, chunk_overlap=2)

    assert [chunk.token_count for chunk in chunks] == [
        len(encoding.encode(chunk.content)) for chunk in chunks
    ]
    assert all(chunk.token_count <= 10 for chunk in chunks)


def test_split_text_returns_no_chunks_for_empty_input():
    assert split_text("", chunk_size=4, chunk_overlap=1) == []


def test_split_text_records_the_actual_token_count_for_each_chunk():
    encoding = tiktoken.get_encoding("cl100k_base")

    chunks = split_text("你好，Graph RAG！", chunk_size=100, chunk_overlap=0)

    assert len(chunks) == 1
    assert chunks[0].token_count == len(encoding.encode(chunks[0].content))


def test_split_text_keeps_chinese_and_emoji_lossless_at_chunk_boundaries():
    text = "你好😀世界🚀再见"

    chunks = split_text(text, chunk_size=2, chunk_overlap=0)

    assert "".join(chunk.content for chunk in chunks) == text
    assert all("\ufffd" not in chunk.content for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (4, -1), (4, 4)],
)
def test_split_text_rejects_invalid_chunk_parameters(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        split_text("text", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
