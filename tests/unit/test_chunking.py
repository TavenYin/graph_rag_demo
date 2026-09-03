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


def test_split_text_extracts_image_and_link_references_from_content():
    chunks = split_text(
        "请见 [文档](https://example.com/doc \"说明\") 和 ![示意图](https://example.com/a.png \"截图\")。",
        chunk_size=200,
        chunk_overlap=0,
    )

    assert chunks[0].content == "请见 @@LINK:link_1@@ 和 @@IMG:img_1@@。"
    assert chunks[0].search_text == "请见 文档 和 示意图。"
    assert chunks[0].metadata["references"] == [
        {
            "key": "link_1",
            "type": "link",
            "url": "https://example.com/doc",
            "text": "文档",
            "title": "说明",
        },
        {
            "key": "img_1",
            "type": "image",
            "url": "https://example.com/a.png",
            "alt": "示意图",
            "title": "截图",
        },
    ]


def test_split_text_assigns_a_distinct_reference_key_to_each_occurrence():
    chunks = split_text(
        "[第一处](https://example.com/doc) 和 [第二处](https://example.com/doc)",
        chunk_size=200,
        chunk_overlap=0,
    )

    assert chunks[0].content == "@@LINK:link_1@@ 和 @@LINK:link_2@@"
    assert [item["key"] for item in chunks[0].metadata["references"]] == ["link_1", "link_2"]


def test_split_text_keeps_only_each_chunks_references_in_metadata():
    chunks = split_text(
        "# 链接\n\n[文档](https://example.com/doc)\n\n# 图片\n\n![示意图](https://example.com/a.png)",
        chunk_size=200,
        chunk_overlap=0,
    )

    assert chunks[0].content == "# 链接\n\n@@LINK:link_1@@"
    assert [item["key"] for item in chunks[0].metadata["references"]] == ["link_1"]
    assert chunks[1].content == "# 图片\n\n@@IMG:img_1@@"
    assert [item["key"] for item in chunks[1].metadata["references"]] == ["img_1"]


def test_split_text_extracts_references_from_repeated_heading_prefixes():
    chunks = split_text(
        "# [文档](https://example.com/doc)\n\n第一段。\n\n第二段。",
        chunk_size=5,
        chunk_overlap=0,
    )

    assert len(chunks) == 2
    assert all(chunk.content.startswith("# @@LINK:link_1@@\n\n") for chunk in chunks)
    assert all(chunk.search_text.startswith("# 文档\n\n") for chunk in chunks)
    assert all(
        [item["key"] for item in chunk.metadata["references"]] == ["link_1"]
        for chunk in chunks
    )


def test_split_text_keeps_a_reference_atomic_within_the_chunk_budget():
    long_label = "退款说明" * 20
    text = f"前段。 [{long_label}](https://help.example.com/refund) 后段。"

    chunks = split_text(text, chunk_size=12, chunk_overlap=0)

    reference_chunks = [chunk for chunk in chunks if "@@LINK:link_1@@" in chunk.content]
    assert len(reference_chunks) == 1
    assert reference_chunks[0].token_count <= 12
    assert "oversized_block" not in reference_chunks[0].metadata
    assert reference_chunks[0].metadata["references"][0]["key"] == "link_1"


def test_split_text_merges_paragraph_list_and_quote_before_recursive_split():
    text = """退款需要三步。

- 打开订单
- 点退款

> 注意：到账要 3 天。
"""

    chunks = split_text(text, chunk_size=100, chunk_overlap=0)

    assert len(chunks) == 1
    assert "退款需要三步。" in chunks[0].content
    assert "- 打开订单" in chunks[0].content
    assert "> 注意：到账要 3 天。" in chunks[0].content


def test_split_text_removes_emphasis_markers_before_splitting_long_text():
    text = "开头 **" + ("重要内容" * 12) + "** 和 *备注内容* 结尾"

    chunks = split_text(text, chunk_size=12, chunk_overlap=0)

    combined = "".join(chunk.content.replace(" ", "") for chunk in chunks)
    assert "*" not in combined
    assert combined == ("开头" + ("重要内容" * 12) + "和备注内容结尾")


def test_split_text_keeps_inline_code_atomic_when_it_exceeds_the_budget():
    code = "`" + ("some_long_identifier_" * 8) + "`"

    chunks = split_text(f"开头 {code} 结尾", chunk_size=12, chunk_overlap=0)

    code_chunks = [chunk for chunk in chunks if code in chunk.content]
    assert len(code_chunks) == 1
    assert code_chunks[0].content == code
    assert code_chunks[0].metadata["oversized_block"] is True


def test_split_text_repeats_list_marker_on_every_long_item_continuation():
    chunks = split_text("- " + ("很长的列表项。" * 20), chunk_size=12, chunk_overlap=0)

    assert len(chunks) > 1
    assert all(chunk.content.startswith("- ") for chunk in chunks)


def test_split_text_repeats_quote_marker_on_every_continuation():
    chunks = split_text("> " + ("很长的引用内容。" * 20), chunk_size=12, chunk_overlap=0)

    assert len(chunks) > 1
    assert all(chunk.content.startswith("> ") for chunk in chunks)


def test_split_text_keeps_code_block_atomic_and_preserves_markdown():
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


def test_split_text_repeats_table_header_on_each_row_slice():
    rows = "\n".join(f"| f{index} | 说明{index} |" for index in range(8))
    text = f"| 字段 | 说明 |\n| --- | --- |\n{rows}"

    chunks = split_text(text, chunk_size=24, chunk_overlap=0)

    body_rows = [
        line for chunk in chunks for line in chunk.content.splitlines() if line.startswith("| f")
    ]

    assert len(chunks) > 1
    assert all("| 字段 | 说明 |" in chunk.content for chunk in chunks)
    assert all("| --- | --- |" in chunk.content for chunk in chunks)
    assert body_rows == [f"| f{index} | 说明{index} |" for index in range(8)]


def test_split_text_attaches_short_leading_prose_to_table_slices():
    rows = "\n".join(f"| f{index} | 说明{index} |" for index in range(8))
    text = f"字段说明如下。\n\n| 字段 | 说明 |\n| --- | --- |\n{rows}"

    chunks = split_text(text, chunk_size=24, chunk_overlap=0)

    assert len(chunks) > 1
    assert all(chunk.content.startswith("字段说明如下。") for chunk in chunks)
    assert all("| 字段 | 说明 |" in chunk.content for chunk in chunks)


def test_split_text_does_not_copy_an_oversized_table_row_into_overlap():
    oversized_cell = "超长单元格" * 20
    text = f"| 列 |\n| --- |\n| {oversized_cell} |\n| 短行1 |\n| 短行2 |"

    chunks = split_text(text, chunk_size=20, chunk_overlap=5)

    oversized_chunks = [chunk for chunk in chunks if oversized_cell in chunk.content]
    assert len(oversized_chunks) == 1
    assert oversized_chunks[0].content == f"| 列 |\n| --- |\n| {oversized_cell} |"
    assert oversized_chunks[0].metadata["oversized_block"] is True


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


def test_split_text_validates_chunk_configuration_for_empty_input():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        split_text("", chunk_size=0, chunk_overlap=0)


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
