"""Chinese tokenization shared by FTS document indexing and queries."""

from __future__ import annotations

import jieba


STOPWORDS = frozenset(
    {
        "的", "了", "在", "是", "和", "与", "及", "对", "等", "也", "都", "就", "还",
        "又", "更", "最", "很", "太", "非常", "比较", "已经", "一直", "这", "那", "它",
        "他", "她", "我", "你", "您", "我们", "你们", "他们", "它们", "这个", "那个",
        "这些", "那些", "这样", "那样", "而", "但", "或", "且", "因为", "所以", "如果",
        "虽然", "但是", "然而", "不过", "可是", "并且", "以及", "或者", "啊", "呢", "吗",
        "吧", "呀", "哦", "嗯", "哈", "啦", "一", "二", "三", "四", "五", "六", "七",
        "八", "九", "十", "百", "千", "万", "个", "只", "条", "件", "种", "类", "次",
        "回", "遍", "场", "些", "从", "向", "往", "到", "为", "被", "把", "让", "使",
        "由", "通过", "根据", "可以", "能够", "应该", "必须", "需要", "想要", "希望", "可能",
        "也许", "一般", "通常", "经常", "总是", "有时", "偶尔", "大概", "大约", "什么",
        "怎么", "如何", "为什么", "哪里", "哪个", "哪些", "多少",
    }
)


def tokenize_for_fts(text: str) -> str:
    """Return a space-separated sequence of meaningful terms for ``simple`` FTS."""
    if not text or not text.strip():
        return ""

    terms = (term.strip() for term in jieba.lcut(text.strip()))
    return " ".join(term for term in terms if len(term) >= 2 and term not in STOPWORDS)
