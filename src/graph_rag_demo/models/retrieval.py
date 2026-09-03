from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SearchMatch:
    query_index: int
    query: str
    channel: Literal["vector", "fulltext"]
    rank: int


@dataclass(frozen=True)
class RankedChunk:
    chunk_id: int
    content: str
    match: SearchMatch
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    chunk_id: int
    content: str
    score: float
    matches: tuple[SearchMatch, ...]
    metadata: dict[str, object] = field(default_factory=dict)
