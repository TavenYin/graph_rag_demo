from dataclasses import dataclass
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


@dataclass(frozen=True)
class SearchResult:
    chunk_id: int
    content: str
    score: float
    matches: tuple[SearchMatch, ...]
