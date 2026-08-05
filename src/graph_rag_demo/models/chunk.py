from dataclasses import dataclass


@dataclass(frozen=True)
class TokenChunk:
    content: str
    token_count: int
    index: int
