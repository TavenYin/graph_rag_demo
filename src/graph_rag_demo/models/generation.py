from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerPayload:
    answer: str
    used_chunk_ids: list[int]


@dataclass(frozen=True)
class AskResult:
    answer: str
    used_chunk_ids: list[int]
