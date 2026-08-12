from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, StrictInt


class QueryExpansionPayload(BaseModel):
    """Validated JSON returned by the retrieval-query expansion prompt."""

    model_config = ConfigDict(extra="forbid", strict=True)

    queries: list[str]


class AnswerPayload(BaseModel):
    """Validated JSON returned by the grounded-answer prompt."""

    model_config = ConfigDict(extra="forbid", strict=True)

    answer: str
    used_chunk_ids: list[StrictInt]


@dataclass(frozen=True)
class AskResult:
    answer: str
    used_chunk_ids: list[int]
