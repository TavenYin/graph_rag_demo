import pytest
from pydantic import ValidationError

from graph_rag_demo.models.generation import AnswerPayload, QueryExpansionPayload


def test_query_expansion_payload_parses_model_json() -> None:
    payload = QueryExpansionPayload.model_validate_json('{"queries": ["实体检索"]}')

    assert payload.queries == ["实体检索"]


def test_answer_payload_rejects_non_integer_chunk_ids() -> None:
    with pytest.raises(ValidationError):
        AnswerPayload.model_validate_json(
            '{"answer": "答案", "used_chunk_ids": ["1"]}'
        )
