import pytest

from graph_rag_demo.config import Settings


def test_missing_api_key_is_allowed_when_real_clients_are_disabled(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("USE_REAL_CLIENTS", "false")

    Settings.from_env().validate()


def test_missing_api_key_is_rejected_when_real_clients_are_enabled(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("USE_REAL_CLIENTS", "true")

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        Settings.from_env().validate()


def test_non_1024_embedding_dimension_is_rejected(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")

    with pytest.raises(ValueError, match="1024"):
        Settings.from_env().validate()


def test_vector_min_similarity_defaults_to_point_six(monkeypatch):
    monkeypatch.delenv("VECTOR_MIN_SIMILARITY", raising=False)

    assert Settings.from_env().vector_min_similarity == pytest.approx(0.6)


def test_vector_min_similarity_must_be_within_cosine_similarity_range(monkeypatch):
    monkeypatch.setenv("VECTOR_MIN_SIMILARITY", "1.1")

    with pytest.raises(ValueError, match="VECTOR_MIN_SIMILARITY"):
        Settings.from_env().validate()
