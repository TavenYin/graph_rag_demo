"""Configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
from os import getenv


def _read_bool(name: str, default: bool) -> bool:
    value = getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _read_int(name: str, default: int) -> int:
    value = getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def _read_float(name: str, default: float) -> float:
    value = getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error


@dataclass(frozen=True)
class Settings:
    database_url: str
    use_real_clients: bool
    dashscope_api_key: str | None
    dashscope_base_url: str
    llm_model: str
    embedding_model: str
    embedding_dimensions: int
    chunk_size: int
    chunk_overlap: int
    vector_max_distance: float

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=getenv(
                "DATABASE_URL",
                "postgresql+asyncpg://graph_rag:graph_rag@localhost:5432/graph_rag_demo",
            ),
            use_real_clients=_read_bool("USE_REAL_CLIENTS", False),
            dashscope_api_key=getenv("DASHSCOPE_API_KEY") or None,
            dashscope_base_url=getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            llm_model=getenv("LLM_MODEL", "qwen-plus"),
            embedding_model=getenv("EMBEDDING_MODEL", "text-embedding-v4"),
            embedding_dimensions=_read_int("EMBEDDING_DIMENSIONS", 1024),
            chunk_size=_read_int("CHUNK_SIZE", 400),
            chunk_overlap=_read_int("CHUNK_OVERLAP", 80),
            vector_max_distance=_read_float("VECTOR_MAX_DISTANCE", 0.4),
        )

    def validate(self) -> None:
        if self.use_real_clients and not self.dashscope_api_key:
            raise ValueError("DASHSCOPE_API_KEY is required when USE_REAL_CLIENTS is true")
        if self.embedding_dimensions != 1024:
            raise ValueError("EMBEDDING_DIMENSIONS must be 1024 for text-embedding-v4")
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be at least 0 and less than CHUNK_SIZE")
        if not 0 < self.vector_max_distance <= 2:
            raise ValueError("VECTOR_MAX_DISTANCE must be greater than 0 and at most 2")
