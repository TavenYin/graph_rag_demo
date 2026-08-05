"""Small async PostgreSQL boundary used by application services."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from graph_rag_demo.config import Settings


class Database:
    """Owns the async engine and exposes one transaction per service operation."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._session_factory = async_sessionmaker(engine, expire_on_commit=False)

    @classmethod
    def create(cls, settings: Settings) -> "Database":
        engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        return cls(engine)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._session_factory() as session:
            async with session.begin():
                yield session

    async def healthcheck(self) -> bool:
        async with self._engine.connect() as connection:
            return await connection.scalar(text("SELECT 1")) == 1

    async def close(self) -> None:
        await self._engine.dispose()
