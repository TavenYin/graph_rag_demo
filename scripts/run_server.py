"""Run the local API server after environment variables have been loaded."""

from __future__ import annotations

import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "graph_rag_demo.api:create_app",
        factory=True,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )
