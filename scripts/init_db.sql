CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS kb_document (
    id BIGSERIAL PRIMARY KEY,
    title TEXT,
    checksum CHAR(64) NOT NULL UNIQUE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS kb_chunk (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES kb_document(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL CHECK (token_count > 0),
    content_tsv TSVECTOR NOT NULL,
    embedding vector(1024) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_kb_chunk_document_id
    ON kb_chunk (document_id);

CREATE INDEX IF NOT EXISTS idx_kb_chunk_content_tsv
    ON kb_chunk USING GIN (content_tsv);

CREATE INDEX IF NOT EXISTS idx_kb_chunk_embedding
    ON kb_chunk USING hnsw (embedding vector_cosine_ops);
