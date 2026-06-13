-- pgvector schema for the AI Code-Generation Assistant.
-- Run once: psql "$DATABASE_URL" -f sql/schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- text-embedding-3-small => 1536 dims. Adjust if you change the model.
CREATE TABLE IF NOT EXISTS code_chunks (
    chunk_id   TEXT PRIMARY KEY,
    path       TEXT,
    symbol     TEXT,
    kind       TEXT,
    start_line INT,
    end_line   INT,
    content    TEXT,
    embedding  vector(1536)
);

-- Approximate nearest-neighbour index (cosine).
CREATE INDEX IF NOT EXISTS code_chunks_embedding_idx
    ON code_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Helps BM25-side filtering / lookups by path.
CREATE INDEX IF NOT EXISTS code_chunks_path_idx ON code_chunks (path);
