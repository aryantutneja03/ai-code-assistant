"""Vector store: pgvector when `DATABASE_URL` is set, else an in-memory store.

Both backends expose the same interface: `upsert`, `search` (cosine), and
`count`. The in-memory store keeps everything in process so the demo runs with
no Postgres.
"""
from __future__ import annotations

import math

from .config import Settings
from .schemas import CodeChunk


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore:
    backend = "in-memory"

    def __init__(self) -> None:
        self._chunks: dict[str, CodeChunk] = {}
        self._vectors: dict[str, list[float]] = {}

    def upsert(self, chunks: list[CodeChunk], vectors: list[list[float]]) -> None:
        for chunk, vec in zip(chunks, vectors):
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = vec

    def search(self, query_vec: list[float], top_k: int) -> list[tuple[CodeChunk, float]]:
        scored = [
            (self._chunks[cid], _cosine(query_vec, vec))
            for cid, vec in self._vectors.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def all_chunks(self) -> list[CodeChunk]:
        return list(self._chunks.values())

    def count(self) -> int:
        return len(self._chunks)


class PgVectorStore:
    backend = "pgvector"

    def __init__(self, settings: Settings, dim: int) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self._conn = psycopg.connect(settings.database_url, autocommit=True)
        register_vector(self._conn)
        self._dim = dim
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS code_chunks (
                chunk_id   TEXT PRIMARY KEY,
                path       TEXT,
                symbol     TEXT,
                kind       TEXT,
                start_line INT,
                end_line   INT,
                content    TEXT,
                embedding  vector({dim})
            )
            """
        )

    def upsert(self, chunks: list[CodeChunk], vectors: list[list[float]]) -> None:
        with self._conn.cursor() as cur:
            for chunk, vec in zip(chunks, vectors):
                cur.execute(
                    """
                    INSERT INTO code_chunks
                      (chunk_id, path, symbol, kind, start_line, end_line, content, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (chunk_id) DO UPDATE
                      SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
                    """,
                    (
                        chunk.chunk_id,
                        chunk.path,
                        chunk.symbol,
                        chunk.kind,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.content,
                        vec,
                    ),
                )

    def search(self, query_vec: list[float], top_k: int) -> list[tuple[CodeChunk, float]]:
        rows = self._conn.execute(
            """
            SELECT chunk_id, path, symbol, kind, start_line, end_line, content,
                   1 - (embedding <=> %s::vector) AS score
            FROM code_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vec, query_vec, top_k),
        ).fetchall()
        results = []
        for r in rows:
            chunk = CodeChunk(
                chunk_id=r[0], path=r[1], symbol=r[2], kind=r[3],
                start_line=r[4], end_line=r[5], content=r[6],
            )
            results.append((chunk, float(r[7])))
        return results

    def all_chunks(self) -> list[CodeChunk]:
        rows = self._conn.execute(
            "SELECT chunk_id, path, symbol, kind, start_line, end_line, content FROM code_chunks"
        ).fetchall()
        return [
            CodeChunk(
                chunk_id=r[0], path=r[1], symbol=r[2], kind=r[3],
                start_line=r[4], end_line=r[5], content=r[6],
            )
            for r in rows
        ]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM code_chunks").fetchone()
        return int(row[0]) if row else 0


def build_vector_store(settings: Settings, dim: int):
    if settings.pgvector_enabled:
        try:
            return PgVectorStore(settings, dim)
        except Exception:  # pragma: no cover - fall back if Postgres unreachable
            pass
    return InMemoryVectorStore()
