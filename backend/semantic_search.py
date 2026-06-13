"""Semantic search over the vector store.

Embeds the query and retrieves the most similar code chunks by cosine
similarity. `top_k` controls how many results are returned (tunable for
relevance vs. latency).
"""
from __future__ import annotations

from .config import Settings
from .embeddings import Embedder
from .schemas import RetrievedChunk


class SemanticSearcher:
    def __init__(self, settings: Settings, embedder: Embedder, store):
        self.settings = settings
        self.embedder = embedder
        self.store = store

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.top_k
        query_vec = self.embedder.embed_one(query)
        hits = self.store.search(query_vec, top_k=top_k)
        return [
            RetrievedChunk(chunk=chunk, score=round(float(score), 4))
            for chunk, score in hits
        ]
