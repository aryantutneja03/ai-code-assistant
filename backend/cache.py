"""Semantic cache: caches answers keyed by query embedding similarity.

If a new question is within `semantic_cache_threshold` cosine similarity of a
previously answered question, the cached answer is returned — saving an LLM call
and tokens on repeated/similar queries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class _Entry:
    vector: list[float]
    question: str
    answer: str
    model_used: str
    provider: str


class SemanticCache:
    def __init__(self, threshold: float = 0.95, max_size: int = 512):
        self.threshold = threshold
        self.max_size = max_size
        self._entries: list[_Entry] = []
        self.hits = 0
        self.misses = 0

    def lookup(self, query_vec: list[float]) -> _Entry | None:
        best: _Entry | None = None
        best_sim = 0.0
        for entry in self._entries:
            sim = _cosine(query_vec, entry.vector)
            if sim > best_sim:
                best_sim, best = sim, entry
        if best is not None and best_sim >= self.threshold:
            self.hits += 1
            return best
        self.misses += 1
        return None

    def store(
        self,
        query_vec: list[float],
        question: str,
        answer: str,
        model_used: str,
        provider: str,
    ) -> None:
        self._entries.append(
            _Entry(query_vec, question, answer, model_used, provider)
        )
        if len(self._entries) > self.max_size:
            self._entries.pop(0)
