"""RAG orchestration: retrieve (semantic search) -> (cache) -> generate.

This single `RagEngine` ties together the embedder, vector store, semantic
searcher, a simple semantic cache, and the OpenAI client. Created once and
reused.
"""
from __future__ import annotations

from .cache import SemanticCache
from .chunking import chunk_directory
from .config import Settings, get_settings
from .embeddings import Embedder
from .llm_client import LLMClient
from .prompts import build_messages
from .schemas import AskResponse, RetrievedChunk
from .semantic_search import SemanticSearcher
from .vector_store import build_vector_store


class RagEngine:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.embedder = Embedder(self.settings)
        self.store = build_vector_store(self.settings, self.embedder.dim)
        self.searcher = SemanticSearcher(self.settings, self.embedder, self.store)
        self.cache = SemanticCache(threshold=self.settings.semantic_cache_threshold)
        self.llm = LLMClient(self.settings)

    # ---- indexing ------------------------------------------------------------
    def index_directory(self, root: str, include_ext: list[str]) -> tuple[int, int]:
        chunks = chunk_directory(root, include_ext)
        if not chunks:
            return 0, 0
        vectors = self.embedder.embed([c.content for c in chunks])
        self.store.upsert(chunks, vectors)
        files = len({c.path for c in chunks})
        return files, len(chunks)

    # ---- retrieval -----------------------------------------------------------
    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self.searcher.search(question, top_k=top_k)

    # ---- ask -----------------------------------------------------------------
    def ask(self, question: str, mode: str = "qa", top_k: int | None = None) -> AskResponse:
        query_vec = self.embedder.embed_one(question)
        cached = self.cache.lookup(query_vec)
        if cached is not None:
            return AskResponse(
                answer=cached.answer, model_used=cached.model_used,
                provider=cached.provider, cached=True, contexts=[],
            )
        contexts = self.retrieve(question, top_k=top_k)
        system, user = build_messages(question, contexts, mode)
        result = self.llm.generate(system, user, mode=mode)
        self.cache.store(query_vec, question, result.text, result.model_used, result.provider)
        return AskResponse(
            answer=result.text, model_used=result.model_used,
            provider=result.provider, cached=False, contexts=contexts,
        )

    def ask_stream(self, question: str, mode: str = "qa", top_k: int | None = None):
        contexts = self.retrieve(question, top_k=top_k)
        system, user = build_messages(question, contexts, mode)
        yield from self.llm.stream(system, user, mode=mode)
