"""Embeddings: OpenAI `text-embedding-3` when configured, else a deterministic
local hashing embedder so the pipeline runs offline.

The local embedder is a hashed bag-of-tokens projected into a fixed-dim unit
vector. It is obviously weaker than a real model, but it is deterministic,
dependency-free, and good enough to demonstrate semantic-ish retrieval offline.
"""
from __future__ import annotations

import hashlib
import math
import re

from .config import Settings

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|\d+")


class Embedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.name: str
        self._client = None
        self._gemini = None
        if settings.openai_enabled:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=settings.openai_api_key)
                self.name = f"openai:{settings.openai_embed_model}"
            except Exception:  # pragma: no cover - missing dep / bad key
                self._client = None
        if self._client is None and settings.gemini_enabled:
            try:
                from google import genai

                self._gemini = genai.Client(api_key=settings.gemini_api_key)
                self.name = f"gemini:{settings.gemini_embed_model}"
            except Exception:  # pragma: no cover - missing dep / bad key
                self._gemini = None
        if self._client is None and self._gemini is None:
            self.name = "local:hash-bow"

    @property
    def dim(self) -> int:
        # OpenAI text-embedding-3-small = 1536; Gemini text-embedding-004 = 768.
        if self._client is not None:
            return 1536
        if self._gemini is not None:
            return self.settings.gemini_embed_dim
        return self.settings.local_embed_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._client is not None:
            return self._embed_openai(texts)
        if self._gemini is not None:
            return self._embed_gemini(texts)
        return [self._embed_local(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(
            model=self.settings.openai_embed_model, input=texts
        )
        return [d.embedding for d in resp.data]

    def _embed_gemini(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        resp = self._gemini.models.embed_content(
            model=self.settings.gemini_embed_model,
            contents=texts,
            config=types.EmbedContentConfig(
                output_dimensionality=self.settings.gemini_embed_dim
            ),
        )
        return [list(e.values) for e in resp.embeddings]

    def _embed_local(self, text: str) -> list[float]:
        dim = self.settings.local_embed_dim
        vec = [0.0] * dim
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return vec
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h >> 1) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
