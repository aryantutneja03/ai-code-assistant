"""Pydantic request/response models for the retrieval service."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CodeChunk(BaseModel):
    chunk_id: str
    path: str
    symbol: str = ""
    kind: str = "block"  # function | class | block | doc
    start_line: int = 0
    end_line: int = 0
    content: str


class IndexRequest(BaseModel):
    root: str = Field(..., description="Path to a local repo / directory to index")
    include_ext: list[str] = Field(
        default_factory=lambda: [".py", ".ts", ".js", ".md", ".sql", ".json"]
    )


class IndexResponse(BaseModel):
    indexed_files: int
    indexed_chunks: int
    vector_backend: str
    embedder: str


class RetrievedChunk(BaseModel):
    chunk: CodeChunk
    score: float


class AskRequest(BaseModel):
    question: str
    mode: Literal["qa", "codegen"] = "qa"
    top_k: int | None = None


class AskResponse(BaseModel):
    answer: str
    model_used: str
    provider: str
    cached: bool
    contexts: list[RetrievedChunk]
