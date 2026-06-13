"""FastAPI retrieval + generation service.

Endpoints:
  GET  /health        -> provider/backend status
  POST /index         -> index a local directory
  POST /ask           -> retrieve + generate (JSON)
  POST /ask/stream    -> retrieve + generate (Server-Sent Events stream)
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from .rag import RagEngine
from .schemas import AskRequest, AskResponse, IndexRequest, IndexResponse

engine = RagEngine()

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_FRONTEND = os.path.join(_PROJECT_ROOT, "frontend", "index.html")


def _index_base_dir() -> str | None:
    """If INDEX_BASE_DIR is set, restrict /index to paths inside it (deploy safety)."""
    base = os.getenv("INDEX_BASE_DIR")
    return os.path.realpath(base) if base else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Optionally index a directory on startup so a fresh deployment has content.
    auto = os.getenv("AUTO_INDEX_PATH", "")
    if auto and not os.path.isabs(auto):
        auto = os.path.join(_PROJECT_ROOT, auto)
    if auto and os.path.isdir(auto) and engine.store.count() == 0:
        try:
            engine.index_directory(auto, [".py", ".md"])
        except Exception:
            pass  # never block startup on indexing / provider errors
    yield


app = FastAPI(
    title="AI Code-Generation Assistant", version="1.0.0", lifespan=lifespan
)


@app.get("/")
def ui() -> FileResponse:
    return FileResponse(_FRONTEND)


@app.get("/health")
def health() -> dict:
    s = engine.settings
    return {
        "status": "ok",
        "vector_backend": engine.store.backend,
        "embedder": engine.embedder.name,
        "openai_enabled": s.openai_enabled,
        "gemini_enabled": s.gemini_enabled,
        "pgvector_enabled": s.pgvector_enabled,
        "indexed_chunks": engine.store.count(),
        "cache": {"hits": engine.cache.hits, "misses": engine.cache.misses},
    }


@app.post("/index", response_model=IndexResponse)
def index(req: IndexRequest) -> IndexResponse:
    base = _index_base_dir()
    if base is not None:
        target = os.path.realpath(req.root)
        if os.path.commonpath([base, target]) != base:
            raise HTTPException(
                status_code=400,
                detail=f"Indexing is restricted to paths inside {base}.",
            )
    files, chunks = engine.index_directory(req.root, req.include_ext)
    return IndexResponse(
        indexed_files=files,
        indexed_chunks=chunks,
        vector_backend=engine.store.backend,
        embedder=engine.embedder.name,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return engine.ask(req.question, mode=req.mode, top_k=req.top_k)


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    def event_stream():
        for token in engine.ask_stream(
            req.question, mode=req.mode, top_k=req.top_k
        ):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
