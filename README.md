# AI Code-Generation Assistant with RAG over Codebases

A coding assistant that answers questions about a code repository and
**generates code** by retrieving the most relevant files, functions, and docs
from the repo via **semantic search**, then asking an LLM to answer grounded on
that retrieved context.

Stack: **Python (FastAPI)** service, **pgvector** vector store, **OpenAI** LLM +
`text-embedding-3` embeddings, **semantic search**, a simple **semantic cache**,
and **streaming** responses.

> Runs **offline by default** using a local in-memory vector store, a local
> hash-based embedder, and a deterministic local LLM stub — so you can try the
> full pipeline with **no API key and no Postgres**. When you set
> `OPENAI_API_KEY` and/or `DATABASE_URL`, it automatically upgrades to OpenAI
> and pgvector.

## Architecture

```
  client ──HTTP──▶  Python FastAPI service
                    1. AST-aware chunking      (chunking.py)
                    2. text-embedding-3        (embeddings.py)
                    3. pgvector store          (vector_store.py)
                    4. semantic search         (semantic_search.py)
                    5. semantic cache          (cache.py)
                    6. OpenAI generation       (llm_client.py)
```

## Project structure

```
ai-code-assistant/
  backend/
    main.py             # FastAPI app: /index, /ask, /ask/stream, /health
    config.py           # env-driven settings + provider auto-detection
    schemas.py          # pydantic request/response models
    chunking.py         # AST-aware code chunking (Python + generic + markdown)
    embeddings.py       # OpenAI text-embedding-3 + local fallback
    vector_store.py     # pgvector store + in-memory fallback
    semantic_search.py  # cosine-similarity retrieval (top-k)
    cache.py            # semantic cache (embedding-similarity keyed)
    llm_client.py       # OpenAI client with offline local stub
    prompts.py          # code-generation + QA prompt templates
    rag.py              # orchestrates retrieve -> generate
  eval/
    eval_harness.py     # scores faithfulness + context precision
    test_questions.json
  scripts/
    index_codebase.py   # CLI: index any local repo
    demo.py             # end-to-end offline demo (no key needed)
  sql/
    schema.sql          # pgvector table + index DDL
  requirements.txt
  .env.example
```

## Quick start (offline, no key)

```powershell
cd projects\ai-code-assistant
pip install -r requirements.txt
python scripts\demo.py
```

The demo indexes this project's own backend code, then asks a few questions and
generates code, printing retrieved context + answers.

## Run the API

```powershell
cd projects\ai-code-assistant
uvicorn backend.main:app --reload --port 8000
```

Ask a question (with SSE streaming):

```powershell
curl -N -X POST http://localhost:8000/ask/stream `
  -H "Content-Type: application/json" `
  -d '{"question":"How does semantic search retrieve code chunks?","mode":"qa"}'
```

## Going live (real providers)

Copy `.env.example` to `.env` and fill in any of:

| Variable          | Enables                                            |
| ----------------- | -------------------------------------------------- |
| `OPENAI_API_KEY`  | `text-embedding-3` embeddings + GPT-4o generation  |
| `DATABASE_URL`    | pgvector storage (`postgresql://...`)              |

Create the pgvector schema once:

```powershell
psql "$env:DATABASE_URL" -f sql\schema.sql
```

## Resume bullets this implements

- **Semantic search** over a **pgvector** store of `text-embedding-3`
  embeddings.
- **Code-generation prompts** with few-shot examples + strict output rules
  using the **OpenAI** API.
- **Simple caching** to reuse answers and save tokens on repeated questions.
- **Python FastAPI** service with **streaming** replies and basic tests/eval.
