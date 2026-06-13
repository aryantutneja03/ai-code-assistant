"""End-to-end offline demo (no API keys, no Postgres required).

It indexes this project's own `backend/` directory, then runs a few QA and
code-generation questions, printing retrieved context + the generated answer.

    python scripts/demo.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag import RagEngine  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    engine = RagEngine()

    hr("1) Indexing backend/ code")
    files, chunks = engine.index_directory(
        os.path.join(PROJECT_ROOT, "backend"), include_ext=[".py"]
    )
    print(f"indexed {files} files / {chunks} chunks")
    print(f"vector backend = {engine.store.backend}")
    print(f"embedder       = {engine.embedder.name}")
    print(f"openai={engine.settings.openai_enabled} "
          f"pgvector={engine.settings.pgvector_enabled}")

    demos = [
        ("How does the OpenAI client generate answers?", "qa"),
        ("How does semantic search retrieve code chunks?", "qa"),
        ("Write a function that returns how many chunks are indexed.", "codegen"),
    ]

    for question, mode in demos:
        hr(f"[{mode}] {question}")
        resp = engine.ask(question, mode=mode)
        print(f"provider={resp.provider} model={resp.model_used} cached={resp.cached}")
        print("\nTop retrieved context:")
        for r in resp.contexts:
            print(f"  - {r.chunk.path}:{r.chunk.start_line}-{r.chunk.end_line} "
                  f"({r.chunk.kind} {r.chunk.symbol}) score={r.score}")
        print("\nAnswer:\n" + resp.answer)

    # Demonstrate the semantic cache by repeating a question.
    hr("Semantic cache check (repeat first question)")
    resp = engine.ask(demos[0][0], mode="qa")
    print(f"cached={resp.cached} (hits={engine.cache.hits}, misses={engine.cache.misses})")


if __name__ == "__main__":
    main()
