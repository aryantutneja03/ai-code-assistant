"""CLI to index any local repository into the vector store.

Usage:
    python scripts/index_codebase.py <path-to-repo> [--ext .py .ts .md]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag import RagEngine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a codebase for RAG.")
    parser.add_argument("root", help="Path to the repo/directory to index")
    parser.add_argument(
        "--ext", nargs="*",
        default=[".py", ".ts", ".js", ".md", ".sql", ".json"],
        help="File extensions to include",
    )
    args = parser.parse_args()

    engine = RagEngine()
    files, chunks = engine.index_directory(args.root, include_ext=args.ext)
    print(f"Indexed {files} files / {chunks} chunks")
    print(f"  vector backend : {engine.store.backend}")
    print(f"  embedder       : {engine.embedder.name}")


if __name__ == "__main__":
    main()
