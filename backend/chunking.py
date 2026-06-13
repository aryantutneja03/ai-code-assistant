"""AST-aware code chunking.

For Python files we split on top-level functions and classes using the `ast`
module so each chunk is a semantically meaningful unit (a function/class plus
its docstring). For every other file type we fall back to a sliding-window
line-based chunker. Markdown is chunked on headings.
"""
from __future__ import annotations

import ast
import hashlib
import os

from .schemas import CodeChunk

_WINDOW_LINES = 60
_WINDOW_OVERLAP = 12


def _chunk_id(path: str, start: int, end: int) -> str:
    raw = f"{path}:{start}-{end}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def _python_chunks(path: str, source: str) -> list[CodeChunk]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _window_chunks(path, source)

    lines = source.splitlines()
    chunks: list[CodeChunk] = []
    covered: set[int] = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start) or start
            body = "\n".join(lines[start - 1 : end])
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            chunks.append(
                CodeChunk(
                    chunk_id=_chunk_id(path, start, end),
                    path=path,
                    symbol=node.name,
                    kind=kind,
                    start_line=start,
                    end_line=end,
                    content=body,
                )
            )
            covered.update(range(start, end + 1))

    # Capture module-level code (imports, constants) not covered by a def/class.
    leftover = [i for i in range(1, len(lines) + 1) if i not in covered]
    if leftover:
        block = "\n".join(lines[i - 1] for i in leftover).strip()
        if block:
            chunks.append(
                CodeChunk(
                    chunk_id=_chunk_id(path, 0, 0),
                    path=path,
                    symbol="<module>",
                    kind="block",
                    start_line=leftover[0],
                    end_line=leftover[-1],
                    content=block,
                )
            )
    return chunks or _window_chunks(path, source)


def _markdown_chunks(path: str, source: str) -> list[CodeChunk]:
    lines = source.splitlines()
    chunks: list[CodeChunk] = []
    buf: list[str] = []
    start = 1
    for i, line in enumerate(lines, start=1):
        if line.startswith("#") and buf:
            chunks.append(_md_chunk(path, buf, start, i - 1))
            buf = []
            start = i
        buf.append(line)
    if buf:
        chunks.append(_md_chunk(path, buf, start, len(lines)))
    return chunks


def _md_chunk(path: str, buf: list[str], start: int, end: int) -> CodeChunk:
    content = "\n".join(buf).strip()
    heading = next((l.lstrip("# ").strip() for l in buf if l.startswith("#")), "")
    return CodeChunk(
        chunk_id=_chunk_id(path, start, end),
        path=path,
        symbol=heading,
        kind="doc",
        start_line=start,
        end_line=end,
        content=content,
    )


def _window_chunks(path: str, source: str) -> list[CodeChunk]:
    lines = source.splitlines()
    chunks: list[CodeChunk] = []
    step = _WINDOW_LINES - _WINDOW_OVERLAP
    for start in range(0, max(len(lines), 1), step):
        window = lines[start : start + _WINDOW_LINES]
        if not window:
            break
        content = "\n".join(window).strip()
        if content:
            chunks.append(
                CodeChunk(
                    chunk_id=_chunk_id(path, start + 1, start + len(window)),
                    path=path,
                    symbol="",
                    kind="block",
                    start_line=start + 1,
                    end_line=start + len(window),
                    content=content,
                )
            )
    return chunks


def chunk_file(path: str, source: str) -> list[CodeChunk]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".py":
        return _python_chunks(path, source)
    if ext in {".md", ".markdown"}:
        return _markdown_chunks(path, source)
    return _window_chunks(path, source)


def chunk_directory(root: str, include_ext: list[str]) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "dist", "build"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext not in include_ext:
                continue
            full = os.path.join(dirpath, name)
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                    source = fh.read()
            except OSError:
                continue
            rel = os.path.relpath(full, root)
            chunks.extend(chunk_file(rel, source))
    return chunks
