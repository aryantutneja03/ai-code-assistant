"""Prompt templates for QA and code generation.

Code-generation prompts use few-shot exemplars and strict output rules so the
model returns a single fenced code block plus a short explanation. QA prompts
require the model to ground answers in the retrieved context and cite file
paths.
"""
from __future__ import annotations

from .schemas import RetrievedChunk

SYSTEM_QA = (
    "You are a senior engineer answering questions about a specific codebase. "
    "Answer ONLY using the provided context. If the context is insufficient, "
    "say so explicitly. Always cite the file paths you used in the form "
    "[path:start-end]."
)

SYSTEM_CODEGEN = (
    "You are a senior engineer that writes code for a specific codebase. "
    "Follow the conventions visible in the provided context. "
    "Output rules: (1) return exactly ONE fenced code block with the code, "
    "(2) after the block, add a short 'Why' section in <= 3 bullet points, "
    "(3) do not invent APIs that are not shown or standard. "
    "Cite the context files you relied on as [path:start-end]."
)

# A tiny few-shot exemplar steering the code-gen output format.
CODEGEN_FEWSHOT = """\
Example
Question: Add a function that returns the cosine similarity of two vectors.
Answer:
```python
def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)
```
Why:
- Matches the project's plain-Python, dependency-free style.
- Guards against zero-norm vectors.
"""


def format_context(contexts: list[RetrievedChunk]) -> str:
    blocks = []
    for r in contexts:
        c = r.chunk
        loc = f"{c.path}:{c.start_line}-{c.end_line}"
        header = f"# [{loc}] {c.kind} {c.symbol}".rstrip()
        blocks.append(f"{header}\n{c.content}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, contexts: list[RetrievedChunk], mode: str):
    context_text = format_context(contexts)
    if mode == "codegen":
        system = SYSTEM_CODEGEN
        user = (
            f"{CODEGEN_FEWSHOT}\n\n"
            f"Context from the codebase:\n{context_text}\n\n"
            f"Now answer for this codebase.\nQuestion: {question}"
        )
    else:
        system = SYSTEM_QA
        user = (
            f"Context from the codebase:\n{context_text}\n\n"
            f"Question: {question}"
        )
    return system, user
