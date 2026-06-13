"""RAG evaluation harness.

Computes two offline, reference-light metrics over `test_questions.json`:

  * context_precision  -> fraction of retrieved chunks whose file path is in the
                          question's `expected_files` (are we retrieving the
                          right files?).
  * faithfulness       -> fraction of `expected_keywords` that appear in the
                          generated answer AND in the retrieved context (is the
                          answer grounded in what we retrieved?).

These are intentionally simple, deterministic proxies so the harness runs with
no API keys. With real providers configured the same metrics still apply.

Run:
    python eval/eval_harness.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag import RagEngine  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)


def _context_precision(contexts, expected_files: list[str]) -> float:
    if not contexts:
        return 0.0
    expected = {os.path.basename(f) for f in expected_files}
    hits = sum(1 for c in contexts if os.path.basename(c.chunk.path) in expected)
    return hits / len(contexts)


def _faithfulness(answer: str, contexts, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    ctx_text = " ".join(c.chunk.content for c in contexts).lower()
    ans = answer.lower()
    grounded = sum(1 for k in keywords if k.lower() in ans or k.lower() in ctx_text)
    return grounded / len(keywords)


def main() -> None:
    engine = RagEngine()
    files, chunks = engine.index_directory(
        os.path.join(PROJECT_ROOT, "backend"),
        include_ext=[".py"],
    )
    print(f"Indexed {files} files / {chunks} chunks "
          f"(backend={engine.store.backend}, embedder={engine.embedder.name})\n")

    with open(os.path.join(HERE, "test_questions.json"), encoding="utf-8") as fh:
        cases = json.load(fh)["questions"]

    cp_scores, faith_scores = [], []
    for i, case in enumerate(cases, start=1):
        resp = engine.ask(case["question"], mode=case["mode"])
        cp = _context_precision(resp.contexts, case.get("expected_files", []))
        faith = _faithfulness(resp.answer, resp.contexts, case.get("expected_keywords", []))
        cp_scores.append(cp)
        faith_scores.append(faith)
        top = resp.contexts[0].chunk.path if resp.contexts else "-"
        print(f"[{i}] {case['mode']:<7} ctx_precision={cp:.2f} faithfulness={faith:.2f} "
              f"top={top}")
        print(f"    Q: {case['question']}")

    n = len(cases)
    print("\n=== Summary ===")
    print(f"avg context_precision : {sum(cp_scores)/n:.3f}")
    print(f"avg faithfulness      : {sum(faith_scores)/n:.3f}")
    print(f"cache hits/misses     : {engine.cache.hits}/{engine.cache.misses}")


if __name__ == "__main__":
    main()
