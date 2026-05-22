#!/usr/bin/env python3
"""Integration test: PDF ingestion → embedding → retrieval.

Run from the backend/ directory::

    python test_rag_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure backend/ is on sys.path so that `rag` is importable.
_BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(_BACKEND))

from rag.build_index import main as build_index
from rag.retriever import get_retriever

# ---------------------------------------------------------------------------
# Test queries covering different clinical facets
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    "What is the relationship between EMVI and tumor deposits in rectal cancer MRI?",
    "How does circumferential resection margin predict local recurrence?",
    "What is the role of neoadjuvant chemoradiotherapy in rectal cancer treatment?",
    "Does MRI finding of tumor deposits affect prognosis?",
]


def run_tests() -> int:
    t0 = time.monotonic()

    # -- Step 1: Build / refresh the index --------------------------------
    print("=" * 64)
    print("STEP 1 — Build ChromaDB index from literature PDFs")
    print("=" * 64)
    build_index()

    # -- Step 2: Retrieve and inspect results -----------------------------
    print("\n" + "=" * 64)
    print("STEP 2 — Retrieval test queries")
    print("=" * 64)

    retriever = get_retriever()
    failures = 0

    for query in TEST_QUERIES:
        print(f"\n{'─' * 64}")
        print(f"QUERY: {query}")
        print(f"{'─' * 64}")

        results = retriever.retrieve(query)

        if not results:
            print("WARNING: No relevant chunks found (similarity below threshold).")
            failures += 1
            continue

        print(f"Retrieved {len(results)} chunk(s):\n")
        for i, r in enumerate(results, start=1):
            source = r["metadata"].get("source", "?")
            page = r["metadata"].get("page", "?")
            score = r["score"]
            content_preview = r["content"][:200].replace("\n", " ")
            # Encode ASCII-only for Windows terminal compatibility
            content_preview = content_preview.encode("ascii", errors="replace").decode("ascii")
            print(f"  [{i}] source={source} | page={page} | score={score:.4f}")
            print(f"      {content_preview}...")
            print()

    # -- Step 3: Test format_context output -------------------------------
    print("=" * 64)
    print("STEP 3 — format_context() output preview (first 600 chars)")
    print("=" * 64)
    ctx = retriever.format_context(TEST_QUERIES[0])
    if ctx:
        print(ctx[:600].encode("ascii", errors="replace").decode("ascii"))
        print("...")
    else:
        print("WARNING: format_context returned empty string.")
        failures += 1

    elapsed = time.monotonic() - t0
    print(f"\n{'=' * 64}")
    print(f"Completed in {elapsed:.1f} s.  Failures: {failures}/{len(TEST_QUERIES)}")
    print("=" * 64)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run_tests())
