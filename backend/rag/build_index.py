#!/usr/bin/env python3
"""One-shot script: ingest PDF literature → chunk → embed → persist to ChromaDB.

Usage (from the backend/ directory)::

    python -m rag.build_index

Or from the project root::

    python backend/rag/build_index.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .document_loader import PDFLoader
from .embedding import EmbeddingModel
from .vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag.build_index")

# Resolve paths relative to this file.
_MODULE_DIR = Path(__file__).resolve().parent          # backend/rag/
_PROJECT_ROOT = _MODULE_DIR.parent                     # backend/
_PDF_DIR = _PROJECT_ROOT.parent / "literature_pdf"     # v2_project/literature_pdf/
_STORE_DIR = _PROJECT_ROOT / "chroma_store"            # backend/chroma_store/


def main() -> None:
    t0 = time.monotonic()

    # ------------------------------------------------------------------
    # 1. Load & chunk PDFs
    # ------------------------------------------------------------------
    logger.info("Loading PDFs from %s ...", _PDF_DIR)
    loader = PDFLoader(chunk_size=800, chunk_overlap=150)
    documents = loader.load_directory(_PDF_DIR)
    logger.info("Parsed %d chunks from %d PDF(s).", len(documents), len({d.source for d in documents}))

    if not documents:
        logger.error("No documents produced — check PDF directory.")
        return

    # ------------------------------------------------------------------
    # 2. Embed
    # ------------------------------------------------------------------
    logger.info("Loading embedding model (first run downloads the model) ...")
    embedding = EmbeddingModel(model_name="BAAI/bge-m3")
    texts = [doc.content for doc in documents]
    embeddings = embedding.embed_documents(texts)
    logger.info("Generated %d embeddings (dim=%d).", len(embeddings), embedding.dim)

    # ------------------------------------------------------------------
    # 3. Persist to ChromaDB
    # ------------------------------------------------------------------
    logger.info("Persisting to ChromaDB at %s ...", _STORE_DIR)
    store = VectorStore(persist_dir=str(_STORE_DIR))
    store.add_documents(documents, embeddings)

    elapsed = time.monotonic() - t0
    logger.info("Index build complete in %.1f s. Collection size: %d chunks.", elapsed, store.count)


if __name__ == "__main__":
    main()
