from __future__ import annotations

import logging
from pathlib import Path

from .embedding import EmbeddingModel
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton — lazy initialisation so the model downloads only when first used.
# ---------------------------------------------------------------------------

_retriever: "Retriever | None" = None


class Retriever:
    """Orchestrates embedding + vector-store lookup for a user query."""

    def __init__(
        self,
        embedding: EmbeddingModel,
        vector_store: VectorStore,
        top_k: int = 4,
        similarity_threshold: float = 0.35,
    ) -> None:
        self._embedding = embedding
        self._store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[dict]:
        """Return top-k relevant chunks from the literature."""
        if self._store.count == 0:
            logger.warning("Vector store is empty — no literature indexed.")
            return []

        query_embedding = self._embedding.embed_query(query)
        return self._store.query(
            query_embedding=query_embedding,
            n_results=self.top_k,
            similarity_threshold=self.similarity_threshold,
        )

    def format_context(self, query: str) -> str:
        """Retrieve and format chunks into a single context string.

        Returns an empty string when nothing relevant is found.
        """
        results = self.retrieve(query)
        if not results:
            return ""

        blocks: list[str] = []
        for i, r in enumerate(results, start=1):
            source = r["metadata"].get("source", "unknown")
            blocks.append(
                f"[Source {i}: {source}, "
                f"page {r['metadata'].get('page', '?')}, "
                f"relevance={r['score']:.2f}]\n{r['content']}"
            )
        return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Lazy singleton factory — call from your chat endpoint or FastAPI lifespan.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent  # backend/

def get_retriever(
    persist_dir: str | None = None,
    embedding_model: str = "BAAI/bge-m3",
    top_k: int = 4,
    similarity_threshold: float = 0.35,
) -> Retriever:
    """Return the module-level Retriever singleton, initialising on first call."""
    global _retriever
    if _retriever is None:
        store_dir = persist_dir or str(ROOT / "chroma_store")
        _retriever = Retriever(
            embedding=EmbeddingModel(model_name=embedding_model),
            vector_store=VectorStore(persist_dir=store_dir),
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        logger.info(
            "Retriever initialised (model=%s, chunks=%d, store=%s).",
            embedding_model,
            _retriever._store.count,
            store_dir,
        )
    return _retriever
