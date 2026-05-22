from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from .document_loader import Document

logger = logging.getLogger(__name__)


class VectorStore:
    """Persistent ChromaDB-backed vector store for document chunks.

    ChromaDB is imported lazily inside ``__init__`` so the backend process can
    start even when the package is not installed.
    """

    COLLECTION_NAME = "rectal_mri_literature"

    def __init__(self, persist_dir: str | Path = "./chroma_store") -> None:
        import chromadb
        from chromadb.api.types import EmbeddingFunction

        class _PassThroughEmbedding(EmbeddingFunction):
            """Signals that we pass pre-computed vectors via ``embeddings=``."""

            def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
                raise RuntimeError("PassThrough: use add(embeddings=...) directly")

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=_PassThroughEmbedding(),
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(documents)} documents vs {len(embeddings)} embeddings"
            )

        ids = [
            f"{doc.source}:p{doc.page}:c{doc.chunk_index}"
            for doc in documents
        ]
        metadatas = [
            {"source": doc.source, "page": doc.page, "chunk_index": doc.chunk_index}
            for doc in documents
        ]
        texts = [doc.content for doc in documents]

        self._collection.add(
            ids=ids,
            embeddings=list(embeddings),
            metadatas=metadatas,
            documents=texts,
        )
        logger.info("Added %d chunks to ChromaDB collection.", len(ids))

    def query(
        self,
        query_embedding: Sequence[float],
        n_results: int = 5,
        similarity_threshold: float = 0.0,
    ) -> list[dict]:
        """Return nearest documents above a cosine-similarity threshold.

        Each result dict has keys: ``id``, ``content``, ``metadata``, ``score``.
        """
        raw = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        results: list[dict] = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            cosine_sim = 1.0 - distances[i]  # ChromaDB stores cosine distance
            if cosine_sim >= similarity_threshold:
                results.append(
                    {
                        "id": doc_id,
                        "content": docs[i] if docs else "",
                        "metadata": metas[i] if metas else {},
                        "score": round(cosine_sim, 4),
                    }
                )

        return results
