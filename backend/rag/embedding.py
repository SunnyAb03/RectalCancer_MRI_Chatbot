from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Thin wrapper around a local HuggingFace embedding model.

    Uses ``BAAI/bge-m3`` by default — multilingual, strong on biomedical text,
    runs entirely offline.  The SentenceTransformer is imported lazily so the
    backend process can start even when the heavy ML stack is not installed.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None  # type: ignore[valid-type]

    @property
    def model(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device="cpu",
                trust_remote_code=True,
            )
        return self._model

    @property
    def dim(self) -> int:
        return self.model.get_embedding_dimension()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents. Returns list of embedding vectors."""
        if not texts:
            return []
        embeddings = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=8,
        )
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        emb = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.tolist()
