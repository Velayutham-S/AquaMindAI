"""Embedding generation using a sentence-transformers model.

Responsibilities (single): turn a list of text chunks into a float32 embedding
matrix, in batches. The model is loaded lazily on first use (so importing this
module is cheap) and reused across calls, which makes incremental embedding of a
single newly-uploaded document as efficient as bulk embedding.

It never touches FAISS, metadata, or disk.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("aquamind.knowledge.embedding")


class EmbeddingError(Exception):
    """The embedding model failed to load or encode text."""


class Embedder:
    """Generates embeddings for text chunks in batches."""

    def __init__(self, model_name: str, batch_size: int = 64, normalize: bool = True) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize = normalize
        self._model = None  # loaded lazily

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model '%s' (first use may download it)...", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model ready (dimension=%d).", self.dimension)
        except Exception as error:  # noqa: BLE001
            raise EmbeddingError(f"Failed to load embedding model '{self._model_name}': {error}") from error

    @property
    def dimension(self) -> int:
        self._ensure_model()
        return int(self._model.get_sentence_embedding_dimension())

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a ``(len(texts), dimension)`` float32 embedding matrix.

        Embeddings are L2-normalized when ``normalize`` is set, so downstream
        inner-product search behaves as cosine similarity.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        self._ensure_model()
        try:
            vectors = self._model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as error:  # noqa: BLE001
            raise EmbeddingError(f"Embedding generation failed: {error}") from error
        return np.asarray(vectors, dtype=np.float32)
