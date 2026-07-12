"""FAISS vector store: append-only index with save/load.

Responsibilities (single): persist and grow a FAISS index of chunk embeddings.
Uses ``IndexFlatIP`` (inner product); with L2-normalized vectors this is exact
cosine similarity. Vectors are appended in order, so a vector's position in the
index is its stable ``embedding_id`` and matches the metadata row index.

Designed for incremental use: ``add`` appends new vectors without rebuilding,
and ``save`` persists the current index. It stores no metadata itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger("aquamind.knowledge.vector_store")


class VectorStoreError(Exception):
    """A FAISS index operation failed."""


class FaissVectorStore:
    """Append-only FAISS index of embeddings, keyed by insertion position."""

    def __init__(self, index_path: Path) -> None:
        self._index_path = index_path
        self._index = None  # created on first add or loaded from disk

    # -- persistence ------------------------------------------------------ #

    def load(self) -> None:
        """Load an existing index from disk, if present."""
        if self._index_path.exists():
            try:
                self._index = faiss.read_index(str(self._index_path))
                logger.info("Loaded FAISS index (%d vectors) from %s.",
                            self._index.ntotal, self._index_path)
            except Exception as error:  # noqa: BLE001
                raise VectorStoreError(f"Failed to read FAISS index: {error}") from error

    def save(self) -> None:
        """Persist the current index to disk."""
        if self._index is None:
            return
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self._index_path))
        except Exception as error:  # noqa: BLE001
            raise VectorStoreError(f"Failed to write FAISS index: {error}") from error

    # -- mutation --------------------------------------------------------- #

    def add(self, vectors: np.ndarray) -> list[int]:
        """Append ``vectors`` and return their assigned embedding ids (positions)."""
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            return []
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if self._index is None:
            self._index = faiss.IndexFlatIP(vectors.shape[1])
        if vectors.shape[1] != self._index.d:
            raise VectorStoreError(
                f"Embedding dimension {vectors.shape[1]} does not match index dimension {self._index.d}."
            )

        start_id = self._index.ntotal
        try:
            self._index.add(vectors)
        except Exception as error:  # noqa: BLE001
            raise VectorStoreError(f"Failed to add vectors to FAISS index: {error}") from error
        return list(range(start_id, self._index.ntotal))

    # -- introspection ---------------------------------------------------- #

    @property
    def count(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)

    @property
    def dimension(self) -> int:
        return 0 if self._index is None else int(self._index.d)

    @property
    def index_path(self) -> Path:
        return self._index_path
