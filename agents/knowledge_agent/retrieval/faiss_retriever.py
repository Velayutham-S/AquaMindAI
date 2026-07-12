"""FAISS similarity search for the retrieval pipeline.

Responsibility (single): load the existing FAISS index (built by the knowledge
base pipeline) read-only and return the top-K nearest chunk ids with their
cosine-similarity scores. Because the index is ``IndexFlatIP`` over L2-normalized
vectors, inner product equals cosine similarity.

It reads the same index file the vector store wrote; it never modifies, rebuilds,
or writes the index.
"""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger("aquamind.knowledge.retrieval")


class RetrieverError(Exception):
    """The FAISS index is missing or a search failed."""


class FaissRetriever:
    """Loads the FAISS index and performs top-K cosine similarity search."""

    def __init__(self, index_path: Path) -> None:
        self._index_path = index_path
        self._index = None

    def load(self) -> None:
        """Load the existing FAISS index (read-only)."""
        if not self._index_path.exists():
            raise RetrieverError(
                f"FAISS index not found: {self._index_path}. Build the knowledge base first."
            )
        try:
            self._index = faiss.read_index(str(self._index_path))
        except Exception as error:  # noqa: BLE001
            raise RetrieverError(f"Failed to read FAISS index: {error}") from error
        logger.info("Loaded FAISS index for retrieval (%d vectors, dim=%d).",
                    self._index.ntotal, self._index.d)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Return up to ``top_k`` ``(embedding_id, score)`` pairs, best first.

        Fewer than ``top_k`` results are returned if the index holds fewer
        vectors; FAISS sentinel ids (-1) are filtered out.
        """
        if self._index is None:
            raise RetrieverError("Index not loaded; call load() first.")
        if top_k <= 0:
            return []
        if query_vector.shape[1] != self._index.d:
            raise RetrieverError(
                f"Query dimension {query_vector.shape[1]} != index dimension {self._index.d}."
            )

        k = min(top_k, self._index.ntotal)
        try:
            scores, ids = self._index.search(np.ascontiguousarray(query_vector, dtype=np.float32), k)
        except Exception as error:  # noqa: BLE001
            raise RetrieverError(f"FAISS search failed: {error}") from error

        results: list[tuple[int, float]] = []
        for embedding_id, score in zip(ids[0].tolist(), scores[0].tolist()):
            if embedding_id != -1:
                results.append((int(embedding_id), float(score)))
        return results

    @property
    def count(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)

    @property
    def dimension(self) -> int:
        return 0 if self._index is None else int(self._index.d)
