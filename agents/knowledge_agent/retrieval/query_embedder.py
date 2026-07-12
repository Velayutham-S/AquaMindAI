"""Query embedding for the retrieval pipeline.

Responsibility (single): turn one user query string into a single embedding
vector, using the *same* embedding model that indexed the knowledge base.

It reuses the existing indexing ``Embedder`` (injected by the coordinator) so
the query and the stored chunks live in the same vector space. It does not load
FAISS, resolve metadata, or call any LLM.
"""

from __future__ import annotations

import numpy as np


class QueryEmbeddingError(Exception):
    """The query could not be embedded."""


class QueryEmbedder:
    """Adapts the batch indexing embedder to a single-query interface."""

    def __init__(self, embedder) -> None:
        # ``embedder`` is the shared indexing Embedder (duck-typed: .embed / .dimension / .model_name),
        # injected so the query uses exactly the model used at index time.
        self._embedder = embedder

    def embed_query(self, query: str) -> np.ndarray:
        """Return a ``(1, dimension)`` float32 embedding for ``query``.

        The embedding is normalized identically to the indexed chunks, so inner
        product against the FAISS index equals cosine similarity.
        """
        if not query or not query.strip():
            raise QueryEmbeddingError("Query is empty.")
        vector = self._embedder.embed([query.strip()])
        if vector.ndim != 2 or vector.shape[0] != 1:
            raise QueryEmbeddingError("Embedder did not return a single query vector.")
        return np.ascontiguousarray(vector, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    @property
    def model_name(self) -> str:
        return self._embedder.model_name
