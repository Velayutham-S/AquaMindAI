"""Retrieval coordinator for the AquaMind AI knowledge base.

Orchestrates the read-only retrieval pipeline:

    query -> QueryEmbedder -> FaissRetriever -> MetadataResolver -> top-K chunks

Public API:
    RetrievalCoordinator(config).retrieve(query, top_k=None) -> list[RetrievedChunk]

It reuses the existing knowledge-base components (the indexing ``Embedder`` and
the deterministic extractor/cleaner/chunker for text reconstruction) and reads
the existing FAISS index and metadata. It performs no LLM calls, no
summarization, and no response generation, and it never modifies the knowledge
base.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# --- wire up sibling component packages (no __init__.py; import by location) ---
_RETRIEVAL_DIR = Path(__file__).resolve().parent
_KNOWLEDGE_AGENT_DIR = _RETRIEVAL_DIR.parent
for _sub in ("", "embedding", "preprocessing", "chunking", "ingestion", "retrieval"):
    _path = _KNOWLEDGE_AGENT_DIR / _sub if _sub else _KNOWLEDGE_AGENT_DIR
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import knowledge_config as kb_config  # noqa: E402  (unique module name; avoids sys.modules 'config' collision)
from embedder import Embedder  # noqa: E402  (reused indexing embedder)
from text_cleaner import TextCleaner  # noqa: E402  (reused)
from text_chunker import TextChunker  # noqa: E402  (reused)
from pdf_extractor import extract_pages  # noqa: E402  (reused)
from query_embedder import QueryEmbedder  # noqa: E402
from faiss_retriever import FaissRetriever  # noqa: E402
from metadata_resolver import MetadataResolver, RetrievedChunk  # noqa: E402

logger = logging.getLogger("aquamind.knowledge.retrieval")

__all__ = ["RetrievalCoordinator", "RetrievedChunk"]


class RetrievalCoordinator:
    """End-to-end retrieval over the existing knowledge base."""

    def __init__(self, config: kb_config.KnowledgeBaseConfig = kb_config.CONFIG) -> None:
        self._config = config

        # Reuse the exact indexing embedder (same model + normalization).
        embedder = Embedder(
            model_name=config.embedding_model,
            batch_size=config.embedding_batch_size,
            normalize=config.normalize_embeddings,
        )
        self._query_embedder = QueryEmbedder(embedder)

        self._retriever = FaissRetriever(config.faiss_index_path)
        self._retriever.load()

        self._resolver = MetadataResolver(
            metadata_path=config.metadata_path,
            project_root=config.pdf_root.parent,  # source_path is relative to project root
            text_cleaner=TextCleaner(),
            text_chunker=TextChunker(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                min_chunk_chars=config.min_chunk_chars,
            ),
            extract_pages_fn=extract_pages,
        )
        logger.info("RetrievalCoordinator ready (vectors=%d, default top_k=%d).",
                    self._retriever.count, config.top_k)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the top-K most relevant chunks for ``query`` (best first)."""
        k = self._config.top_k if top_k is None else top_k
        query_vector = self._query_embedder.embed_query(query)
        hits = self._retriever.search(query_vector, k)
        return self._resolver.resolve(hits)

    # -- introspection (for reporting) ------------------------------------ #

    @property
    def embedding_model(self) -> str:
        return self._query_embedder.model_name

    @property
    def embedding_dimension(self) -> int:
        return self._retriever.dimension

    @property
    def vector_count(self) -> int:
        return self._retriever.count
