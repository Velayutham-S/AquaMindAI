"""Centralized configuration for the AquaMind AI Knowledge Base (RAG) pipeline.

Every tunable value and path used by the ingestion, preprocessing, chunking,
embedding, vector-store and metadata components lives here. Nothing is hardcoded
inside the components -- the orchestrator reads this config and injects the
values, so behaviour is changed in exactly one place.

This module is offline infrastructure only. It contains no retrieval, no LLM
calls, and no agent logic.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (derived, never hardcoded to an absolute location)
# --------------------------------------------------------------------------- #

KNOWLEDGE_AGENT_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = KNOWLEDGE_AGENT_DIR.parents[1]  # knowledge_agent -> agents -> root
DEFAULT_PDF_ROOT: Path = PROJECT_ROOT / "pdf"
DEFAULT_KNOWLEDGE_INPUTS_DIR: Path = KNOWLEDGE_AGENT_DIR / "knowledge_inputs"


@dataclass(frozen=True)
class KnowledgeBaseConfig:
    """All configuration for building and maintaining the knowledge base."""

    # --- source documents ---
    pdf_root: Path = DEFAULT_PDF_ROOT
    pdf_extensions: tuple[str, ...] = (".pdf",)

    # --- chunking ---
    chunk_size: int = 1000          # characters per chunk
    chunk_overlap: int = 150        # character overlap between consecutive chunks
    min_chunk_chars: int = 50       # discard fragments shorter than this

    # --- embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
    normalize_embeddings: bool = True  # enables cosine similarity via inner product

    # --- retrieval (reserved for the future Knowledge Agent; unused here) ---
    top_k: int = 5

    # --- artifact locations (the built knowledge base) ---
    knowledge_inputs_dir: Path = DEFAULT_KNOWLEDGE_INPUTS_DIR
    faiss_index_filename: str = "faiss.index"
    metadata_filename: str = "chunks_metadata.json"
    manifest_filename: str = "documents_manifest.json"

    # --- logging ---
    log_level: int = logging.INFO

    @property
    def faiss_index_path(self) -> Path:
        return self.knowledge_inputs_dir / self.faiss_index_filename

    @property
    def metadata_path(self) -> Path:
        return self.knowledge_inputs_dir / self.metadata_filename

    @property
    def manifest_path(self) -> Path:
        return self.knowledge_inputs_dir / self.manifest_filename


#: The default, ready-to-use configuration instance.
CONFIG = KnowledgeBaseConfig()


def configure_logging(level: int = CONFIG.log_level) -> None:
    """Configure clean console logging once (no ``print`` debugging anywhere)."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
        )
