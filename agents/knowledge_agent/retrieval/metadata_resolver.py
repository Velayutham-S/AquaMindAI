"""Metadata resolution for the retrieval pipeline.

Responsibility (single): turn FAISS search hits (embedding_id + score) into fully
described retrieved chunks -- document, category, page, section, chunk id, source
path, similarity score, and the chunk text.

Because the knowledge base stores chunk *metadata* but not chunk *text* (by
design), the text is reconstructed deterministically at resolve time by replaying
the same extract -> clean -> chunk steps on the source PDF and selecting the
chunk at ``chunk_index``. The extractor, cleaner and chunker are injected (the
exact components used at index time), so reconstruction is identical to indexing.
Per-document reconstructions are cached so a query touching several chunks of one
PDF parses it only once. The knowledge base is never modified.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("aquamind.knowledge.retrieval")


class MetadataResolutionError(Exception):
    """The metadata file is missing or unreadable."""


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieval result with full provenance and text."""

    rank: int
    score: float
    embedding_id: int
    chunk_id: str
    document: str
    category: str
    page: int
    section: str | None
    source_path: str
    text: str


class MetadataResolver:
    """Resolves embedding ids to metadata records and reconstructs chunk text."""

    def __init__(self, metadata_path: Path, project_root: Path,
                 text_cleaner, text_chunker, extract_pages_fn) -> None:
        self._metadata_path = metadata_path
        self._project_root = project_root
        self._cleaner = text_cleaner
        self._chunker = text_chunker
        self._extract_pages = extract_pages_fn
        self._records: list[dict] = self._load_metadata()
        self._chunk_cache: dict[str, list[str]] = {}

    def _load_metadata(self) -> list[dict]:
        if not self._metadata_path.exists():
            raise MetadataResolutionError(
                f"Metadata not found: {self._metadata_path}. Build the knowledge base first."
            )
        try:
            data = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise MetadataResolutionError(f"Failed to read metadata: {error}") from error
        if not isinstance(data, list):
            raise MetadataResolutionError("Metadata file is not a list of records.")
        return data

    def resolve(self, hits: list[tuple[int, float]]) -> list[RetrievedChunk]:
        """Resolve ``(embedding_id, score)`` hits into ordered RetrievedChunk items."""
        resolved: list[RetrievedChunk] = []
        for rank, (embedding_id, score) in enumerate(hits, start=1):
            if not 0 <= embedding_id < len(self._records):
                logger.warning("Embedding id %d has no metadata record; skipping.", embedding_id)
                continue
            record = self._records[embedding_id]
            resolved.append(RetrievedChunk(
                rank=rank,
                score=score,
                embedding_id=embedding_id,
                chunk_id=record.get("chunk_id", ""),
                document=record.get("document", ""),
                category=record.get("category", ""),
                page=record.get("page", -1),
                section=record.get("section"),
                source_path=record.get("source_path", ""),
                text=self._reconstruct_text(record),
            ))
        return resolved

    # -- deterministic chunk-text reconstruction -------------------------- #

    def _reconstruct_text(self, record: dict) -> str:
        source_path = record.get("source_path", "")
        chunk_index = record.get("chunk_index", -1)
        chunks = self._document_chunks(source_path)
        if 0 <= chunk_index < len(chunks):
            return chunks[chunk_index]
        logger.warning("Could not reconstruct text for chunk_index %s of '%s'.",
                       chunk_index, source_path)
        return ""

    def _document_chunks(self, source_path: str) -> list[str]:
        """Return the document's ordered chunk texts (cached), matching indexing."""
        if source_path in self._chunk_cache:
            return self._chunk_cache[source_path]

        absolute = (self._project_root / source_path)
        chunks: list[str] = []
        if not absolute.exists():
            logger.warning("Source PDF missing for text reconstruction: %s", absolute)
            self._chunk_cache[source_path] = chunks
            return chunks

        try:
            pages = self._extract_pages(absolute)
            for raw_text in pages:
                cleaned = self._cleaner.clean(raw_text)
                if not cleaned:
                    continue
                chunks.extend(self._chunker.chunk(cleaned))
        except Exception as error:  # noqa: BLE001 - reconstruction is best-effort
            logger.warning("Text reconstruction failed for '%s' (%s).", source_path, error)
            chunks = []

        self._chunk_cache[source_path] = chunks
        return chunks
