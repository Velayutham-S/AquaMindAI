"""Chunk metadata store: one record per embedded chunk.

Responsibilities (single): persist an ordered list of chunk metadata records.
The list is ordered by ``embedding_id`` so record ``i`` describes FAISS vector
``i`` -- the future retriever maps a search hit straight back to its metadata.

Each record captures document, category, source path, page, chunk index, a short
section/heading hint, character count, chunk id, and embedding id. It stores no
vectors and computes no embeddings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("aquamind.knowledge.metadata")


class MetadataStore:
    """Append-only, ordered store of per-chunk metadata records."""

    def __init__(self, metadata_path: Path) -> None:
        self._metadata_path = metadata_path
        self._records: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if not self._metadata_path.exists():
            return []
        try:
            data = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as error:
            logger.warning("Metadata unreadable (%s); starting fresh.", error)
            return []

    def append(self, records: list[dict]) -> None:
        """Append chunk records (must be provided in embedding-id order)."""
        self._records.extend(records)

    def save(self) -> None:
        """Persist all records to disk."""
        self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata_path.write_text(
            json.dumps(self._records, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def next_embedding_id(self) -> int:
        """The embedding id the next appended chunk will receive."""
        return len(self._records)
