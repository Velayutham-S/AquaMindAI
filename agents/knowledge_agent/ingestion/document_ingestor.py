"""Document ingestion: recursive PDF discovery, manifest, and new-file detection.

Responsibilities (single):
  * recursively discover every PDF under the configured root,
  * identify which PDFs are new (not yet in the knowledge base),
  * maintain a persistent manifest of processed documents.

Documents are identified by the SHA-256 hash of their bytes. Keying on content
hash means identical files (even under different names/folders) are treated as
one document, so duplicate PDFs are skipped automatically.

This component performs no cleaning, chunking, or embedding.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("aquamind.knowledge.ingestion")


@dataclass(frozen=True)
class DiscoveredDocument:
    """A PDF found on disk, with the identity used for deduplication."""

    path: Path
    doc_id: str          # sha256 of file bytes
    filename: str
    category: str        # immediate parent folder name (source grouping)
    size_bytes: int
    modified_time: str


class DocumentIngestor:
    """Discovers PDFs and tracks which have already been ingested."""

    def __init__(self, pdf_root: Path, manifest_path: Path, extensions: tuple[str, ...] = (".pdf",)) -> None:
        self._pdf_root = pdf_root
        self._manifest_path = manifest_path
        self._extensions = tuple(ext.lower() for ext in extensions)
        self._manifest: dict[str, dict] = self._load_manifest()

    # -- manifest persistence -------------------------------------------- #

    def _load_manifest(self) -> dict[str, dict]:
        if not self._manifest_path.exists():
            return {}
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as error:
            logger.warning("Manifest unreadable (%s); starting a fresh manifest.", error)
            return {}

    def save_manifest(self) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(self._manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # -- discovery -------------------------------------------------------- #

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def discover(self) -> list[DiscoveredDocument]:
        """Recursively find every supported document under the PDF root."""
        if not self._pdf_root.is_dir():
            raise FileNotFoundError(f"PDF root directory not found: {self._pdf_root}")

        found: list[DiscoveredDocument] = []
        seen_hashes: set[str] = set()
        for path in sorted(self._pdf_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self._extensions:
                continue
            try:
                doc_id = self._hash_file(path)
            except OSError as error:
                logger.warning("Could not read '%s' (%s); skipping.", path.name, error)
                continue

            if doc_id in seen_hashes:
                logger.info("Duplicate content skipped: %s", path.name)
                continue
            seen_hashes.add(doc_id)

            stat = path.stat()
            found.append(DiscoveredDocument(
                path=path,
                doc_id=doc_id,
                filename=path.name,
                category=path.parent.name,
                size_bytes=stat.st_size,
                modified_time=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            ))
        logger.info("Discovered %d unique document(s) under %s.", len(found), self._pdf_root)
        return found

    def is_new(self, doc: DiscoveredDocument) -> bool:
        """True if this document's content has not yet been ingested."""
        return doc.doc_id not in self._manifest

    def new_documents(self) -> list[DiscoveredDocument]:
        """Return only the discovered documents that are not yet in the manifest."""
        return [doc for doc in self.discover() if self.is_new(doc)]

    def record(self, doc: DiscoveredDocument, page_count: int, chunk_count: int) -> None:
        """Register a document as ingested (call after it is fully processed)."""
        self._manifest[doc.doc_id] = {
            "filename": doc.filename,
            "category": doc.category,
            "source_path": str(doc.path),
            "size_bytes": doc.size_bytes,
            "modified_time": doc.modified_time,
            "pages": page_count,
            "chunks": chunk_count,
            "ingested_at": datetime.now(tz=timezone.utc).isoformat(),
        }

    @property
    def manifest(self) -> dict[str, dict]:
        return self._manifest
