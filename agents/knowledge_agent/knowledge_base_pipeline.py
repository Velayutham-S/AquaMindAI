"""Knowledge Base (RAG) pipeline orchestrator for AquaMind AI.

Wires the offline components together:

    discover PDFs -> extract pages -> clean text -> chunk -> embed
                  -> append to FAISS -> append chunk metadata -> update manifest

It supports two workflows with the same code path:

  * ``build()``            -- process every not-yet-ingested PDF under the root
                              (initial full build, or catch-up on new files).
  * ``ingest_pdf(path)``   -- process a single PDF (the future Streamlit admin
                              upload). Only that document is indexed; the rest of
                              the knowledge base is left untouched.

Both are incremental: existing vectors/metadata are appended to, never rebuilt.
After each document the FAISS index, metadata and manifest are saved together so
the knowledge base stays consistent even if a later document fails.

This module builds offline infrastructure only. It performs no retrieval, calls
no LLM, and answers no questions.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# --- wire up sibling component packages (no __init__.py; import by location) ---
_BASE = Path(__file__).resolve().parent
for _sub in ("", "ingestion", "preprocessing", "chunking", "embedding", "vector_store", "metadata"):
    _path = _BASE / _sub if _sub else _BASE
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import knowledge_config as kb_config  # noqa: E402  (unique module name; avoids sys.modules 'config' collision)
from document_ingestor import DocumentIngestor, DiscoveredDocument  # noqa: E402
from pdf_extractor import extract_pages, PdfExtractionError  # noqa: E402
from text_cleaner import TextCleaner  # noqa: E402
from text_chunker import TextChunker  # noqa: E402
from embedder import Embedder, EmbeddingError  # noqa: E402
from faiss_store import FaissVectorStore, VectorStoreError  # noqa: E402
from metadata_store import MetadataStore  # noqa: E402

logger = logging.getLogger("aquamind.knowledge.pipeline")


@dataclass
class BuildResult:
    """Summary statistics for a build/ingest run."""

    discovered_pdfs: int = 0
    processed_pdfs: int = 0
    skipped_pdfs: int = 0
    failed_pdfs: int = 0
    pages_processed: int = 0
    chunks_created: int = 0
    embedding_model: str = ""
    embedding_dimension: int = 0
    vector_count: int = 0
    metadata_entries: int = 0
    faiss_index_bytes: int = 0
    processing_seconds: float = 0.0


class KnowledgePipeline:
    """Builds and incrementally maintains the FAISS-backed knowledge base."""

    def __init__(self, config: kb_config.KnowledgeBaseConfig = kb_config.CONFIG) -> None:
        self._config = config
        self._ingestor = DocumentIngestor(
            pdf_root=config.pdf_root,
            manifest_path=config.manifest_path,
            extensions=config.pdf_extensions,
        )
        self._cleaner = TextCleaner()
        self._chunker = TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_chunk_chars=config.min_chunk_chars,
        )
        self._embedder = Embedder(
            model_name=config.embedding_model,
            batch_size=config.embedding_batch_size,
            normalize=config.normalize_embeddings,
        )
        self._vector_store = FaissVectorStore(config.faiss_index_path)
        self._metadata = MetadataStore(config.metadata_path)

        self._vector_store.load()
        self._check_alignment()

    def _check_alignment(self) -> None:
        if self._vector_store.count != self._metadata.count:
            logger.warning(
                "FAISS vectors (%d) and metadata records (%d) are out of sync; "
                "new documents will still append correctly but existing state may be inconsistent.",
                self._vector_store.count, self._metadata.count,
            )

    # -- public workflows ------------------------------------------------- #

    def build(self) -> BuildResult:
        """Process every PDF under the root that has not yet been ingested."""
        started = time.perf_counter()
        result = BuildResult(embedding_model=self._config.embedding_model)

        documents = self._ingestor.discover()
        result.discovered_pdfs = len(documents)
        pending = [doc for doc in documents if self._ingestor.is_new(doc)]
        result.skipped_pdfs = len(documents) - len(pending)
        logger.info("%d document(s) to process; %d already ingested.", len(pending), result.skipped_pdfs)

        for doc in pending:
            self._ingest_document(doc, result)

        self._finalize_result(result, started)
        return result

    def ingest_pdf(self, pdf_path: str | Path) -> BuildResult:
        """Process a single PDF (e.g. a Streamlit admin upload) incrementally.

        The document is added only if its content is new; duplicates are skipped.
        The rest of the knowledge base is untouched.
        """
        started = time.perf_counter()
        result = BuildResult(embedding_model=self._config.embedding_model)

        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")

        stat = path.stat()
        doc = DiscoveredDocument(
            path=path,
            doc_id=DocumentIngestor._hash_file(path),
            filename=path.name,
            category=path.parent.name,
            size_bytes=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        )
        result.discovered_pdfs = 1
        if not self._ingestor.is_new(doc):
            logger.info("Document already ingested (duplicate content); skipping: %s", doc.filename)
            result.skipped_pdfs = 1
        else:
            self._ingest_document(doc, result)

        self._finalize_result(result, started)
        return result

    # -- per-document processing ------------------------------------------ #

    def _ingest_document(self, doc: DiscoveredDocument, result: BuildResult) -> None:
        logger.info("Processing: %s (%s)", doc.filename, doc.category)
        try:
            pages = extract_pages(doc.path)
        except PdfExtractionError as error:
            logger.error("Skipping corrupt PDF '%s': %s", doc.filename, error)
            result.failed_pdfs += 1
            return

        chunk_texts, records = self._build_chunks(doc, pages)
        if not chunk_texts:
            logger.warning("No usable text in '%s'; recording with 0 chunks.", doc.filename)
            self._ingestor.record(doc, page_count=len(pages), chunk_count=0)
            self._ingestor.save_manifest()
            result.processed_pdfs += 1
            result.pages_processed += len(pages)
            return

        try:
            vectors = self._embedder.embed(chunk_texts)
            embedding_ids = self._vector_store.add(vectors)
        except (EmbeddingError, VectorStoreError) as error:
            logger.error("Embedding/index failure for '%s': %s", doc.filename, error)
            result.failed_pdfs += 1
            return

        for record, embedding_id in zip(records, embedding_ids):
            record["embedding_id"] = embedding_id
        self._metadata.append(records)

        # Persist all three artifacts together to keep the knowledge base consistent.
        self._vector_store.save()
        self._metadata.save()
        self._ingestor.record(doc, page_count=len(pages), chunk_count=len(records))
        self._ingestor.save_manifest()

        result.processed_pdfs += 1
        result.pages_processed += len(pages)
        result.chunks_created += len(records)
        logger.info("Indexed '%s': %d page(s), %d chunk(s).", doc.filename, len(pages), len(records))

    def _build_chunks(self, doc: DiscoveredDocument, pages: list[str]) -> tuple[list[str], list[dict]]:
        """Clean and chunk each page, producing aligned (text, metadata) lists."""
        chunk_texts: list[str] = []
        records: list[dict] = []
        chunk_index = 0
        try:
            source_path = str(doc.path.relative_to(self._config.pdf_root.parent).as_posix())
        except ValueError:
            source_path = str(doc.path.as_posix())

        for page_number, raw_text in enumerate(pages, start=1):
            cleaned = self._cleaner.clean(raw_text)
            if not cleaned:
                continue
            for chunk in self._chunker.chunk(cleaned):
                chunk_texts.append(chunk)
                records.append({
                    "chunk_id": f"{doc.doc_id[:12]}_{chunk_index:05d}",
                    "embedding_id": None,  # assigned after FAISS insertion
                    "document": doc.filename,
                    "category": doc.category,
                    "source_path": source_path,
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "section": self._section_hint(chunk),
                    "char_count": len(chunk),
                    "doc_id": doc.doc_id,
                })
                chunk_index += 1
        return chunk_texts, records

    @staticmethod
    def _section_hint(chunk: str) -> str | None:
        """Best-effort heading hint: the chunk's first non-empty line, truncated."""
        for line in chunk.splitlines():
            line = line.strip()
            if line:
                return line[:100]
        return None

    # -- result finalization ---------------------------------------------- #

    def _finalize_result(self, result: BuildResult, started: float) -> None:
        result.vector_count = self._vector_store.count
        result.metadata_entries = self._metadata.count
        result.embedding_dimension = self._vector_store.dimension
        if self._config.faiss_index_path.exists():
            result.faiss_index_bytes = self._config.faiss_index_path.stat().st_size
        result.processing_seconds = time.perf_counter() - started


def _print_report(result: BuildResult) -> None:
    mb = result.faiss_index_bytes / (1024 * 1024)
    print("=" * 60)
    print("KNOWLEDGE BASE BUILD REPORT")
    print("=" * 60)
    print(f"Total PDFs (discovered)   : {result.discovered_pdfs}")
    print(f"Processed this run        : {result.processed_pdfs}")
    print(f"Skipped (already indexed) : {result.skipped_pdfs}")
    print(f"Failed (corrupt/empty)    : {result.failed_pdfs}")
    print(f"Total Pages (this run)    : {result.pages_processed}")
    print(f"Total Chunks (this run)   : {result.chunks_created}")
    print(f"Embedding Model           : {result.embedding_model}")
    print(f"Embedding Dimension       : {result.embedding_dimension}")
    print(f"Vector Count (total)      : {result.vector_count}")
    print(f"FAISS Index Size          : {mb:.2f} MB ({result.faiss_index_bytes} bytes)")
    print(f"Metadata Entries (total)  : {result.metadata_entries}")
    print(f"Processing Time           : {result.processing_seconds:.1f}s")
    print("=" * 60)


def main() -> int:
    kb_config.configure_logging()
    pipeline = KnowledgePipeline()
    result = pipeline.build()
    _print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
