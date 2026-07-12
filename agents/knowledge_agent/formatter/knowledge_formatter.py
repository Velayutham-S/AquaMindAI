"""Knowledge Formatter for the AquaMind AI Knowledge Agent.

Single responsibility: convert the Retrieval Coordinator's output
(``list[RetrievedChunk]``) into a structured, deterministic knowledge-agent
response envelope -- the Knowledge Agent counterpart of the Data Agent's
evidence-based response.

This is a **pure transformation layer**. It does NOT call an LLM, summarize,
generate answers, rank, reorder, filter, deduplicate, aggregate, or modify chunk
text. Chunk order and chunk text are preserved exactly as retrieved. It has no
side effects.

Public interface:
    KnowledgeFormatter().format(chunks) -> dict
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # import only for typing; the formatter is duck-typed at runtime
    from metadata_resolver import RetrievedChunk


class KnowledgeFormatter:
    """Transforms retrieved chunks into a structured knowledge-agent response."""

    AGENT_NAME: str = "knowledge_agent"
    QUERY_TYPE: str = "knowledge"
    RETRIEVAL_METHOD: str = "semantic_search"
    STATUS_SUCCESS: str = "SUCCESS"
    STATUS_NO_RESULTS: str = "NO_RESULTS"

    def format(self, chunks: list["RetrievedChunk"]) -> dict[str, Any]:
        """Return a structured response envelope for the retrieved chunks.

        Every chunk field is preserved as one evidence item, in retrieval order.
        An empty input yields ``status = NO_RESULTS`` with ``total_evidence = 0``
        and ``evidence = []``. This method never raises for empty results.
        """
        evidence = [self._format_chunk(chunk) for chunk in chunks]
        status = self.STATUS_SUCCESS if evidence else self.STATUS_NO_RESULTS
        return {
            "agent_name": self.AGENT_NAME,
            "status": status,
            "query_type": self.QUERY_TYPE,
            "total_evidence": len(evidence),
            "retrieval_method": self.RETRIEVAL_METHOD,
            "evidence": evidence,
        }

    @staticmethod
    def _format_chunk(chunk: "RetrievedChunk") -> dict[str, Any]:
        """Map one ``RetrievedChunk`` to an evidence item, preserving every field.

        Chunk text is copied verbatim into ``content``; nothing is altered.
        """
        return {
            "chunk_id": chunk.chunk_id,
            "document": chunk.document,
            "category": chunk.category,
            "page": chunk.page,
            "section": chunk.section,
            "source_path": chunk.source_path,
            "similarity_score": chunk.score,
            "content": chunk.text,
        }
