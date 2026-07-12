"""Chunking: split cleaned text into overlapping, size-bounded chunks.

Responsibilities (single): produce character-bounded chunks with a configurable
size and overlap, preferring to break on paragraph/sentence/word boundaries so
chunks stay readable. Chunks shorter than ``min_chunk_chars`` are discarded.

This component is metadata-agnostic: it returns plain chunk strings. The
orchestrator attaches page/document metadata to each chunk.
"""

from __future__ import annotations


class TextChunker:
    """Splits text into overlapping character-bounded chunks."""

    def __init__(self, chunk_size: int, chunk_overlap: int, min_chunk_chars: int = 0) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")
        self._chunk_size = chunk_size
        self._overlap = chunk_overlap
        self._min_chunk_chars = min_chunk_chars

    def chunk(self, text: str) -> list[str]:
        """Return the list of chunks for ``text`` (empty list if nothing usable)."""
        text = (text or "").strip()
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        length = len(text)
        step = self._chunk_size - self._overlap

        while start < length:
            end = min(start + self._chunk_size, length)
            if end < length:
                end = self._preferred_break(text, start, end)
            chunk = text[start:end].strip()
            if len(chunk) >= self._min_chunk_chars:
                chunks.append(chunk)
            if end >= length:
                break
            start = max(end - self._overlap, start + step)
        return chunks

    def _preferred_break(self, text: str, start: int, hard_end: int) -> int:
        """Move the cut back to the nearest paragraph/sentence/space boundary.

        Only searches within the latter part of the window so chunks do not
        shrink drastically; falls back to the hard boundary if none is found.
        """
        window_floor = start + (self._chunk_size // 2)
        for separator in ("\n\n", ". ", ".\n", "\n", " "):
            boundary = text.rfind(separator, window_floor, hard_end)
            if boundary != -1:
                return boundary + len(separator)
        return hard_end
