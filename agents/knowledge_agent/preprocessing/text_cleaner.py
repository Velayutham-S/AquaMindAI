"""Text preprocessing: clean raw PDF-extracted text for chunking.

Responsibilities (single): normalize whitespace, repair common PDF extraction
artifacts (hyphenated line breaks, stray control characters), and collapse
excessive blank lines -- while preserving paragraph structure that is useful for
downstream chunking. It performs no chunking, embedding, or semantic changes.
"""

from __future__ import annotations

import re
import unicodedata

# Precompiled patterns (module-level so they compile once).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")          # join words split across lines
_SPACES = re.compile(r"[ \t\u00a0]+")                   # runs of spaces/tabs/nbsp
_SPACE_BEFORE_NEWLINE = re.compile(r"[ \t]+\n")
_MULTI_NEWLINE = re.compile(r"\n{3,}")                   # 3+ newlines -> paragraph break


class TextCleaner:
    """Cleans and normalizes raw extracted text."""

    def clean(self, text: str) -> str:
        """Return a cleaned, whitespace-normalized version of ``text``.

        Returns an empty string for empty/whitespace-only input.
        """
        if not text or not text.strip():
            return ""

        # Normalize unicode (e.g. ligatures, full-width chars) to a canonical form.
        text = unicodedata.normalize("NFKC", text)
        # Standardize line endings.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Rejoin words hyphenated across a line break: "ground-\nwater" -> "groundwater".
        text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
        # Drop non-printable control characters.
        text = _CONTROL_CHARS.sub("", text)
        # Collapse horizontal whitespace and trailing spaces before newlines.
        text = _SPACES.sub(" ", text)
        text = _SPACE_BEFORE_NEWLINE.sub("\n", text)
        # Collapse 3+ newlines into a single paragraph break.
        text = _MULTI_NEWLINE.sub("\n\n", text)
        return text.strip()
