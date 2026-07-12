"""PDF text extraction: turn a PDF file into per-page text.

Uses ``pypdfium2`` for fast, robust text extraction. Corrupt PDFs raise
``PdfExtractionError``; empty pages yield empty strings (the caller decides how
to handle them). This component does not clean or chunk text.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pypdfium2 as pdfium

logger = logging.getLogger("aquamind.knowledge.extraction")


class PdfExtractionError(Exception):
    """The PDF could not be opened or read."""


def extract_pages(pdf_path: Path) -> list[str]:
    """Return the text of each page in ``pdf_path`` as a list of strings.

    The list length equals the page count; a page with no extractable text
    contributes an empty string (so page numbers stay aligned).

    Raises:
        PdfExtractionError: if the PDF cannot be opened or parsed.
    """
    try:
        document = pdfium.PdfDocument(str(pdf_path))
    except Exception as error:  # noqa: BLE001 - pdfium raises assorted errors
        raise PdfExtractionError(f"Cannot open PDF '{pdf_path.name}': {error}") from error

    pages: list[str] = []
    try:
        for page_index in range(len(document)):
            try:
                page = document[page_index]
                text_page = page.get_textpage()
                pages.append(text_page.get_text_range() or "")
                text_page.close()
                page.close()
            except Exception as error:  # noqa: BLE001 - skip a bad page, keep the rest
                logger.warning("Failed to read page %d of '%s' (%s); using empty text.",
                               page_index + 1, pdf_path.name, error)
                pages.append("")
    finally:
        document.close()
    return pages
