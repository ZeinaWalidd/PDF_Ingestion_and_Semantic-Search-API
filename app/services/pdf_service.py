import logging
import re
from typing import cast

import pymupdf

logger = logging.getLogger(__name__)

PDF_SIGNATURE = b"%PDF-"
SIGNATURE_SEARCH_WINDOW = 1024
LINE_BREAK_HYPHEN = re.compile(r"(\w)-\n\s*([a-z])")


class PDFExtractionError(Exception):
    """Raised when a file is neither a readable PDF nor plain text."""


def extract_pages(file_bytes: bytes, source: str = "<upload>") -> list[dict]:

    if PDF_SIGNATURE in file_bytes[:SIGNATURE_SEARCH_WINDOW]:
        return _extract_pdf_pages(file_bytes)

    logger.warning("%s has no PDF signature; treating it as plain text.", source)
    return _extract_plain_text(file_bytes)


def _extract_pdf_pages(file_bytes: bytes) -> list[dict]:
    try:
        document = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFExtractionError("The file is a corrupt or unreadable PDF.") from exc

    try:
        if document.needs_pass:
            raise PDFExtractionError("Password-protected PDFs are not supported.")

        pages = []
        for index in range(document.page_count):
            raw = cast(str, document[index].get_text("text"))
            text = _join_hyphenated_words(raw).strip()
            if text:
                pages.append({"page": index + 1, "text": text})
        return pages
    finally:
        document.close()


def _join_hyphenated_words(text: str) -> str:
    return LINE_BREAK_HYPHEN.sub(r"\1\2", text)


def _extract_plain_text(file_bytes: bytes) -> list[dict]:
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PDFExtractionError("The file is not a valid PDF.") from exc

    if any(not (ch.isprintable() or ch.isspace()) for ch in text):
        raise PDFExtractionError("The file is not a valid PDF.")

    text = text.strip()
    if not text:
        return []
    return [{"page": 1, "text": text}]
