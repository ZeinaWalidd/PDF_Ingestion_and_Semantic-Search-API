import logging

import pymupdf

logger = logging.getLogger(__name__)

PDF_SIGNATURE = b"%PDF-"
SIGNATURE_SEARCH_WINDOW = 1024


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

        pages = []
        for page_number, page in enumerate(document, start=1):
            text = page.get_text().strip()
            if text:
                pages.append({"page": page_number, "text": text})
        return pages
    finally:
        document.close()


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
