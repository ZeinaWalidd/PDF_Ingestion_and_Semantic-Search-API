import hashlib
import logging
from dataclasses import dataclass

from app.config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    DATA_DIR,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MAX_FILES_PER_REQUEST,
)
from app.services.chunking_service import Chunk, chunk_pages
from app.services.pdf_service import PDFExtractionError, extract_pages

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """A problem with the client's input; the message is safe to return as a 400."""


@dataclass
class PreparedDocument:
    doc_id: str
    filename: str
    pages: list[dict]
    chunks: list[Chunk]


def check_filename(filename: str) -> None:
    if not filename.lower().endswith(".pdf"):
        raise IngestionError("Only PDF files are accepted.")


def check_size(filename: str, size: int | None) -> None:
    if size is not None and size > MAX_FILE_SIZE_BYTES:
        raise IngestionError(
            f"File '{filename}' exceeds the {MAX_FILE_SIZE_MB} MB limit."
        )


def check_file_count(count: int) -> None:
    if count > MAX_FILES_PER_REQUEST:
        raise IngestionError(
            f"Too many files: at most {MAX_FILES_PER_REQUEST} PDFs per request."
        )


def _content_hash(pages: list[dict]) -> str:
    digest = hashlib.sha256()
    for page in pages:
        digest.update(f"{page['page']}\x00{page['text']}\x00".encode("utf-8"))
    return digest.hexdigest()


def prepare_document(filename: str, file_bytes: bytes) -> PreparedDocument:
    """Validate, extract and chunk one file. Raises IngestionError on bad input.

    This is CPU-bound, so callers in async code should run it in a thread.
    """
    if not file_bytes:
        raise IngestionError(f"File '{filename}' is empty.")

    try:
        pages = extract_pages(file_bytes, source=filename)
    except PDFExtractionError as exc:
        raise IngestionError(f"Failed to process '{filename}': {exc}") from exc

    if not pages:
        raise IngestionError(f"No text could be extracted from '{filename}'.")

    chunks = chunk_pages(
        pages=pages,
        document=filename,
        chunk_size=CHUNK_SIZE_WORDS,
        overlap=CHUNK_OVERLAP_WORDS,
    )
    logger.info(
        "%s: extracted %d pages and %d chunks", filename, len(pages), len(chunks)
    )
    return PreparedDocument(
        # Hash the extracted text, not the raw bytes: renamed or re-saved copies
        # of the same document get the same doc_id, so they're stored once.
        doc_id=_content_hash(pages),
        filename=filename,
        pages=pages,
        chunks=chunks,
    )


def read_directory(path: str) -> list[tuple[str, bytes]]:
    """Return (filename, bytes) for every PDF directly inside `path`. """
    
    path = path.strip()
    if not path:
        raise IngestionError("Directory path cannot be empty.")

    root = DATA_DIR.resolve()
    directory = (root / path).resolve()

    if not directory.is_relative_to(root):
        raise IngestionError(f"Directory must be inside {DATA_DIR}.")
    if not directory.is_dir():
        raise IngestionError(f"Directory '{path}' does not exist.")

    pdf_paths = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    )
    if not pdf_paths:
        raise IngestionError(f"No PDF files found in '{path}'.")
    check_file_count(len(pdf_paths))

    files = []
    for pdf_path in pdf_paths:
        check_size(pdf_path.name, pdf_path.stat().st_size)
        files.append((pdf_path.name, pdf_path.read_bytes()))

    logger.info("Found %d PDF(s) in %s", len(files), directory)
    return files
