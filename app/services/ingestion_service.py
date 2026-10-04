import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from app.services.chunking_service import Chunk
from app.services.ports import (
    Chunker,
    ChunkRepository,
    Embedder,
    ExtractionError,
    TextExtractor,
)

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """A problem with the client's input; the message is safe to return as a 400."""


@dataclass
class PreparedDocument:
    doc_id: str
    filename: str
    chunks: list[Chunk]


def check_filename(filename: str) -> None:
    if not filename.lower().endswith(".pdf"):
        raise IngestionError("Only PDF files are accepted.")


def check_size(filename: str, size: int | None, max_size_mb: int) -> None:
    if size is not None and size > max_size_mb * 1024 * 1024:
        raise IngestionError(
            f"File '{filename}' exceeds the {max_size_mb} MB limit."
        )


def check_file_count(count: int, max_files: int) -> None:
    if count > max_files:
        raise IngestionError(
            f"Too many files: at most {max_files} PDFs per request."
        )

# creates a deterministic fingerprint of extracted document content (SHA-256)
def _content_hash(pages: list[dict]) -> str:
    digest = hashlib.sha256()
    for page in pages:
        digest.update(f"{page['page']}\x00{page['text']}\x00".encode("utf-8"))
    return digest.hexdigest()


def read_directory(
    path: str, data_dir: Path, max_size_mb: int, max_files: int
) -> list[tuple[str, bytes]]:
    """Return (filename, bytes) for every PDF directly inside `path`, which must be under `data_dir`."""

    path = path.strip()
    if not path:
        raise IngestionError("Directory path cannot be empty.")

    root = data_dir.resolve()
    directory = (root / path).resolve()

    if not directory.is_relative_to(root):
        raise IngestionError(f"Directory must be inside {data_dir}.")
    if not directory.is_dir():
        raise IngestionError(f"Directory '{path}' does not exist.")

    pdf_paths = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    )
    if not pdf_paths:
        raise IngestionError(f"No PDF files found in '{path}'.")
    check_file_count(len(pdf_paths), max_files)

    files = []
    for pdf_path in pdf_paths:
        check_size(pdf_path.name, pdf_path.stat().st_size, max_size_mb)
        files.append((pdf_path.name, pdf_path.read_bytes()))

    logger.info("Found %d PDF(s) in %s", len(files), directory)
    return files


class IngestionService:

    def __init__(
        self,
        extractor: TextExtractor,
        chunker: Chunker,
        embedder: Embedder,
        repository: ChunkRepository,
    ):
        self._extractor = extractor
        self._chunker = chunker
        self._embedder = embedder
        self._repository = repository

    def ingest(self, files: list[tuple[str, bytes]]) -> list[str]:
        documents = [self._prepare(filename, data) for filename, data in files]
        for document in documents:
            self._embed_and_store(document)
        return [document.filename for document in documents]

    def _prepare(self, filename: str, file_bytes: bytes) -> PreparedDocument:
        if not file_bytes:
            raise IngestionError(f"File '{filename}' is empty.")

        try:
            pages = self._extractor.extract_pages(file_bytes, source=filename)
        except ExtractionError as exc:
            raise IngestionError(f"Failed to process '{filename}': {exc}") from exc

        if not pages:
            raise IngestionError(f"No text could be extracted from '{filename}'.")

        chunks = self._chunker.chunk(pages, document=filename)
        logger.info(
            "%s: extracted %d pages and %d chunks", filename, len(pages), len(chunks)
        )
        return PreparedDocument(
            doc_id=_content_hash(pages),
            filename=filename,
            chunks=chunks,
        )

    def _embed_and_store(self, document: PreparedDocument) -> None:
        started = time.perf_counter()
        vectors = self._embedder.embed_documents(
            [chunk.content for chunk in document.chunks]
        )
        self._repository.upsert_document(document.doc_id, document.chunks, vectors)
        logger.info(
            "%s: embedded and stored %d chunks in %.2fs (doc_id=%s)",
            document.filename,
            len(vectors),
            time.perf_counter() - started,
            document.doc_id[:12],
        )
