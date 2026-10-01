from dataclasses import dataclass
from typing import Protocol

from app.services.chunking_service import Chunk


@dataclass
class SearchHit:
    document: str
    document_id: str | None
    page: int
    chunk_id: int
    content: str
    score: float


class ExtractionError(Exception):
    """Raised by a TextExtractor when a file can't be read; the message is shown to the client."""


class TextExtractor(Protocol):
    def extract_pages(self, file_bytes: bytes, source: str) -> list[dict]: ...


class Chunker(Protocol):
    def chunk(self, pages: list[dict], document: str) -> list[Chunk]: ...


class Embedder(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class ChunkRepository(Protocol):
    def is_ready(self) -> bool: ...

    def upsert_document(
        self,
        doc_id: str,
        filename: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
        replace: bool = False,
        document_id: str | None = None,
    ) -> int: ...

    def search(
        self, vector: list[float], limit: int, min_score: float
    ) -> list[SearchHit]: ...
