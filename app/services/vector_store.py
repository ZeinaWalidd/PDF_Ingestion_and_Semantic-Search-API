import logging
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from app.services.chunking_service import Chunk

logger = logging.getLogger(__name__)

POINT_ID_NAMESPACE = uuid.UUID("6f1c2a8e-3b7d-4e59-9a0c-5d2f8b1e7c43")


@dataclass
class SearchHit:
    document: str
    page: int
    chunk_id: int
    content: str
    score: float


class VectorStore:

    def __init__(self, url: str, collection: str, dimension: int):
        self._client = QdrantClient(url=url, timeout=30)
        self._collection = collection
        self._ensure_collection(dimension)

    def _ensure_collection(self, dimension: int) -> None:
        if self._client.collection_exists(self._collection):
            info = self._client.get_collection(self._collection)
            vectors = info.config.params.vectors
            existing = vectors.size if isinstance(vectors, models.VectorParams) else None
            if existing != dimension:
                raise RuntimeError(
                    f"Collection '{self._collection}' stores {existing}-dim vectors "
                    f"but the embedding model produces {dimension}-dim vectors."
                )
            logger.info("Using existing collection '%s'", self._collection)
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE
            ),
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="doc_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        logger.info("Created collection '%s' (dim=%d)", self._collection, dimension)

    def is_ready(self) -> bool:
        try:
            self._client.collection_exists(self._collection)
            return True
        except Exception:
            return False

    def upsert_document(
        self, doc_id: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        # IDs are derived from (content hash, chunk index), so re-ingesting the
        # same file overwrites its points instead of duplicating them.
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(POINT_ID_NAMESPACE, f"{doc_id}:{chunk.chunk_id}")),
                vector=vector,
                payload={
                    "doc_id": doc_id,
                    "document": chunk.document,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(
            collection_name=self._collection, points=points, wait=True
        )

        # If the same file was ingested earlier with different chunk settings,
        # drop its leftover higher-index chunks.
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id", match=models.MatchValue(value=doc_id)
                    ),
                    models.FieldCondition(
                        key="chunk_id", range=models.Range(gte=len(chunks))
                    ),
                ]
            ),
            wait=True,
        )

    def search(self, vector: list[float], limit: int) -> list[SearchHit]:
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        hits = []
        for point in response.points:
            payload = point.payload
            if payload is None:
                logger.warning("Point %s has no payload; skipping", point.id)
                continue
            hits.append(
                SearchHit(
                    document=payload["document"],
                    page=payload["page"],
                    chunk_id=payload["chunk_id"],
                    content=payload["content"],
                    score=point.score,
                )
            )
        return hits
