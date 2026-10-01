import logging
import threading
import uuid

from qdrant_client import QdrantClient, models

from app.services.chunking_service import Chunk
from app.services.ports import SearchHit

logger = logging.getLogger(__name__)

POINT_ID_NAMESPACE = uuid.UUID("6f1c2a8e-3b7d-4e59-9a0c-5d2f8b1e7c43")
SHARED_SCOPE = "shared"


def _document_key(filename: str, document_id: str | None) -> str:

    return f"id:{document_id}" if document_id else f"name:{filename}"


def _point_id(scope: str, doc_id: str, chunk_id: int) -> str:
    seed = f"{doc_id}:{chunk_id}" if scope == SHARED_SCOPE else f"{scope}:{doc_id}:{chunk_id}"
    return str(uuid.uuid5(POINT_ID_NAMESPACE, seed))


def _match(field: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=field, match=models.MatchValue(value=value))


class VectorStore:

    def __init__(self, url: str, collection: str, dimension: int):
        self._client = QdrantClient(url=url, timeout=30)
        self._collection = collection
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._ensure_collection(dimension)
        self._ensure_payload_indexes()

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
        logger.info("Created collection '%s' (dim=%d)", self._collection, dimension)

    def _ensure_payload_indexes(self) -> None:

        for field in ("doc_id", "scope", "document_key"):
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

    def is_ready(self) -> bool:
        try:
            self._client.collection_exists(self._collection)
            return True
        except Exception:
            return False

    def upsert_document(
        self,
        doc_id: str,
        filename: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
        replace: bool = False,
        document_id: str | None = None,
    ) -> int:

        key = _document_key(filename, document_id)

        scope = key if document_id else SHARED_SCOPE

        with self._lock_for(key):
            self._write_points(doc_id, scope, key, document_id, chunks, vectors)
            return self._delete_other_versions(doc_id, key) if replace else 0

    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _delete_other_versions(self, doc_id: str, key: str) -> int:
        old_versions = models.Filter(
            must=[_match("document_key", key)],
            must_not=[_match("doc_id", doc_id)],
        )
        removed = self._client.count(
            collection_name=self._collection, count_filter=old_versions, exact=True
        ).count
        if removed:
            self._client.delete(
                collection_name=self._collection,
                points_selector=old_versions,
                wait=True,
            )
        return removed

    def _write_points(
        self,
        doc_id: str,
        scope: str,
        key: str,
        document_id: str | None,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        # IDs are derived from (scope, content hash, chunk index), so
        # re-ingesting the same document overwrites its points instead of
        # duplicating them.
        points = [
            models.PointStruct(
                id=_point_id(scope, doc_id, chunk.chunk_id),
                vector=vector,
                payload={
                    "doc_id": doc_id,
                    "scope": scope,
                    "document_key": key,
                    "document_id": document_id,
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

        # If the same document was ingested earlier with different chunk
        # settings, drop its leftover higher-index chunks.
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.Filter(
                must=[
                    _match("doc_id", doc_id),
                    _match("scope", scope),
                    models.FieldCondition(
                        key="chunk_id", range=models.Range(gte=len(chunks))
                    ),
                ]
            ),
            wait=True,
        )

    def search(
        self, vector: list[float], limit: int, min_score: float
    ) -> list[SearchHit]:
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=limit,
            score_threshold=min_score,
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
                    document_id=payload.get("document_id"),
                    page=payload["page"],
                    chunk_id=payload["chunk_id"],
                    content=payload["content"],
                    score=point.score,
                )
            )
        return hits
