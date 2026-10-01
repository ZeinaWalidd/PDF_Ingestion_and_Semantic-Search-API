import logging
import time

from app.services.ports import ChunkRepository, Embedder, SearchHit

logger = logging.getLogger(__name__)


class SearchService:

    def __init__(self, embedder: Embedder, repository: ChunkRepository):
        self._embedder = embedder
        self._repository = repository

    def search(self, query: str, top_k: int, min_score: float) -> list[SearchHit]:
        started = time.perf_counter()
        hits = self._repository.search(
            self._embedder.embed_query(query), limit=top_k, min_score=min_score
        )
        logger.info(
            "Search %r returned %d results in %.2fs",
            query[:80], len(hits), time.perf_counter() - started,
        )
        return hits
