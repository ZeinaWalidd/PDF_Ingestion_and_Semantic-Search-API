import logging
import time

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:

    def __init__(self, model_name: str, batch_size: int = 32):
        started = time.perf_counter()
        self._model = SentenceTransformer(model_name, device="cpu")
        self._batch_size = batch_size
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise RuntimeError(f"Model {model_name} does not report an embedding dimension.")
        self.dimension: int = dimension
        logger.info(
            "Loaded embedding model %s (dim=%d, max_tokens=%d) in %.1fs",
            model_name,
            self.dimension,
            self._model.max_seq_length,
            time.perf_counter() - started,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        # Normalized vectors have length 1, so cosine similarity is just a
        # dot product and scores land in a predictable [-1, 1] range.
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # turns NumPy array into Python lists to be accepted by the repository
        return vectors.tolist()
