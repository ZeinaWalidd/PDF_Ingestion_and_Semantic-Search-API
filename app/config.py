import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_FILES_PER_REQUEST = int(os.getenv("MAX_FILES_PER_REQUEST", "20"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "150"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "30"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_chunks")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "50"))
# Calibrated for all-MiniLM-L6-v2 on the sample data: off-topic queries peaked
# at 0.13, on-topic ones started at 0.17.
DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "0.15"))
