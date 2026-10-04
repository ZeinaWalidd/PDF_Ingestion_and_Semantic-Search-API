from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    data_dir: Path = Path("/data")

    max_file_size_mb: int = Field(50, gt=0)
    max_files_per_request: int = Field(20, gt=0)

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_words: int = Field(150, gt=0)
    chunk_overlap_words: int = Field(30, ge=0)

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "pdf_chunks"
    default_top_k: int = Field(5, ge=1)
    max_top_k: int = Field(50, ge=1)
    default_min_score: float = Field(0.15, ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError("CHUNK_OVERLAP_WORDS must be smaller than CHUNK_SIZE_WORDS.")
        if self.default_top_k > self.max_top_k:
            raise ValueError("DEFAULT_TOP_K cannot be larger than MAX_TOP_K.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
