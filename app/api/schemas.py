from pydantic import BaseModel, Field, field_validator

from app.config import DEFAULT_MIN_SCORE, DEFAULT_TOP_K, MAX_TOP_K


class SearchRequest(BaseModel):
    query: str = Field(..., max_length=1000, examples=["Explain how vector embeddings work."])
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    min_score: float = Field(DEFAULT_MIN_SCORE, ge=-1.0, le=1.0)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query cannot be empty.")
        return value


class SearchResult(BaseModel):
    document: str
    document_id: str | None = None
    score: float
    content: str
    page: int
    chunk_id: int


class SearchResponse(BaseModel):
    results: list[SearchResult]


class IngestResponse(BaseModel):
    message: str
    files: list[str]
