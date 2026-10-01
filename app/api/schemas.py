from pydantic import BaseModel, Field, field_validator

from app.config import get_settings

_settings = get_settings()


class SearchRequest(BaseModel):
    query: str = Field(..., max_length=1000, examples=["Explain how vector embeddings work."])
    top_k: int = Field(_settings.default_top_k, ge=1, le=_settings.max_top_k)
    min_score: float = Field(_settings.default_min_score, ge=-1.0, le=1.0)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query cannot be empty.")
        return value


class SearchResult(BaseModel):
    document: str
    score: float
    content: str
    page: int
    chunk_id: int


class SearchResponse(BaseModel):
    results: list[SearchResult]


class IngestResponse(BaseModel):
    message: str
    files: list[str]
