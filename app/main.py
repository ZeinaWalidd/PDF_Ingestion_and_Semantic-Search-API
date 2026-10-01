import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import (
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    MAX_FILES_PER_REQUEST,
    MAX_TOP_K,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.ingestion_service import (
    IngestionError,
    PreparedDocument,
    check_file_count,
    check_filename,
    check_size,
    prepare_document,
    read_directory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once, before the server accepts requests: loading takes
    # seconds and the model is reused by every ingest and search call.
    app.state.embedder = EmbeddingService(EMBEDDING_MODEL)
    app.state.store = VectorStore(
        url=QDRANT_URL,
        collection=QDRANT_COLLECTION,
        dimension=app.state.embedder.dimension,
    )
    yield


app = FastAPI(
    title="PDF Ingestor & Semantic Search API",
    version="1.0.0",
    lifespan=lifespan,
)

INTERNAL_ERROR_MESSAGES = {
    "/ingest/": "Failed to process uploaded file.",
    "/search/": "Search processing failed.",
}


# The Swagger spec returns errors as {"error": "..."}, while FastAPI's
# default is {"detail": ...}. These handlers make every error path match.

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # FastAPI answers malformed requests with 422; the spec only defines 400.
    messages = []
    for error in exc.errors():
        if error["type"] == "json_invalid":
            messages.append("Request body is not valid JSON.")
            continue
        if error["type"] == "value_error":
            messages.append(str(error["ctx"]["error"]))
            continue
        field = ".".join(str(part) for part in error["loc"] if part != "body")
        messages.append(f"{field}: {error['msg']}" if field else error["msg"])
    return JSONResponse(status_code=400, content={"error": "; ".join(messages)})


@app.exception_handler(IngestionError)
async def ingestion_error_handler(request: Request, exc: IngestionError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    message = INTERNAL_ERROR_MESSAGES.get(request.url.path, "Internal server error.")
    return JSONResponse(status_code=500, content={"error": message})



@app.get("/health")
def health(request: Request):
    if not request.app.state.store.is_ready():
        raise HTTPException(status_code=503, detail="Vector database is unavailable.")
    return {"status": "healthy"}


# /ingest/ reads the multipart form by hand because `input` may hold files
# or a directory path string, which a typed FastAPI parameter can't express.
# This schema keeps the endpoint usable from /docs.
MAX_DOCUMENT_ID_LENGTH = 200

INGEST_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["input"],
                    "properties": {
                        "input": {
                            "type": "array",
                            "items": {"type": "string", "format": "binary"},
                            "description": (
                                "One or more PDF files, or a directory path "
                                "inside the data volume (e.g. /data/pdfs)."
                            ),
                        },
                        "replace": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "Replace previously ingested versions of the same "
                                "document (same document_id, or same filename when "
                                "no document_id is given) instead of keeping both."
                            ),
                        },
                        "document_id": {
                            "type": "string",
                            "maxLength": MAX_DOCUMENT_ID_LENGTH,
                            "description": (
                                "Optional stable ID for a single uploaded document "
                                "(e.g. finance/report). Use it to tell apart "
                                "documents that share a filename."
                            ),
                        },
                    },
                }
            }
        },
    }
}


@app.post("/ingest/", openapi_extra=INGEST_OPENAPI)
async def ingest(request: Request):
    async with request.form(max_files=MAX_FILES_PER_REQUEST) as form:
        items = form.getlist("input")
        if not items:
            raise HTTPException(status_code=400, detail="input: Field required")
        replace = _parse_bool(form.get("replace"), field="replace")
        document_id = _parse_document_id(form.get("document_id"))
        files = await _collect_files(items)

    if document_id is not None and len(files) != 1:
        raise IngestionError("document_id can only be used when ingesting a single file.")
    if replace and document_id is None:
        _check_unique_filenames([filename for filename, _ in files])

    # Validate every file before storing any, so one bad file in a batch
    # fails the whole request instead of leaving a partial ingestion.
    documents = []
    for filename, file_bytes in files:
        # Parsing is CPU-bound; running it in a worker thread keeps the event
        # loop free to serve other requests (concurrent uploads, /health).
        documents.append(
            await run_in_threadpool(prepare_document, filename, file_bytes)
        )

    for document in documents:
        await run_in_threadpool(
            _embed_and_store, request.app, document, replace, document_id
        )

    ingested_files = [document.filename for document in documents]
    noun = "document" if len(ingested_files) == 1 else "documents"
    return {
        "message": f"Successfully ingested {len(ingested_files)} PDF {noun}.",
        "files": ingested_files,
    }


def _parse_bool(value: UploadFile | str | None, field: str) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower() if isinstance(value, str) else None
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no", ""):
        return False
    raise IngestionError(f"{field} must be true or false.")


def _parse_document_id(value: UploadFile | str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise IngestionError("document_id must be a non-empty string.")
    value = value.strip()
    if len(value) > MAX_DOCUMENT_ID_LENGTH:
        raise IngestionError(
            f"document_id must be at most {MAX_DOCUMENT_ID_LENGTH} characters."
        )
    return value


def _check_unique_filenames(filenames: list[str]) -> None:
    # With replace=true a filename identifies one document, so two different
    # files with the same name in one request would be ambiguous.
    seen = set()
    for filename in filenames:
        if filename in seen:
            raise IngestionError(
                f"Duplicate filename '{filename}': cannot replace by filename "
                "when one request contains it more than once."
            )
        seen.add(filename)


async def _collect_files(items: list[UploadFile | str]) -> list[tuple[str, bytes]]:
    """Turn the `input` form entries into (filename, bytes) pairs."""
    files = []
    for item in items:
        if isinstance(item, UploadFile):
            filename = item.filename or ""
            check_filename(filename)
            check_size(filename, item.size)
            files.append((filename, await item.read()))
        else:
            files.extend(await run_in_threadpool(read_directory, item))

    check_file_count(len(files))
    return files


def _embed_and_store(
    app: FastAPI, document: PreparedDocument, replace: bool, document_id: str | None
) -> None:
    started = time.perf_counter()
    embedder: EmbeddingService = app.state.embedder
    store: VectorStore = app.state.store

    vectors = embedder.embed_documents([chunk.content for chunk in document.chunks])
    removed = store.upsert_document(
        document.doc_id,
        document.filename,
        document.chunks,
        vectors,
        replace=replace,
        document_id=document_id,
    )
    logger.info(
        "%s: embedded and stored %d chunks in %.2fs (doc_id=%s)",
        document.filename,
        len(vectors),
        time.perf_counter() - started,
        document.doc_id[:12],
    )
    if removed:
        logger.info("%s: replaced %d chunks of older versions", document.filename, removed)


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


@app.post("/search/", response_model=SearchResponse)
def search(body: SearchRequest, request: Request):
    started = time.perf_counter()
    embedder: EmbeddingService = request.app.state.embedder
    store: VectorStore = request.app.state.store

    hits = store.search(
        embedder.embed_query(body.query), limit=body.top_k, min_score=body.min_score
    )
    logger.info(
        "Search %r returned %d results in %.2fs",
        body.query[:80], len(hits), time.perf_counter() - started,
    )
    return SearchResponse(
        results=[
            SearchResult(
                document=hit.document,
                document_id=hit.document_id,
                score=round(hit.score, 4),
                content=hit.content,
                page=hit.page,
                chunk_id=hit.chunk_id,
            )
            for hit in hits
        ]
    )