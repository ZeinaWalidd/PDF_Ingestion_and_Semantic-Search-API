import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import EMBEDDING_MODEL, MAX_FILES_PER_REQUEST
from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import (
    IngestionError,
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
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once, before the server accepts requests: loading takes
    # seconds and the model is reused by every ingest and search call.
    app.state.embedder = EmbeddingService(EMBEDDING_MODEL)
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
def health():
    return {"status": "healthy"}


# /ingest/ reads the multipart form by hand because `input` may hold files
# or a directory path string, which a typed FastAPI parameter can't express.
# This schema keeps the endpoint usable from /docs.
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
                        }
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
        files = await _collect_files(items)

    # Validate every file before storing any, so one bad file in a batch
    # fails the whole request instead of leaving a partial ingestion.
    documents = []
    for filename, file_bytes in files:
        # Parsing is CPU-bound; running it in a worker thread keeps the event
        # loop free to serve other requests (concurrent uploads, /health).
        documents.append(
            await run_in_threadpool(prepare_document, filename, file_bytes)
        )

    embedder: EmbeddingService = request.app.state.embedder
    for document in documents:
        started = time.perf_counter()
        vectors = await run_in_threadpool(
            embedder.embed_documents, [chunk.content for chunk in document.chunks]
        )
        logger.info(
            "%s: embedded %d chunks in %.2fs",
            document.filename, len(vectors), time.perf_counter() - started,
        )

    ingested_files = [document.filename for document in documents]
    noun = "document" if len(ingested_files) == 1 else "documents"
    return {
        "message": f"Successfully ingested {len(ingested_files)} PDF {noun}.",
        "files": ingested_files,
    }


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


@app.post("/search/")
def search():
    return {
        "results": [],
    }