import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import router
from app.config import Settings, get_settings
from app.services.chunking_service import SentenceChunker
from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionError, IngestionService
from app.services.pdf_service import PyMuPDFExtractor
from app.services.search_service import SearchService
from app.services.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
# Don't show httpx logs below WARNING.
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

INTERNAL_ERROR_MESSAGES = {
    "/ingest/": "Failed to process uploaded file.",
    "/search/": "Search processing failed.",
}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):

        embedder = EmbeddingService(settings.embedding_model)
        repository = VectorStore(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            dimension=embedder.dimension,
        )
        app.state.settings = settings
        app.state.repository = repository
        app.state.ingestion_service = IngestionService(
            extractor=PyMuPDFExtractor(),
            chunker=SentenceChunker(
                chunk_size=settings.chunk_size_words,
                overlap=settings.chunk_overlap_words,
            ),
            embedder=embedder,
            repository=repository,
        )
        app.state.search_service = SearchService(embedder, repository)
        yield

    app = FastAPI(
        title="PDF Ingestor & Semantic Search API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    _register_exception_handlers(app)
    return app


# The Swagger spec returns errors as {"error": "..."}, while FastAPI's
# default is {"detail": ...}. These handlers make every error path match.

async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
        headers=getattr(exc, "headers", None),
    )


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


async def ingestion_error_handler(request: Request, exc: IngestionError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    message = INTERNAL_ERROR_MESSAGES.get(request.url.path, "Internal server error.")
    return JSONResponse(status_code=500, content={"error": message})


def _register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(IngestionError, ingestion_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)


app = create_app()
