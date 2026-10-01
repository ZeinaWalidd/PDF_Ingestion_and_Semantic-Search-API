import logging

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.services.pdf_service import (
    PDFExtractionError,
    extract_pages,
)
from app.services.chunking_service import chunk_pages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Ingestor & Semantic Search API",
    version="1.0.0",
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


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    message = INTERNAL_ERROR_MESSAGES.get(request.url.path, "Internal server error.")
    return JSONResponse(status_code=500, content={"error": message})



@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ingest/")
async def ingest(
    input: list[UploadFile] = File(...)
):
    if not input:
        raise HTTPException(
            status_code=400,
            detail="No files were provided",
        )

    ingested_files = []

    for uploaded_file in input:
        filename = uploaded_file.filename or ""

        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are accepted.",
            )

        file_bytes = await uploaded_file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"File '{filename}' is empty.",
            )

        try:
            pages = extract_pages(file_bytes, source=filename)
        except PDFExtractionError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process '{filename}': {exc}",
            ) from exc

        if not pages:
            raise HTTPException(
                status_code=400,
                detail=f"No text could be extracted from '{filename}'.",
            )

        chunks = chunk_pages(
            pages=pages,
            document=filename,
        )

        ingested_files.append(filename)

        logger.info(
            "%s: extracted %d pages and %d chunks",
            filename, len(pages), len(chunks),
        )

    noun = "document" if len(ingested_files) == 1 else "documents"
    return {
        "message": f"Successfully ingested {len(ingested_files)} PDF {noun}.",
        "files": ingested_files,
    }


@app.post("/search/")
def search():
    return {
        "results": [],
    }