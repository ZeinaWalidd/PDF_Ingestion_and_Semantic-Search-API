from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from app.api.dependencies import (
    IngestionServiceDep,
    RepositoryDep,
    SearchServiceDep,
    SettingsDep,
)
from app.api.schemas import IngestResponse, SearchRequest, SearchResponse, SearchResult
from app.config import Settings
from app.services.ingestion_service import (
    IngestionError,
    check_file_count,
    check_filename,
    check_size,
    read_directory,
)

router = APIRouter()


@router.get("/health")
def health(repository: RepositoryDep):
    if not repository.is_ready():
        raise HTTPException(status_code=503, detail="Vector database is unavailable.")
    return {"status": "healthy"}


MAX_DOCUMENT_ID_LENGTH = 200

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


@router.post("/ingest/", response_model=IngestResponse, openapi_extra=INGEST_OPENAPI)
async def ingest(
    request: Request, service: IngestionServiceDep, settings: SettingsDep
):
    async with request.form(max_files=settings.max_files_per_request) as form:
        items = form.getlist("input")
        if not items:
            raise HTTPException(status_code=400, detail="input: Field required")
        replace = _parse_bool(form.get("replace"), field="replace")
        document_id = _parse_document_id(form.get("document_id"))
        files = await _collect_files(items, settings)

    # Parsing and embedding are CPU-bound; a worker thread keeps the event
    # loop free for other requests (concurrent uploads, searches, /health).
    ingested = await run_in_threadpool(service.ingest, files, replace, document_id)

    noun = "document" if len(ingested) == 1 else "documents"
    return IngestResponse(
        message=f"Successfully ingested {len(ingested)} PDF {noun}.",
        files=ingested,
    )


@router.post("/search/", response_model=SearchResponse)
def search(body: SearchRequest, service: SearchServiceDep):
    hits = service.search(body.query, top_k=body.top_k, min_score=body.min_score)
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


async def _collect_files(
    items: list[UploadFile | str], settings: Settings
) -> list[tuple[str, bytes]]:
    files = []
    for item in items:
        if isinstance(item, UploadFile):
            filename = item.filename or ""
            check_filename(filename)
            check_size(filename, item.size, settings.max_file_size_mb)
            files.append((filename, await item.read()))
        else:
            files.extend(
                await run_in_threadpool(
                    read_directory,
                    item,
                    settings.data_dir,
                    settings.max_file_size_mb,
                    settings.max_files_per_request,
                )
            )

    check_file_count(len(files), settings.max_files_per_request)
    return files
