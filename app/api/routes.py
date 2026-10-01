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
        files = await _collect_files(items, settings)

    # Parsing and embedding are CPU-bound; a worker thread keeps the event
    # loop free for other requests (concurrent uploads, searches, /health).
    ingested = await run_in_threadpool(service.ingest, files)

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
                score=round(hit.score, 4),
                content=hit.content,
                page=hit.page,
                chunk_id=hit.chunk_id,
            )
            for hit in hits
        ]
    )


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
