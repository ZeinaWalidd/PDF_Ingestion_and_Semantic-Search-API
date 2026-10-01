from fastapi import FastAPI, File, HTTPException, UploadFile

from app.services.pdf_service import (
    PDFExtractionError,
    extract_pages,
)
from app.services.chunking_service import chunk_pages

app = FastAPI(
    title="PDF Ingestor & Semantic Search API",
    version="1.0.0",
)


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
            pages = extract_pages(file_bytes)
            
            chunks = chunk_pages(
                pages=pages,
                document=filename,
            )
        except PDFExtractionError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process '{filename}'.",
            ) from exc
        
        if not pages:
            raise HTTPException(
                status_code=400,
                detail=f"No text could be extracted from '{filename}'.",
            )
            
        ingested_files.append(filename)

        print(
            f"{filename}: extracted "
            f"{len(pages)} pages and "
            f"{len(chunks)} chunks"
        )
    
    return {
        "message": (
            f"Successfully ingested "
            f"{len(ingested_files)} PDF document(s)."
            ),
        "files": ingested_files,
    }
    
    
@app.post("/search/")
def search():
    return {
        "results": [],
    }
    
    
