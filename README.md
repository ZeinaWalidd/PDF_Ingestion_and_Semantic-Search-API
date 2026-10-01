# PDF Ingestion & Semantic Search API

A containerized service that ingests PDF documents, splits them into chunks,
embeds each chunk with an open-source sentence-transformer model, stores the
vectors in Qdrant, and returns the most semantically relevant chunks for a
natural-language query.

- **API:** FastAPI (`/ingest/`, `/search/`, `/health`, interactive docs at `/docs`)
- **PDF extraction:** PyMuPDF
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU)
- **Vector database:** Qdrant
- **Orchestration:** Docker Compose, driven by `orchestrate.sh`

---

## Quick start

Requirements: Docker with Compose v2. Nothing else needs to be installed on the host.

```bash
# Build and start the app + vector DB; returns once both are healthy
./orchestrate.sh --action start

# Ingest a document
curl -X POST "http://localhost:8000/ingest/" -F "input=@data/sample.pdf"

# Search
curl -X POST "http://localhost:8000/search/" \
  -H "Content-Type: application/json" \
  -d '{"query": "How does semantic search work?"}'

# Stop and remove containers, volumes and networks
./orchestrate.sh --action terminate
```

The first `start` builds the image and downloads the embedding model, which
takes a few minutes. Later starts reuse the image and are ready in about 20 seconds.

---

## API

The contract follows the provided [Swagger spec](docs/swagger.yaml). Every error
response has the shape `{"error": "<message>"}`.

### `POST /ingest/`

`multipart/form-data` with a field named `input`, which may be:

| Input | Example |
|---|---|
| A single PDF | `-F "input=@data/sample.pdf"` |
| Several PDFs | `-F "input=@data/sample.pdf" -F "input=@data/football_rules.pdf"` |
| A directory path inside the container | `-F "input=/data"` |

`./data` on the host is mounted read-only at `/data`, so `-F "input=/data"`
ingests every sample PDF. Relative paths are resolved against `/data`. Paths
outside it are rejected. Subdirectories are not searched.

```json
{"message": "Successfully ingested 4 PDF documents.",
 "files": ["football_rules.pdf", "renewable_energy.pdf", "sample.pdf", "sourdough_baking.pdf"]}
```

### `POST /search/`

```json
{"query": "How does semantic search work?", "top_k": 3}
```

`query` is required and cannot be blank. `top_k` is optional (default 5, range 1–50).

```json
{
  "results": [
    {
      "document": "sample.pdf",
      "score": 0.688,
      "content": "Semantic Search: An Overview Traditional keyword search matches ...",
      "page": 1,
      "chunk_id": 0
    }
  ]
}
```

`score` is cosine similarity, where higher means more relevant. In the sample
data, relevant chunks score about 0.45–0.70 and unrelated ones score below 0.1.
`page` and `chunk_id` are extra fields beyond the spec, so each result can be
traced back to its source.

### `GET /health`

Returns `200 {"status": "healthy"}` when the API is up and Qdrant is reachable,
and `503` otherwise. Docker's healthcheck and `orchestrate.sh` both rely on it.

---

## Architecture

```
            ┌─────────────────────────── app container ───────────────────────────┐
 POST       │                                                                     │
 /ingest/ ──┼─► collect inputs ─► validate + extract ─► chunk ─► embed ─► upsert ──┼──► Qdrant
            │   (files / dir)      (pdf_service)       (chunking) (MiniLM)          │   (vectors +
 POST       │                                                                     │    payload)
 /search/ ──┼─► validate query ─► embed query ─► nearest-neighbour query ◄─────────┼───
            └─────────────────────────────────────────────────────────────────────┘
```

```
app/
├── main.py                    # HTTP layer: routes, request models, error handlers, startup
├── config.py                  # All settings, overridable via environment variables
└── services/
    ├── pdf_service.py         # Bytes → pages of text (PDF parsing, plain-text fallback)
    ├── chunking_service.py    # Pages → sentence-aware, overlapping chunks
    ├── ingestion_service.py   # Input validation, directory handling, per-document pipeline
    ├── embedding_service.py   # Wraps the sentence-transformer model
    └── vector_store.py        # Wraps Qdrant: collection setup, upsert, search
```

### Ingestion flow

1. **Collect.** Uploaded files and directory paths are turned into a list of
   `(filename, bytes)` pairs. The filename, the size and the number of files
   are checked before any file contents are read.
2. **Prepare every document first.** Each file is validated, its text is
   extracted, and it's chunked. If **any** file fails, the request returns 400
   and **nothing is stored**, so a bad file in a batch never leaves a partial
   ingestion behind.
3. **Embed and store.** Each document's chunks are embedded in one batch and
   upserted into Qdrant.

CPU-heavy work (parsing, embedding) runs in a thread pool, so the event loop
stays free and concurrent uploads, searches and health checks don't block one
another.

---

## Design decisions

### Chunking: whole sentences, sized to the model's limit
MiniLM truncates input after **256 tokens** without warning, and English
averages about 1.3 tokens per word. Chunks are therefore at most **150 words**,
which measured at 196 tokens at most on real text. Chunks overlap by up to
**30 words** of whole sentences.

- **Sentence packing:** sentences are added to a chunk until the next one would
  exceed the limit, so chunks don't end mid-sentence. A run longer than one
  chunk (a table, or text with no punctuation) is split with a sliding window,
  so no chunk goes over the limit.
- **Per-page chunks:** chunks never cross a page, so every result points to
  exactly one page.
- **Dehyphenation:** words hyphenated across line breaks (`embed-\nding`) are
  rejoined before chunking.

### Document identity: content hash
`doc_id` is the SHA-256 of the extracted text. Each chunk's Qdrant point ID is
derived from `(doc_id, chunk_index)`. As a result:
- **Uploading again is safe:** re-uploading the same document, even under
  another name, or several uploads of it at once, overwrites the same points
  instead of creating duplicates.
- **Leftover chunks are removed:** if the chunk settings change, a re-upload
  deletes chunks with a higher index than the new chunk count.

### Embeddings
- **Model:** `all-MiniLM-L6-v2` is small (~90 MB), fast on CPU and a well-known
  baseline.
- **Normalised vectors:** vectors are scaled to length 1, so cosine similarity is
  a dot product and scores have a consistent range.
- **Baked into the image:** the model is downloaded at build time and the
  container runs with `HF_HUB_OFFLINE=1`, so startup is fast and never needs the
  internet. It's loaded once at startup, not per request.
- **CPU-only torch:** used to keep the image small enough. The CUDA build alone
  is several GB.

### Why Qdrant
A single container with a REST API, a readiness endpoint for health checks,
cosine distance with an HNSW index, and payload filtering. Filtering is used to
clean up leftover chunks. The collection is created on startup if it's missing,
and startup fails with a clear error if an existing collection has a different
vector dimension.

### Orchestration
`orchestrate.sh` checks that Docker and Compose v2 are available, then runs
`docker compose up --build --wait`. Both containers have healthchecks, and the
app waits for Qdrant to be healthy, so the script returns only once the API can
serve requests. If startup fails, it prints recent logs and exits non-zero.
`terminate` removes containers, volumes and networks. It keeps the built image,
so the next start doesn't download dependencies again.

---

## Edge cases

| Case | Behaviour |
|---|---|
| Not a `.pdf` file | 400 `Only PDF files are accepted.` |
| Empty file | 400 `File '<name>' is empty.` |
| Corrupt PDF (has a PDF header, but can't be parsed) | 400 `...corrupt or unreadable PDF.` |
| Password-protected PDF | 400 `...Password-protected PDFs are not supported.` |
| PDF with no extractable text (e.g. scanned) | 400 `No text could be extracted from '<name>'.` |
| Binary data named `.pdf` | 400 `...not a valid PDF.` |
| Plain UTF-8 text named `.pdf` | **Accepted** as a one-page document, and a warning is logged (see below) |
| File over 50 MB or more than 20 files | 400 with the limit in the message |
| Directory outside `/data`, missing, or with no PDFs | 400 |
| Missing `input` field | 400 `input: Field required` |
| Empty or whitespace-only query | 400 `Query cannot be empty.` |
| Malformed JSON, wrong types, `top_k` out of range | 400 with a description |
| Qdrant unreachable | `/health` returns 503; other endpoints return 500 with a generic message, and the traceback is logged. The app recovers automatically once Qdrant is back. |
| Concurrent uploads and searches | Handled in parallel; uploading the same document at once still stores it only once |

**Why plain text is accepted:** the provided test suite uploads plain text named
`sample.pdf` and expects a 200. The fallback is deliberately narrow. Only files
**without** a `%PDF-` header that decode as clean UTF-8 text are accepted. A
file that claims to be a PDF but is broken is rejected rather than read as text.

Sample files for each case are in [`data/edge_cases/`](data/edge_cases/), and
`data/` is generated by [`scripts/generate_sample_data.py`](scripts/generate_sample_data.py).

---

## Configuration

All settings are environment variables with defaults in [`app/config.py`](app/config.py).
They can be overridden in `docker-compose.yml`.

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Also a build argument, since the model is baked into the image |
| `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` | `150` / `30` | Chunk size and overlap |
| `DEFAULT_TOP_K` / `MAX_TOP_K` | `5` / `50` | Number of search results |
| `MAX_FILE_SIZE_MB` / `MAX_FILES_PER_REQUEST` | `50` / `20` | Upload limits |
| `DATA_DIR` | `/data` | Root for directory ingestion |
| `QDRANT_URL` / `QDRANT_COLLECTION` | `http://qdrant:6333` / `pdf_chunks` | Vector database |
| `START_TIMEOUT_SECONDS` | `180` | How long `orchestrate.sh` waits for the services to become healthy |

---

## Testing

With the stack running:

```bash
pip install pytest requests
pytest tests/suite.py
```

`tests/suite.py` is the provided end-to-end test. It ingests a file and runs a
search. pytest only finds `test_*.py` files on its own, so pass the path
explicitly.

To follow the logs:

```bash
docker compose logs -f app
```

Each ingestion logs the pages, chunks and embedding time per document, and each
search logs its query, result count and time taken.

---

## Limitations and next steps

- **No OCR.** Scanned PDFs are rejected. Adding Tesseract would cover them.
- **Content-based identity means edits are new documents.** Uploading an edited
  `report.pdf` stores it alongside the old version. A "replace by filename" mode
  would need an explicit document ID from the client.
- **Batches are processed in memory.** That's up to 20 × 50 MB per request. Large
  ingestions should become a background job queue, with the endpoint returning a
  job ID.
- **No relevance cut-off.** Search always returns the `top_k` closest chunks.
  A minimum score, or a re-ranking step with a cross-encoder, would improve
  precision.
- **Simple sentence splitting.** It splits after abbreviations like "e.g.",
  which makes a chunk end slightly early but never mid-word.
- **Image size of about 2.3 GB**, mostly PyTorch. An ONNX runtime such as
  `fastembed` would cut it substantially.
- **No authentication or rate limiting.** This is a local evaluation setup.
