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

`query` is required and cannot be blank. Optional fields:

| Field | Default | Meaning |
|---|---|---|
| `top_k` | 5 (range 1–50) | Maximum number of results |
| `min_score` | 0.15 (range -1 to 1) | Drop results below this similarity; `-1` turns the cut-off off |

A query unrelated to anything ingested returns `{"results": []}` rather than
the "least bad" chunks.

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

`score` is cosine similarity, where higher means more relevant.
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
├── main.py                    # create_app(): composition root, builds adapters + services, error handlers
├── config.py                  # Settings class, validated at startup, overridable via environment variables
├── api/                       # HTTP layer only
│   ├── routes.py              # Parse request → call service → shape response
│   ├── schemas.py             # Request/response models
│   └── dependencies.py        # FastAPI Depends() providers
└── services/
    ├── ports.py               # TextExtractor, Chunker, Embedder and ChunkRepository interfaces (Protocols)
    ├── ingestion_service.py   # IngestionService: validate → extract → chunk → embed → store
    ├── search_service.py      # SearchService: embed query → nearest-neighbour search
    ├── pdf_service.py         # TextExtractor adapter: PDF parsing with a plain-text fallback
    ├── chunking_service.py    # Chunker adapter: sentence-aware, overlapping chunks
    ├── embedding_service.py   # Embedder adapter for sentence-transformers
    └── vector_store.py        # ChunkRepository adapter for Qdrant
```

### Layers and patterns

```
 api/routes.py ──Depends()──► IngestionService / SearchService ──► TextExtractor, Chunker, Embedder, ChunkRepository (ports)
   (HTTP only)                  (business rules, orchestration)          ▲             ▲            ▲              ▲
                                                               PyMuPDFExtractor SentenceChunker EmbeddingService VectorStore
                                                                  (PyMuPDF)                 (sentence-transformers) (Qdrant)
```

- **Service layer.** `IngestionService` and `SearchService` hold the business
  rules: validating every file before storing any, document identity, and
  timing logs. Routes only translate HTTP into service calls,
  so the same services could back a CLI or a queue worker.
- **Ports and adapters.** The services depend on the `TextExtractor`,
  `Chunker`, `Embedder` and `ChunkRepository` protocols in `ports.py`, never on
  PyMuPDF, sentence-transformers or Qdrant directly. Swapping one step (an OCR
  extractor for scanned PDFs, a token-based chunker, an ONNX `fastembed`
  adapter to shrink the image, or pgvector instead of Qdrant) means writing one
  new adapter, and the services don't change. Extractors signal unreadable
  files by raising `ExtractionError`, so every extractor's errors become a 400
  the same way. The type checker verifies that the current adapters satisfy
  the protocols.
- **Dependency injection.** Routes declare what they need
  (`service: SearchServiceDep`, `settings: SettingsDep`) through FastAPI's
  `Depends`, instead of reading global state. `main.create_app()` is the
  composition root: the single place that chooses the concrete adapters and
  builds each object once at startup. Any adapter can be passed in instead,
  for example to use fakes in tests:

  ```python
  app = create_app(Settings(data_dir=tmp_path), embedder=FakeEmbedder(), repository=InMemoryRepository())
  with TestClient(app) as client:
      ...
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
  exceed the limit, so chunks don't end mid-sentence.
- **Sentence detection without NLP dependencies.** A boundary is `.`, `!` or `?`
  followed by a word that can start a sentence (a capital, digit or quote), so
  "e.g. the model" stays together. A period after a known abbreviation
  (`Dr.`, `Fig.`, `et al.`, `vs.`), a single initial (`J. Smith`) or a dotted
  form (`U.S.`, `Ph.D.`) is not a boundary either. spaCy or NLTK would handle
  more cases, but they would add a large dependency for a small gain: a wrong
  boundary only makes a chunk end slightly early. A run longer than one
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
  instead of creating duplicates. No locking is needed, because concurrent
  writes of the same document write identical points.
- **Edited files are new documents:** an edited file has a different hash, so
  it's stored next to the old version rather than replacing it. Versioning
  wasn't part of the brief, so it's left out to keep the write path simple.

### Relevance cut-off
Search drops results below `min_score` (default **0.15**), and Qdrant applies
the filter itself through `score_threshold`. The default is calibrated, not
guessed. On the sample data, the best match for off-topic queries ("What is the
capital of France?", "best pizza toppings", …) scored **0.07–0.13**, and the
best match for on-topic queries scored **0.17–0.73**. The value depends on the
model, so it is configurable (`DEFAULT_MIN_SCORE`) and can be overridden per
request.

The lowest on-topic score came from "what is HNSW" (0.17), because small
embedding models handle acronyms and rare terms poorly. Hybrid search, which
combines vectors with keyword scoring such as BM25, is the standard fix.

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
cosine distance with an HNSW index, and a server-side score threshold. The
collection is created on startup if it's missing,
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
[`tests/test_edge_cases.py`](tests/test_edge_cases.py) checks them against the running stack.

---

## Configuration

All settings are fields of the `Settings` class in [`app/config.py`](app/config.py),
and each one can be overridden by the environment variable below, for example in
`docker-compose.yml`. Values are validated when the app starts, so an invalid
configuration (a non-numeric limit, `CHUNK_OVERLAP_WORDS` not smaller than
`CHUNK_SIZE_WORDS`, or `DEFAULT_TOP_K` above `MAX_TOP_K`) stops startup with a
clear error instead of failing on the first request.

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Also a build argument, since the model is baked into the image |
| `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` | `150` / `30` | Chunk size and overlap |
| `DEFAULT_TOP_K` / `MAX_TOP_K` | `5` / `50` | Number of search results |
| `DEFAULT_MIN_SCORE` | `0.15` | Default relevance cut-off (cosine similarity) |
| `MAX_FILE_SIZE_MB` / `MAX_FILES_PER_REQUEST` | `50` / `20` | Upload limits |
| `DATA_DIR` | `/data` | Root for directory ingestion |
| `QDRANT_URL` / `QDRANT_COLLECTION` | `http://qdrant:6333` / `pdf_chunks` | Vector database |
| `START_TIMEOUT_SECONDS` | `180` | How long `orchestrate.sh` waits for the services to become healthy |

---

## Testing

```bash
pip install pytest requests

# Chunker unit tests: no Docker needed, run in well under a second
pytest tests/test_chunking.py

# Everything, with the stack running
pytest
```

- `tests/test_chunking.py` checks the chunker on its own: sentence boundaries
  (including `Dr.`, `Fig.`, `J. Smith`, `U.S.` and `e.g.`), the word limit,
  overlap between chunks, that no text is lost, the sliding window for long
  unpunctuated runs, and that chunks never cross pages. The chunker uses only
  the standard library, so these tests need nothing besides pytest.
- `tests/suite.py` is the provided end-to-end test: it ingests a file and runs
  a search.
- `tests/test_edge_cases.py` covers invalid files, a bad file inside a batch,
  invalid directories, empty and malformed queries, and concurrent uploads,
  including the same file uploaded five times at once being stored only once.

To follow the logs:

```bash
docker compose logs -f app
```

Each ingestion logs the pages, chunks and embedding time per document, and each
search logs its query, result count and time taken.

---

## Limitations and next steps

- **No OCR.** Scanned PDFs are rejected. Adding Tesseract would cover them.
- **No document versioning or deletion.** Re-ingesting an edited file keeps the
  old version searchable. Replacing by a stable document ID, plus a delete
  endpoint, would be the next step.
- **Batches are processed in memory.** That's up to 20 × 50 MB per request. Large
  ingestions should become a background job queue, with the endpoint returning a
  job ID.
- **Vector-only retrieval.** Acronyms and rare terms score low (see "Relevance
  cut-off"). Hybrid search (BM25 plus vectors) and a cross-encoder re-ranking
  step would improve precision. The 0.15 cut-off was calibrated on a small
  sample set and should be re-tuned on real data.
- **Rule-based sentence splitting.** Abbreviations missing from the list, or a
  sentence that ends with an initial ("vitamin C. The…"), can move a chunk
  boundary by a sentence.
- **Image size of about 2.3 GB**, mostly PyTorch. An ONNX runtime such as
  `fastembed` would cut it substantially.
- **No authentication or rate limiting.** These are deliberately out of scope:
  the assessment doesn't ask for them, and its usage flow and test suite call
  the API without credentials. In production I'd put an API key or OAuth in
  front of the service and rate-limit at the gateway.
