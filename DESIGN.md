# Design Summary

A containerized service that ingests PDFs, embeds their text, stores the vectors
in a vector database, and returns the most semantically relevant chunks for a
natural-language query. Full setup and API details are in the
[README](README.md); this document covers the design and the reasoning behind it.

## Components

| Part | Choice | Why |
|---|---|---|
| API | FastAPI | Async I/O, request validation and generated `/docs` from the same type hints |
| PDF extraction | PyMuPDF | Fast, pure-wheel install, per-page text and a password check |
| Embeddings | `all-MiniLM-L6-v2` (384-dim, CPU) | ~90 MB, well-known baseline, fast enough on CPU |
| Vector database | Qdrant | Single container, REST API, readiness endpoint, cosine + HNSW, server-side score threshold |
| Orchestration | Docker Compose via `orchestrate.sh` | One command up and down, with healthchecks on both containers |

The embedding model is baked into the image at build time and the container runs
with `HF_HUB_OFFLINE=1`, so startup is fast and never needs the network.

## Data flow

```
POST /ingest/  →  collect (files or directory)  →  validate  →  extract text
               →  chunk per page  →  embed in batches  →  upsert into Qdrant

POST /search/  →  validate query  →  embed query  →  nearest-neighbour search
               →  results above min_score
```

Ingestion prepares (extracts and chunks) **every** file before storing any, so
one bad file in a batch returns 400 and leaves nothing behind.

## Structure

```
app/
├── main.py       create_app(): composition root + error handlers
├── config.py     Settings, validated at startup
├── api/          routes, request/response schemas, Depends() providers
└── services/     ports.py (interfaces) + services + adapters
```

- **Service layer.** `IngestionService` and `SearchService` hold the rules.
  Routes only translate HTTP into service calls, so the same services could
  back a CLI or a worker.
- **Ports and adapters.** Services depend on the `TextExtractor`, `Chunker`,
  `Embedder` and `ChunkRepository` protocols in `ports.py`, never on PyMuPDF,
  sentence-transformers or Qdrant. Replacing the database or the model means
  writing one adapter and changing one line in `create_app()`.
- **Dependency injection.** `create_app()` is the only place that names concrete
  classes. It builds each object once at startup, because loading the model
  takes seconds. Routes receive what they need through `Depends`.

## Key decisions

**Chunking: whole sentences, 150 words, 30-word overlap.** MiniLM silently
truncates after 256 tokens, and English runs about 1.3 tokens per word, so 150
words keeps every chunk inside the model's limit (at most 196 tokens on real
text). Chunks end on sentence boundaries rather than mid-sentence, and overlap
preserves context across boundaries. Boundaries are detected with a regex plus a
short abbreviation list instead of spaCy or NLTK: a missed boundary only ends a
chunk a sentence early, which isn't worth a large dependency. Chunks never cross
a page, so every result maps to exactly one page. A run longer than one chunk (a
table, or text without punctuation) falls back to a sliding window.

**Document identity: content hash.** `doc_id` is the SHA-256 of the extracted
text, and each point ID is derived from `(doc_id, chunk index)`. Re-uploading the
same document, even renamed or several times at once, overwrites the same points
instead of duplicating them, and no locking is needed because concurrent writes
are identical. An edited file hashes differently, so it's stored as a new
document; versioning and deletion were left out as outside the brief.

**Relevance cut-off, calibrated not guessed.** Results below `min_score`
(default 0.15) are dropped, with the filter applied inside Qdrant. On the sample
data, off-topic queries peaked at 0.07–0.13 while on-topic queries scored
0.17–0.73. An unrelated query therefore returns `{"results": []}` rather than
the least-bad chunks. The value depends on the model, so it's configurable and
can be overridden per request.

**Normalised vectors.** Vectors are scaled to length 1, so cosine similarity is
a dot product and scores stay in a predictable range, which is what makes a
fixed threshold meaningful.

**Concurrency.** Parsing and embedding are CPU-bound, so they run in a worker
thread (`run_in_threadpool`). The event loop stays free, and concurrent uploads,
searches and health checks don't block one another.

**Errors.** Every failure returns `{"error": "..."}` to match the provided
Swagger spec, including FastAPI's 422 validation responses, which are rewritten
as 400. Client mistakes get a specific message; unexpected failures return a
generic 500 and the traceback goes to the logs only. `/health` returns 503 when
Qdrant is unreachable, and the app recovers on its own once it's back.

**Configuration.** All settings are environment variables with defaults in
`app/config.py`. They're validated at startup, so an inconsistent configuration
fails immediately rather than on the first request.

## Testing

- `tests/test_chunking.py` — 16 unit tests for the chunker, no Docker required.
- `tests/suite.py` — the provided end-to-end ingest-and-search test.
- `tests/test_edge_cases.py` — invalid files, a bad file inside a batch, invalid
  directories, empty and malformed queries, and concurrent uploads (including
  the same file five times at once being stored only once).

## Known limitations

- **No OCR**, so scanned PDFs are rejected with a clear message.
- **No versioning or delete**, so re-ingesting an edited file keeps the old
  version searchable.
- **Vector-only retrieval**, so acronyms and rare terms score low; hybrid search
  (BM25 + vectors) and re-ranking would be the next step.
- **Requests are processed in memory**, up to 20 files of 50 MB; large
  ingestions should become a background job.
- **No authentication or rate limiting**, deliberately out of scope here, but
  needed in production.
- **Image is ~2.3 GB**, mostly PyTorch; an ONNX runtime such as `fastembed`
  would cut it substantially.
