# Code walkthrough

What each part of the codebase does, function by function. Covers V0 through V4 Part 1
(containerization; V4's Langfuse tracing/tests/CI are Part 2, not written yet).

## Layout

```
src/rag/
  config.py      all settings, read from environment variables
  types.py       shared RetrievedChunk data type
  embeddings.py  dense embeddings (sentence-transformers / BGE)
  sparse.py      sparse BM25 embeddings (fastembed)
  indexing.py    load PDF, chunk, embed, write to Qdrant
  retrieval.py   dense / hybrid / hybrid_reranker search
  reranker.py    cross-encoder reranking
  llm.py         local Hugging Face model, generates answers and judges
  pipeline.py    glues retrieval + generation together
  db.py          SQLAlchemy engine/session setup
  db_models.py   DocumentRecord, QueryLog, FeedbackRecord ORM models
  schemas.py     Pydantic request/response models for the API
  api.py         FastAPI app (V3)

ingest.py         CLI: index documents
ask.py            CLI: ask a question
compare_modes.py  CLI: compare retrieval modes side by side
serve.py          CLI: start the FastAPI service

evaluation/
  metrics.py          Recall@K, MRR, nDCG, context precision (pure functions)
  build_dataset.py    generates evaluation/dataset.json from the indexed corpus
  retrieval_eval.py   scores retrieval only (fast, no LLM)
  generation_eval.py  scores generation via LLM-as-judge (faithfulness, relevancy)
  run_eval.py         runs both stages for all 3 modes, writes results.csv

tests/
  conftest.py    isolates tests from the real qdrant_data/app.db
  test_api.py    API integration tests

docker-compose.yml   qdrant, postgres, self-hosted langfuse (V4)
Dockerfile           API image (build-only, not part of the default compose stack)
```

## config.py

Every tunable value lives here as a module-level constant, read from an environment variable with
a default. Nothing else in the codebase hardcodes a model name or a number, so behavior changes by
setting env vars (or `.env`), not by editing code.

| Constant | Default | Used for |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | dense embedding model |
| `LLM_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | local generation/judge model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 120 | text splitter settings |
| `TOP_K` | 5 | final number of chunks returned to the LLM |
| `CANDIDATE_K` | 20 | candidate pool size before reranking |
| `RETRIEVAL_MODE` | `dense` | default mode for `retrieve()` |
| `SPARSE_MODEL` | `Qdrant/bm25` | fastembed sparse model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | cross-encoder model |
| `QDRANT_URL` | `http://localhost:6333` | real Qdrant server; blank falls back to embedded mode |
| `QDRANT_PATH` / `QDRANT_COLLECTION` | `./qdrant_data` / `documents` | embedded Qdrant storage |
| `DATA_DIR` / `UPLOAD_DIR` | `./data` / `./data/uploads` | curated PDFs vs. API uploads |
| `DATABASE_URL` | Postgres compose service | app DB; use a `sqlite:///` URL to skip Docker |
| `EVAL_NUM_QUESTIONS` / `EVAL_SEED` | 30 / 42 | evaluation dataset size and sampling seed |
| `EVAL_DATASET_PATH` / `EVAL_RESULTS_PATH` | `evaluation/dataset.json` / `results.csv` | eval I/O |

## types.py

```python
@dataclass
class RetrievedChunk:
    text: str      # the chunk's raw text
    source: str    # filename, e.g. "pytorch.pdf"
    page: int      # 1-indexed page number
    score: float   # meaning depends on retrieval mode, see retrieval.py below
```

This is the one shared type that flows from retrieval through reranking into the final answer, so
every stage can be swapped without the others caring how the score was computed.

## embeddings.py (dense)

- `get_embedding_model()`: loads the `sentence-transformers` model named in `config.EMBEDDING_MODEL`.
  Wrapped in `@lru_cache(maxsize=1)` so it is only loaded once per process, not once per call.
- `embed_texts(texts)`: takes a list of strings, returns a list of embedding vectors (as plain
  Python lists). `normalize_embeddings=True` makes the vectors unit-length, which is what lets
  cosine similarity work correctly in Qdrant.
- `embed_query(text)`: same as above for a single string, used when embedding the user's question.
- `embedding_dimension()`: returns the vector size of the loaded model, used once at collection
  creation time so Qdrant knows how big to make the dense vector slot.

## sparse.py (BM25 via fastembed)

- `get_sparse_model()`: loads `fastembed.SparseTextEmbedding` for `config.SPARSE_MODEL`
  (`Qdrant/bm25`). Also `lru_cache`d.
- `embed_sparse_texts(texts)`: used at **ingestion** time. Returns one `SparseVector` per chunk.
  Each vector is a set of (token id, weight) pairs, not a dense array, since most tokens in the
  vocabulary don't appear in a given chunk.
- `embed_sparse_query(text)`: used at **query** time. Same shape, but computed slightly
  differently (see the note in `docs/v1-notes.md`): document vectors carry a per-term
  weight, query vectors are unweighted (all values are 1), and Qdrant applies IDF weighting at
  query time via the `Modifier.IDF` setting on the collection.

## indexing.py

- `load_document(path)`: loads one PDF with LangChain's `PyPDFLoader` (one `Document` per page),
  stamps each page's metadata with `source` (the filename) and `page` (1-indexed, since `pypdf`
  numbers pages from 0).
- `load_documents(data_dir=config.DATA_DIR)`: finds every `*.pdf` in `data_dir` and calls
  `load_document` on each, used by the CLI's bulk path.
- `chunk_documents(documents)`: runs LangChain's `RecursiveCharacterTextSplitter` over the pages,
  using `config.CHUNK_SIZE` / `config.CHUNK_OVERLAP`. Each output chunk keeps its parent page's
  `source`/`page` metadata.
- `get_qdrant_client()`: **dual-mode**. Returns `QdrantClient(url=config.QDRANT_URL)` (a real
  server, e.g. the docker-compose `qdrant` service) if `QDRANT_URL` is set, otherwise
  `QdrantClient(path=config.QDRANT_PATH)` (embedded/on-disk, no server needed). See
  `docs/v4-notes.md` for why both modes exist.
- `ensure_collection(client, recreate=True)`: creates the collection with **two** named vectors,
  `"dense"` (size = embedding model's dimension, cosine distance) and `"sparse"` (with
  `modifier=Modifier.IDF`, BM25-style IDF weighting at query time). `recreate=True` (the CLI's
  default) drops and rebuilds the whole collection every run, idempotent, no migration logic
  needed. `recreate=False` (the API's incremental path) only creates the collection if it doesn't
  exist yet, leaving other documents' chunks untouched.
- `_point_id(document_id, chunk_index)`: deterministic id,
  `uuid.uuid5(POINT_ID_NAMESPACE, f"{document_id}:{chunk_index}")`. Re-indexing the same document
  produces the same point ids (idempotent upsert), and ids never collide across documents.
- `index_chunks(client, chunks, document_id)`: embeds (dense + sparse) and upserts every chunk as
  a `PointStruct`, payload carries `text`/`source`/`page`/`document_id`. Batches the upsert 256
  points at a time, a single request with thousands of points can exceed Qdrant server's 32MB
  HTTP request limit once real vectors and text are included (embedded mode has no such limit,
  see `docs/v4-notes.md`). Returns how many chunks were indexed.
- `index_documents(data_dir=config.DATA_DIR)`: the CLI's bulk entry point (`ingest.py`). Wipes and
  rebuilds the whole collection, `document_id` is each PDF's filename stem.
- `index_document_file(path, document_id)`: the API's incremental entry point. Indexes one PDF
  without touching the rest of the collection; `document_id` is the caller's choice (the API uses
  the SQLite/Postgres row's own primary key, linking a Qdrant chunk back to its DB row).
- `delete_document(document_id)`: deletes every point whose payload `document_id` matches, via a
  Qdrant filter, used by the API's `DELETE /documents/{id}`.

## retrieval.py

Three retrieval modes share one return type (`list[RetrievedChunk]`), so nothing downstream needs
to know which mode produced them.

- `_to_chunk(point)`: converts a raw Qdrant search hit into a `RetrievedChunk`.
- `_retrieve_dense(question, top_k)`: embeds the question, searches only the `"dense"` named
  vector, returns the top `top_k` hits. `score` here is cosine similarity (0 to 1).
- `_retrieve_hybrid(question, candidate_k)`: embeds the question both ways (dense + sparse), then
  issues one Qdrant query with two `Prefetch` stages (one per named vector, each pulling
  `candidate_k` candidates) fused with `FusionQuery(fusion=Fusion.RRF)`, Qdrant's built-in
  Reciprocal Rank Fusion. `score` here is the RRF score, not a similarity, so it isn't comparable
  across modes.
- `retrieve(question, mode=config.RETRIEVAL_MODE, top_k=config.TOP_K)`: the public entry point.
  - `mode="dense"` calls `_retrieve_dense` directly.
  - `mode="hybrid"` calls `_retrieve_hybrid` with `candidate_k=top_k` (no reranking stage, so it
    only needs to fetch exactly what it returns).
  - `mode="hybrid_reranker"` calls `_retrieve_hybrid` with the larger `config.CANDIDATE_K` pool,
    then hands those candidates to `reranker.rerank()` to cut down to `top_k`.

## reranker.py

- `get_reranker()`: loads a `sentence_transformers.CrossEncoder` for `config.RERANKER_MODEL`,
  `lru_cache`d like the other models.
- `rerank(question, chunks, top_k)`: scores every `(question, chunk.text)` pair with the
  cross-encoder (this is why it's more expensive than dense/BM25: it looks at the question and
  each candidate *together*, instead of comparing precomputed vectors), sorts by that score
  descending, and returns the top `top_k`. The returned `RetrievedChunk.score` is the raw
  cross-encoder logit, unbounded and not comparable to cosine or RRF scores.

## llm.py

- `_load_model()`: loads the tokenizer and model named in `config.LLM_MODEL` with
  `device_map="auto"` (puts it on the GPU if available) and `float16` on CUDA. `lru_cache`d so the
  multi-second weight load only happens once per process.
- `warm_up()`: calls `_load_model()` eagerly. Used by the API's startup hook so the first real
  request isn't slow, the CLI doesn't call this and pays the load cost on every invocation.
- `generate(system_prompt, user_prompt, max_new_tokens=400)`: the generic building block. Builds a
  two-message chat prompt, formats it with the model's own chat template
  (`tokenizer.apply_chat_template`), and greedily generates (`do_sample=False`, so outputs are
  deterministic) up to `max_new_tokens` new tokens. Returns just the newly generated text, with
  the echoed prompt stripped off. Used both for answering questions and, in `evaluation/
  generation_eval.py`, for judging them, one code path for both.
- `generate_answer(question, context, max_new_tokens=400)`: thin wrapper around `generate()` with
  a fixed system prompt instructing the model to answer only from the provided context.

## pipeline.py

- `Answer`: a small dataclass bundling the generated `answer` text with the `sources` that were
  retrieved for it.
- `answer_question(question, mode=config.RETRIEVAL_MODE, top_k=config.TOP_K)`: calls
  `retrieve()`, joins the retrieved chunks into a single context string (each chunk prefixed with
  its `[source, page N]` so the LLM can see where each snippet came from), calls
  `generate_answer()`, and returns an `Answer`. This is the one function `ask.py`, `api.py`'s
  `/query` endpoint, and `evaluation/generation_eval.py` all call, one shared path for "how the
  system answers a question."

## db.py, db_models.py (V3)

- `db.py`: builds a SQLAlchemy `engine` from `config.DATABASE_URL` (adds
  `check_same_thread=False` only for SQLite, Postgres doesn't need it), a `SessionLocal`
  sessionmaker, a declarative `Base`, a `get_db()` FastAPI dependency (yields a session, closes it
  after the request), and `init_db()` (`Base.metadata.create_all`, called at API startup).
- `db_models.py`: three ORM models. `DocumentRecord` (id, filename, `status`
  pending/processing/ready/failed, `chunk_count`, `error_message`, `created_at`). `QueryLog` (one
  row per `/query` call: question, mode, answer, `latency_ms`). `FeedbackRecord` (rating/comment,
  optionally linked to a `QueryLog` row via `query_id`).

## schemas.py (V3)

Pydantic request/response models for the API, kept separate from the ORM models in `db_models.py`
so the API's wire format can evolve independently of the storage schema. `QueryRequest.mode` is a
`Literal["dense", "hybrid", "hybrid_reranker"]`, so an invalid mode is a 422 before it ever reaches
`answer_question()`.

## api.py (V3)

FastAPI app. `lifespan()` (an `asynccontextmanager`, the modern replacement for the deprecated
`@app.on_event("startup")`) runs once at process start: `init_db()`, creates `UPLOAD_DIR`, and
warms up all three models (embedding, sparse, LLM) so the first request isn't slow.

- `GET /health`: checks Qdrant and the database are reachable, returns `HealthResponse`.
- `POST /documents`: validates the upload is a `.pdf`, creates a `DocumentRecord` with
  `status="pending"`, saves the file to `UPLOAD_DIR`, and schedules `_process_document` as a
  `BackgroundTasks` job (runs after the response is sent, in the same process, no external task
  queue). Returns 202 with the pending record.
- `_process_document(document_id, path)`: the background job. Sets `status="processing"`, calls
  `indexing.index_document_file()`, sets `status="ready"` with the real `chunk_count`, or
  `status="failed"` with `error_message` on any exception.
- `GET /documents`: lists all `DocumentRecord` rows.
- `DELETE /documents/{id}`: 404 if missing, otherwise deletes the Qdrant chunks
  (`indexing.delete_document`), the uploaded file, and the DB row.
- `POST /query`: calls `answer_question()`, times it, logs a `QueryLog` row, returns the answer
  with sources and the new `query_id`.
- `POST /feedback`: 404 if `query_id` is given but doesn't exist, otherwise stores a
  `FeedbackRecord`.

## evaluation/ (V2)

- `metrics.py`: `hit_rank`, `recall_at_k`, `reciprocal_rank`, `ndcg_at_k`, `context_precision`,
  pure functions operating on `(source, page)` tuples, no I/O.
- `build_dataset.py`: samples chunks straight from the indexed Qdrant collection (seeded random,
  filters short/boilerplate chunks), asks the LLM (via `rag.llm.generate`) to write one question
  and a reference answer per sampled chunk, writes `evaluation/dataset.json`. The sampled chunk's
  `(source, page)` is the gold label, no manual citation-writing.
- `retrieval_eval.py`: for each dataset question and mode, calls `rag.retrieval.retrieve()`,
  times it, scores it with `metrics.py`. No LLM generation, fast.
- `generation_eval.py`: for each dataset question and mode, calls `rag.pipeline.answer_question()`
  (the real end-to-end path), then a judge prompt (also via `rag.llm.generate`) scores
  faithfulness and answer relevancy 1-5 each.
- `run_eval.py`: runs both stages for all three modes, writes `evaluation/results.csv`.

## tests/ (V3)

- `conftest.py`: sets isolated env vars (`QDRANT_URL=""` to force embedded mode, a temp
  `QDRANT_PATH`, a temp SQLite `DATABASE_URL`, a distinct `QDRANT_COLLECTION`) **at module load
  time**, before `rag.config` or `rag.api` is ever imported, since `config.py` reads env vars at
  import time. This keeps the test suite from touching the real `qdrant_data/`/`app.db` or
  needing a running Qdrant/Postgres container. A session-scoped `client` fixture wraps the FastAPI
  app in a `TestClient` (models load once for the whole test run); `sample_pdf_path` slices the
  first 2 pages of `data/pytorch.pdf` for fast upload tests.
- `test_api.py`: health check; full document lifecycle (upload, poll until `ready`, query,
  feedback, delete, confirm gone); two 404 cases (unknown `query_id`, unknown document id).

## CLI scripts

- `ingest.py`: one line, calls `index_documents()` and prints the chunk count.
- `ask.py`: parses `question` and `--mode {dense,hybrid,hybrid_reranker}`, calls
  `answer_question()`, prints the answer then each unique `(source, page)` with its score.
- `compare_modes.py`: fixed semantic + exact-keyword query lists, calls `retrieve()` once per mode
  per query, prints the top 5 hits side by side.
- `serve.py`: `uvicorn.run("rag.api:app", ...)`, starts the API on port 8000.

## docker-compose.yml, Dockerfile (V4)

`docker-compose.yml` brings up the RAG app's data services (`qdrant`, `postgres`) plus a
self-hosted Langfuse stack (`langfuse-web`, `langfuse-worker`, `langfuse-postgres`, `clickhouse`,
`redis`, `minio`), adapted from Langfuse's official compose file. The API itself is **not** in
this file, it runs on the host via `serve.py` so it can use the GPU directly. See
`docs/v4-notes.md` for the full reasoning, including two real bugs found while wiring this up.

`Dockerfile` builds the API as a container anyway (multi-stage, `uv`-based), for a possible future
GPU-enabled cloud deployment, but isn't part of the default `docker compose up` stack.
