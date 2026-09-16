# V3 notes

What the production API adds on top of V0-V2, and why it's built the way it is. Real output
captured from a live run against `data/pytorch.pdf`, not illustrative.

## What changed

V0-V2 were scripts: `ingest.py` rebuilds the whole index from `data/*.pdf`, `ask.py` answers one
question per process. V3 turns `rag.pipeline.answer_question()` into a long-lived FastAPI service
(`src/rag/api.py`, run via `serve.py`) with five endpoints, backed by SQLite for application data
(document records, query history, feedback) alongside the existing Qdrant index for retrieval data.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Qdrant and database reachability |
| `POST /documents` | upload a PDF, indexed in the background |
| `GET /documents` | list documents and their status |
| `DELETE /documents/{id}` | remove a document's chunks, DB row, and uploaded file |
| `POST /query` | ask a question, get an answer with sources and a `query_id` |
| `POST /feedback` | rate a specific answer by `query_id` |

Swagger UI comes for free at `/docs`.

## SQLite now, Postgres in V4

No Docker or PostgreSQL is installed on this machine, the same situation V0 hit with Qdrant.
Rather than a native Postgres install or a hosted free tier, V3 uses SQLAlchemy over a local
SQLite file (`app.db`, gitignored). V4 already introduces Docker Compose with a real Postgres
container, so swapping `DATABASE_URL` there is a one-line config change, not a rewrite, since all
the application code goes through SQLAlchemy's engine abstraction. The roadmap's "Add PostgreSQL"
checklist item stays honestly unchecked until then.

## Incremental indexing

`ingest.py`'s bulk path (`index_documents`) wipes and rebuilds the whole Qdrant collection every
run, fine for a CLI, wrong for an API where documents arrive one at a time. `src/rag/indexing.py`
now has:

- `index_documents(data_dir)`: unchanged CLI behavior, still wipes and rebuilds everything.
- `index_document_file(path, document_id)`: new, indexes one PDF without touching the rest of the
  collection.
- `ensure_collection(client, recreate=True)`: the API path calls this with `recreate=False`, so it
  only creates the collection if missing.
- `delete_document(document_id)`: deletes a document's chunks via a Qdrant payload filter.

This needed a point-ID scheme that can't collide across documents. Point IDs moved from sequential
integers to `uuid.uuid5(namespace, f"{document_id}:{chunk_index}")`, and every point's payload now
carries a `document_id` field. The API uses the SQLite row's own primary key as `document_id`,
directly linking a Qdrant chunk back to its owning database row, the roadmap's "connect PostgreSQL
document IDs with Qdrant metadata" item. `ingest.py`'s bulk path uses the PDF's filename stem
instead, since there's no DB row involved there.

## Model warm-up

The CLI (`ask.py`) reloads the embedding, sparse, and LLM models on every process invocation, a
known V0 limitation (`docs/v0-notes.md`). The API is long-lived, so its FastAPI
`lifespan` handler loads all three once at startup:

```text
INFO:     Waiting for application startup.
Loading weights: 100%|##########| 199/199 ...   <- embedding + sparse models
Loading weights: 100%|##########| 434/434 ...   <- LLM
INFO:     Application startup complete.
```

After that, every request reuses the already-loaded models. This is the concrete difference
between a script and a service.

## Real run

```text
$ curl http://localhost:8000/health
{"status":"ok","qdrant":true,"database":true}

$ curl -F "file=@data/pytorch.pdf" http://localhost:8000/documents
{"id":1,"filename":"pytorch.pdf","status":"pending","chunk_count":0,...}

$ curl http://localhost:8000/documents
[{"id":1,"filename":"pytorch.pdf","status":"ready","chunk_count":8813,...}]

$ curl -X DELETE http://localhost:8000/documents/1
(204 No Content)
```

`uv run pytest` covers the same lifecycle automatically (upload, poll until ready, query, feedback,
delete, plus 404 cases) against a tiny 2-page test PDF, isolated from the real `qdrant_data`/
`app.db` via env vars set in `tests/conftest.py` before any `rag.*` module is imported.

## Which mode to call

V2 (`docs/v2-notes.md`) found **hybrid** beats both `dense` and `hybrid_reranker` on every metric
on the evaluation set. `config.RETRIEVAL_MODE` still defaults to `dense` as the conservative
baseline, that default wasn't changed as a side effect of building the API, but callers should pass
`"mode": "hybrid"` in `POST /query` for the best results:

```json
{"question": "What is a tensor in PyTorch?", "mode": "hybrid"}
```

## Known limitations

- **No auth.** Not in the V3 roadmap checklist, out of scope here.
- **Background tasks, not a task queue.** `BackgroundTasks` runs in the same process as the API
  server, fine at this scale, would need a real queue (Celery, RQ, etc.) once uploads are frequent
  or the server needs to restart without losing in-flight indexing jobs. Not part of V3's scope.
- **SQLite**, see above, real Postgres arrives in V4.
