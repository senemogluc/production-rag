# V4 notes

V4 is split into two parts. This covers Part 1 (containerization), done now. Part 2
(observability instrumentation, tests, CI) is queued for later.

## What changed

V0-V3 ran Qdrant embedded (on-disk, no server) and the API against SQLite. V4 introduces real
services via Docker Compose, with the API itself still running on the host.

| Component | Before (V0-V3) | Now (V4) |
|---|---|---|
| Qdrant | embedded, `QdrantClient(path=...)` | real server container, `QdrantClient(url=...)` |
| Database | SQLite file | PostgreSQL container |
| API | host-run | still host-run (see below) |
| Observability | none | self-hosted Langfuse (containers up, tracing code not wired in yet) |

`docker-compose.yml` brings up 8 containers: `qdrant`, `postgres` (the app's own database), and
Langfuse's own stack (`langfuse-web`, `langfuse-worker`, `langfuse-postgres`, `clickhouse`,
`redis`, `minio`), adapted from Langfuse's official compose file.

## Why the API stays host-run

The API needs the GPU for local model inference (embeddings, reranker, the LLM). Passing a GPU
into a Windows Docker Desktop container needs WSL2 GPU passthrough working correctly, unverified
and a real risk area on this machine. Rather than debug that, `docker compose` only brings up the
data services (Qdrant, Postgres, Langfuse); `serve.py`/`ask.py`/etc. keep running directly on the
host via `uv run`, pointed at the containerized services through `QDRANT_URL`/`DATABASE_URL`.

An API `Dockerfile` was still written (multi-stage, `uv`-based) since the roadmap asks for one,
useful for a future GPU-enabled cloud deployment, but it is not part of the default
`docker compose up` stack, and would run on CPU only as written.

## Why self-hosted Langfuse, and what that costs

Two options existed: Langfuse Cloud (lightweight, but sends trace data to a third-party SaaS,
breaking the fully-local story used everywhere else in this project) or self-hosted (heavier, but
consistent with "everything runs locally"). Self-hosted was chosen. The real cost: 6 extra
containers, 8GB+ RAM recommended, 20GB+ disk. This is a legitimate trade-off for a portfolio
project, not something to gloss over.

The Langfuse project and API keys are already set up (`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`
in `.env`). The actual tracing code (spans for retrieval, reranking, generation) is Part 2, not
written yet, so nothing is traced to Langfuse yet even though the containers are running.

## Dual-mode Qdrant client

`get_qdrant_client()` in `src/rag/indexing.py` now checks `config.QDRANT_URL` first:

```python
def get_qdrant_client() -> QdrantClient:
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL)
    return QdrantClient(path=config.QDRANT_PATH)
```

`QDRANT_URL` defaults to `http://localhost:6333` (the compose-exposed port), so the containerized
server is the new default. Blanking `QDRANT_URL` in `.env` falls back to V0-V3's embedded mode, no
Docker required, verified working both ways, not just documented. `tests/test_api.py` forces this
fallback (`QDRANT_URL=""`) so the test suite never depends on a running container.

## Two real bugs found while verifying this

Both only showed up once talking to a real Qdrant/Postgres over the network, embedded mode and
SQLite never exercised these code paths.

1. **Langfuse password mismatch.** `langfuse-postgres`'s password was set via
   `LANGFUSE_POSTGRES_PASSWORD`, but the connection string Langfuse's own services actually use
   (`DATABASE_URL` in their compose, renamed `LANGFUSE_DATABASE_URL` here to avoid colliding with
   the app's own `DATABASE_URL`) still had the literal default password hardcoded. Fixed by
   setting `LANGFUSE_DATABASE_URL` explicitly to match.
2. **Qdrant's HTTP request size limit.** Bulk-upserting all 8,813 chunks in one request hit
   Qdrant server's 32MB request size limit (`~89MB` payload once real vectors and text are
   included). Embedded mode has no such limit since it skips the HTTP layer entirely. Fixed by
   batching `index_chunks()`'s upsert into groups of 256 points, which also benefits the API's
   incremental per-document indexing path.

## Verified working

- `docker compose up -d` brings up all 8 containers, all reach healthy status.
- `uv run python ingest.py` and `uv run python ask.py "..." --mode hybrid` work against the
  containerized Qdrant.
- `uv run python serve.py` + `GET /health` returns `{"status":"ok","qdrant":true,"database":true}`
  against containerized Postgres and Qdrant together.
- Setting `QDRANT_URL=""` switches back to embedded mode and finds the old V0-V3 `qdrant_data/`
  collection, confirming the fallback is real.
- Langfuse UI (`http://localhost:3000`) is reachable, project and API keys are created.

## Still to do (Part 2)

- Instrument `pipeline.py`/`retrieval.py`/`reranker.py`/`llm.py` with Langfuse spans (env-gated,
  no-ops if keys aren't set).
- Unit tests (`evaluation/metrics.py`, point-id determinism) and an evaluation regression test
  against V2's committed baseline.
- GitHub Actions CI workflow.
- README/CLAUDE.md updates and the V4 roadmap checklist.
