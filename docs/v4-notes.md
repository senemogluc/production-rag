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
| Observability | none | Langfuse Cloud (free tier, keys set, tracing code not wired in yet) |

`docker-compose.yml` brings up 2 containers: `qdrant` and `postgres`, the app's own database.

## Why the API stays host-run

The API needs the GPU for local model inference (embeddings, reranker, the LLM). Passing a GPU
into a Windows Docker Desktop container needs WSL2 GPU passthrough working correctly, unverified
and a real risk area on this machine. Rather than debug that, `docker compose` only brings up the
data services (Qdrant, Postgres, Langfuse); `serve.py`/`ask.py`/etc. keep running directly on the
host via `uv run`, pointed at the containerized services through `QDRANT_URL`/`DATABASE_URL`.

An API `Dockerfile` was still written (multi-stage, `uv`-based) since the roadmap asks for one,
useful for a future GPU-enabled cloud deployment, but it is not part of the default
`docker compose up` stack, and would run on CPU only as written.

## Why Langfuse Cloud, not self-hosted

Self-hosted Langfuse was tried first: `langfuse-web`, `langfuse-worker`, and four supporting
services (`langfuse-postgres`, `clickhouse`, `redis`, `minio`), adapted from Langfuse's official
compose file. It worked (see the password-mismatch bug below, found and fixed while getting it
running), but stepping back, it was 6 extra containers, 8GB+ RAM, 20GB+ disk, just to trace
queries for a single-developer local project. That's real infrastructure for a problem that
doesn't need it here.

Switched to Langfuse Cloud's free Hobby tier instead: 50k units/month (a unit is roughly one
trace/span/score, so tens of thousands of queries/month of headroom), no credit card, a 30-day
rolling retention window (data stays accessible for 30 days after being sent, then ages out,
sending never stops). The containers were torn down (`docker compose rm`, volumes removed), and
`docker-compose.yml` now only has `qdrant` and `postgres`. The trade-off: trace data (questions,
answers, retrieval scores, latencies) now leaves the machine, the one part of this project that
isn't fully local. Given traces are operational metadata, not the retrieval/generation pipeline
itself (which stays 100% local, no API keys, same as V0), this was judged an acceptable, honestly
documented exception, not a silent compromise.

The Langfuse project and API keys are set up (`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` in
`.env`, pointing at `https://cloud.langfuse.com`). The actual tracing code (spans for retrieval,
reranking, generation) is Part 2, not written yet, so nothing is traced yet even though the keys
are configured.

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

## Real bugs found while verifying this

Only showed up once talking to a real Qdrant over the network, embedded mode never exercised this
code path: bulk-upserting all 8,813 chunks in one request hit Qdrant server's 32MB HTTP request
size limit (`~89MB` payload once real vectors and text are included). Fixed by batching
`index_chunks()`'s upsert into groups of 256 points, which also benefits the API's incremental
per-document indexing path.

(While self-hosted Langfuse was still in the stack, a similar class of bug showed up there too:
`langfuse-postgres`'s password was set via one env var but the connection string Langfuse's own
services actually used still had the default hardcoded, fixed by setting it explicitly. Moot now
that self-hosted Langfuse is gone, kept here only as the reason that detour is worth mentioning at
all: real infrastructure surfaces real integration bugs, which is itself part of why it turned out
to be more than this project needed.)

## Verified working

- `docker compose up -d` brings up `qdrant` and `postgres`, both reach healthy status.
- `uv run python ingest.py` and `uv run python ask.py "..." --mode hybrid` work against the
  containerized Qdrant.
- `uv run python serve.py` + `GET /health` returns `{"status":"ok","qdrant":true,"database":true}`
  against containerized Postgres and Qdrant together.
- Setting `QDRANT_URL=""` switches back to embedded mode and finds the old V0-V3 `qdrant_data/`
  collection, confirming the fallback is real.
- Langfuse Cloud project and API keys are created and in `.env`.

## Still to do (Part 2)

- Instrument `pipeline.py`/`retrieval.py`/`reranker.py`/`llm.py` with Langfuse spans (env-gated,
  no-ops if keys aren't set).
- Unit tests (`evaluation/metrics.py`, point-id determinism) and an evaluation regression test
  against V2's committed baseline.
- GitHub Actions CI workflow.
- README/CLAUDE.md updates and the V4 roadmap checklist.
