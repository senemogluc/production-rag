# V4 notes

V4 was split into two parts: Part 1 (containerization) and Part 2 (observability, testing, CI).
Both are done. This doc covers both.

## What changed

V0-V3 ran Qdrant embedded (on-disk, no server) and the API against SQLite. V4 introduces real
services via Docker Compose, with the API itself still running on the host.

| Component | Before (V0-V3) | Now (V4) |
|---|---|---|
| Qdrant | embedded, `QdrantClient(path=...)` | real server container, `QdrantClient(url=...)` |
| Database | SQLite file | PostgreSQL container |
| API | host-run | still host-run (see below) |
| Observability | none | Langfuse Cloud (free tier), full trace on every query |

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
`.env`, pointing at `https://cloud.langfuse.com`, note: the SDK reads `LANGFUSE_BASE_URL` as the
current variable name, `LANGFUSE_HOST` still works as a legacy fallback).

## Tracing architecture

`src/rag/tracing.py` is a thin wrapper around the `langfuse` SDK's `get_client()`:

```python
ENABLED = bool(config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY)

@contextmanager
def observation(name, as_type="span", **kwargs):
    if not ENABLED:
        yield None
        return
    with _client.start_as_current_observation(name=name, as_type=as_type, **kwargs) as obs:
        yield obs
```

Every call site guards with `if obs:` before calling `.update(...)`, so the exact same code path
runs whether tracing is on or off, no separate "traced" and "untraced" versions of any function.
Blanking `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` (as `tests/api/conftest.py` does) disables
tracing completely with zero behavior change elsewhere.

`pipeline.answer_question()` opens the root observation (`as_type="chain"`), since it's the one
function both `ask.py` and `POST /query` call through. Anything opened inside that call
automatically nests under it via OpenTelemetry context propagation, no trace/span ids passed by
hand:

- `embeddings.embed_texts()`: `as_type="embedding"`
- `retrieval.retrieve()`: `as_type="retriever"`, records the mode and every retrieved
  `(source, page, score)`
- `reranker.rerank()`: `as_type="span"`, records candidate count and the reranked scores
- `llm.generate()`: `as_type="generation"`, records the model, prompt, output, and token counts
  (computed from `inputs["input_ids"].shape` / the generated token count, since a local
  `transformers` model doesn't report usage/cost like a hosted API would)

Verified with a real query (`ask.py "How do you move a model to the GPU?" --mode hybrid`), pulled
back from Langfuse's API right after:

```text
CHAIN       rag.answer_question   30.2s
  RETRIEVER retrieval              7.1s
  EMBEDDING dense-embedding        5.9s
  GENERATION generation            9.4s
```

Every roadmap latency-breakdown item (embedding, retrieval, reranking, generation, total request)
is satisfied by this, Langfuse tracks each observation's start/end time automatically, no manual
`latency_ms` bookkeeping needed anywhere in the app code.

## Testing: a real isolation bug, and the fix

`tests/api/` (moved here from flat `tests/`) isolates the API test suite by setting
`QDRANT_PATH`/`DATABASE_URL`/etc. via `os.environ` at `conftest.py`'s **module** level, before
`rag.config` is ever imported (it reads env vars once, at import time, and caches them as plain
module constants). This worked fine in V3.

V4 added `tests/test_eval_regression.py`, which needs to see the *real*, fully-indexed corpus, the
opposite of what `tests/api/conftest.py` sets up. Running both in one `pytest` invocation
(`pytest tests/`) actually broke this: pytest collects `tests/api/` first (alphabetically before
the top-level `test_*.py` files), so `tests/api/conftest.py`'s isolated, empty `QDRANT_PATH` gets
baked into `rag.config` before the regression test's module even imports it, since it's the same
Python process. The result wasn't an error, it was a *silent* wrong answer: Recall@5 measured as
`0.000` instead of the real ~0.83, which would have looked exactly like a real regression, not a
test bug.

Fix: two separate `pytest` invocations, always. `.github/workflows/ci.yml`, `README.md`, and
`CLAUDE.md` all run `pytest tests/ --ignore=tests/api` and `pytest tests/api/` as separate steps,
never combined. Both `tests/api/conftest.py` and `tests/test_eval_regression.py` carry an explicit
comment explaining why, so this doesn't get silently reintroduced later.

## Evaluation regression test

`tests/test_eval_regression.py` runs `evaluation.retrieval_eval.evaluate_retrieval()` for
`mode="hybrid"` against the live index and asserts Recall@5 and MRR stay within 0.15 of V2's
committed baseline (0.833 and 0.623). Retrieval-only, no LLM generation, so it's fast enough for
CI. The tolerance is deliberately loose, a 30-question synthetic eval set has real variance run to
run, this test is meant to catch a *large* regression (a broken chunking change, a fusion bug),
not to reproduce V2's exact numbers.

## CI

`.github/workflows/ci.yml`: checkout, `uv sync`, `uv run python ingest.py` (embedded Qdrant,
`data/pytorch.pdf`, already committed to the repo), then the two separate `pytest` steps above. No
service containers, no Langfuse (keys aren't set in CI, so tracing silently no-ops), a few minutes
total. This mirrors the same "smallest useful version" reasoning used throughout the project: CI
proves the application logic and retrieval quality don't regress, it doesn't re-prove the
`docker-compose` networking works, that's covered by the manual verification below.

One accepted cost: `torch` is pinned to a CUDA build (`pytorch-cu126` index, ~2.4GB) for local GPU
use, and CI installs the exact same lockfile, so every CI run downloads that same large wheel even
though the GitHub-hosted runner has no GPU (it just runs on CPU, `llm.py` already handles that
fallback). Maintaining a second, CPU-only dependency set just for CI was judged not worth the
added complexity for this project's scale, `astral-sh/setup-uv`'s cache mitigates the repeat cost.

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
Docker required, verified working both ways, not just documented. `tests/api/conftest.py` forces
this fallback (`QDRANT_URL=""`) so the test suite never depends on a running container.

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
- A real query traced end to end, pulled back from Langfuse's API and confirmed correct: root
  chain, nested retrieval/embedding/generation spans, real latencies and token counts (see above).
- `uv run pytest tests/ --ignore=tests/api` and `uv run pytest tests/api/` both pass, run
  separately as documented.
