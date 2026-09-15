# production-rag
ContextForge — Production RAG &amp; Knowledge Retrieval Platform

A production-oriented Retrieval-Augmented Generation system, built up incrementally from a naive
pipeline to a fully evaluated, observable, production-grade service. See
[docs/production_rag_roadmap.md](docs/production_rag_roadmap.md) for the full plan and the
reasoning behind each stage.

## V0 — Naive RAG

The current version indexes `data/pytorch.pdf` and answers questions about it end to end:
PDF → chunks → local embeddings → Qdrant → retrieval → local LLM → answer with page citations.

Everything runs locally: embeddings via `sentence-transformers` (BGE), vector storage via an
embedded/on-disk Qdrant instance, and generation via a local Hugging Face model on GPU — no API
keys required.

### Setup

```bash
uv sync
```

### Usage

```bash
uv run python ingest.py             # index data/*.pdf into the local Qdrant store
uv run python ask.py "What is a tensor in PyTorch?"
```

Example output:

```text
Answer:
A tensor in PyTorch is a multi-dimensional matrix or array used for numerical
computations...

Sources:
- pytorch.pdf, page 596
- pytorch.pdf, page 454
```

Configuration (embedding model, LLM model, chunk size/overlap, top-k, storage paths) lives in
`src/rag/config.py` and can be overridden via environment variables, see `.env.example`.

See [docs/v0-retrieval-notes.md](docs/v0-retrieval-notes.md) for observed dense-only retrieval
behavior and limitations.

## V1 — Better Retrieval

Dense-only search struggles on exact-term queries (API names, error codes). V1 adds BM25 sparse
retrieval as a second named vector in the same Qdrant collection (via `fastembed`'s `Qdrant/bm25`
model), fuses it with dense search using Qdrant's built-in Reciprocal Rank Fusion, and adds a
cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over the fused candidates.

Three retrieval modes are selectable at query time:

```bash
uv run python ask.py "torch.nn.Conv2d" --mode dense              # baseline, semantic only
uv run python ask.py "torch.nn.Conv2d" --mode hybrid             # dense + BM25, RRF-fused
uv run python ask.py "torch.nn.Conv2d" --mode hybrid_reranker    # hybrid, then cross-encoder reranked
```

`RETRIEVAL_MODE` in `.env`/`config.py` sets the default (`dense`). `CANDIDATE_K` controls how many
fused candidates are pulled before reranking.

To compare all three modes across a fixed set of semantic and exact-keyword queries:

```bash
uv run python compare_modes.py
```

See [docs/v1-retrieval-notes.md](docs/v1-retrieval-notes.md) for observed trade-offs between the
retrieval modes and between chunk sizes.

## V2 — Evaluation-Driven RAG

V1 compared retrieval modes on a handful of hand-picked queries, eyeballed manually. V2 replaces
that with a labeled evaluation set and real metrics: Recall@K, MRR, nDCG, context precision/recall,
retrieval latency, and LLM-judged answer faithfulness/relevancy, computed for all three retrieval
modes.

```bash
uv run python evaluation/build_dataset.py   # generate the evaluation set (30 synthetic Q&A pairs)
uv run python evaluation/run_eval.py        # run retrieval + generation eval, write results.csv
```

Result: **hybrid** beats both dense and hybrid_reranker on every metric in this evaluation
(Recall@5 0.83 vs 0.73, plus better faithfulness/relevancy, at lower latency than the reranked
mode), so it's the recommended default. See [docs/v2-notes.md](docs/v2-notes.md) for the full
comparison table and interpretation, including the reranker's surprising underperformance here.

## V3 — Production API

V0-V2 are scripts. V3 turns the pipeline into a FastAPI service backed by SQLite (real PostgreSQL
arrives in V4 alongside Docker Compose), so documents can be uploaded and queried over HTTP instead
of the CLI.

```bash
uv run python serve.py    # starts the API on http://localhost:8000, Swagger UI at /docs
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Qdrant and database reachability |
| `POST /documents` | upload a PDF, indexed in the background |
| `GET /documents` | list documents and their status |
| `DELETE /documents/{id}` | remove a document |
| `POST /query` | ask a question, get an answer with sources |
| `POST /feedback` | rate an answer by its `query_id` |

```bash
curl -F "file=@data/pytorch.pdf" http://localhost:8000/documents
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a tensor in PyTorch?", "mode": "hybrid"}'
```

Pass `"mode": "hybrid"` per V2's evidence that it's the best-performing configuration.

Run the API integration tests with:

```bash
uv run pytest
```

See [docs/v3-notes.md](docs/v3-notes.md) for the SQLite/Postgres decision, the incremental
indexing design, and the model warm-up win over the CLI.

## Docs

- [docs/code-walkthrough.md](docs/code-walkthrough.md): what each module and function does.
- [docs/v0-retrieval-notes.md](docs/v0-retrieval-notes.md): dense-only retrieval baseline.
- [docs/v1-retrieval-notes.md](docs/v1-retrieval-notes.md): dense vs hybrid vs reranked, chunk-size trade-offs.
- [docs/v2-notes.md](docs/v2-notes.md): evaluation methodology, results table, and default configuration choice.
- [docs/v3-notes.md](docs/v3-notes.md): production API design decisions.
