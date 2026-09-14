# production-rag
ContextForge — Production RAG &amp; Knowledge Retrieval Platform

A production-oriented Retrieval-Augmented Generation system, built up incrementally from a naive
pipeline to a fully evaluated, observable, production-grade service. See
[production_rag_roadmap.md](production_rag_roadmap.md) for the full plan and the reasoning behind
each stage.

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
`src/rag/config.py` and can be overridden via environment variables — see `.env.example`.

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
