# Code walkthrough

What each part of the codebase does, function by function. Written for V0 (naive RAG) and V1
(better retrieval).

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
  llm.py         local Hugging Face model, generates the final answer
  pipeline.py    glues retrieval + generation together

ingest.py         CLI: index documents
ask.py            CLI: ask a question
compare_modes.py  CLI: compare retrieval modes side by side
```

## config.py

Every tunable value lives here as a module-level constant, read from an environment variable with
a default. Nothing else in the codebase hardcodes a model name or a number, so behavior changes by
setting env vars (or `.env`), not by editing code.

| Constant | Default | Used for |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | dense embedding model |
| `LLM_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | local generation model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 120 | text splitter settings |
| `TOP_K` | 5 | final number of chunks returned to the LLM |
| `CANDIDATE_K` | 20 | candidate pool size before reranking |
| `RETRIEVAL_MODE` | `dense` | default mode for `retrieve()` |
| `SPARSE_MODEL` | `Qdrant/bm25` | fastembed sparse model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | cross-encoder model |
| `QDRANT_PATH` / `QDRANT_COLLECTION` | `./qdrant_data` / `documents` | embedded Qdrant storage |
| `DATA_DIR` | `./data` | where source PDFs live |

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
  differently (see the note in `docs/v1-retrieval-notes.md`): document vectors carry a per-term
  weight, query vectors are unweighted (all values are 1), and Qdrant applies IDF weighting at
  query time via the `Modifier.IDF` setting on the collection.

## indexing.py

- `load_documents(data_dir=config.DATA_DIR)`: finds every `*.pdf` in `data_dir`, loads it with
  LangChain's `PyPDFLoader` (one `Document` per page), and stamps each page's metadata with
  `source` (the filename) and `page` (1-indexed, since `pypdf` numbers pages from 0).
- `chunk_documents(documents)`: runs LangChain's `RecursiveCharacterTextSplitter` over the pages,
  using `config.CHUNK_SIZE` / `config.CHUNK_OVERLAP`. Each output chunk keeps its parent page's
  `source`/`page` metadata.
- `get_qdrant_client()`: returns a `QdrantClient` pointed at `config.QDRANT_PATH`, an on-disk
  folder. No server process involved, the client reads/writes the files directly.
- `ensure_collection(client)`: drops the collection if it exists and recreates it with **two**
  named vectors: `"dense"` (size = embedding model's dimension, cosine distance) and `"sparse"`
  (with `modifier=Modifier.IDF`, telling Qdrant to apply BM25-style IDF weighting at query time).
  Recreating from scratch on every run keeps ingestion idempotent, no migration logic needed.
- `index_documents(data_dir=config.DATA_DIR) -> int`: the main entry point. Loads, chunks, embeds
  (both dense and sparse), and upserts every chunk as a `PointStruct` whose `vector` field carries
  both named vectors and whose `payload` carries `text`/`source`/`page`. Returns how many chunks
  were indexed.

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
- `generate_answer(question, context, max_new_tokens=400)`: builds a two-message chat prompt
  (a system prompt instructing the model to answer only from context, plus a user message with the
  context and question), formats it with the model's own chat template
  (`tokenizer.apply_chat_template`), and greedily generates (`do_sample=False`, so answers are
  deterministic) up to `max_new_tokens` new tokens. Returns just the newly generated text, with the
  echoed prompt stripped off.

## pipeline.py

- `Answer`: a small dataclass bundling the generated `answer` text with the `sources` that were
  retrieved for it.
- `answer_question(question, mode=config.RETRIEVAL_MODE, top_k=config.TOP_K)`: calls
  `retrieve()`, joins the retrieved chunks into a single context string (each chunk prefixed with
  its `[source, page N]` so the LLM can see where each snippet came from), calls
  `generate_answer()`, and returns an `Answer`. This is the one function both `ask.py` and any
  future API layer (V3) would call.

## ask.py (CLI)

Parses one positional argument (`question`) and one optional flag, `--mode
{dense,hybrid,hybrid_reranker}` (default from `config.RETRIEVAL_MODE`). Calls
`answer_question()`, prints the answer, then prints each unique `(source, page)` with its score,
so retrieval quality is visible without needing a separate script.

## ingest.py (CLI)

One line: calls `index_documents()` and prints how many chunks were indexed. All the real logic
lives in `rag.indexing` so it can be imported and reused (e.g. by tests, or later by an API).

## compare_modes.py (CLI)

Defines two small fixed query lists, `SEMANTIC_QUERIES` and `KEYWORD_QUERIES`. For each query it
calls `retrieve()` once per mode in `VALID_MODES` and prints the top 5 hits (score, page, a text
snippet) per mode, so the three modes can be eyeballed side by side on the same question. This is
what produced the numbers in `docs/v1-retrieval-notes.md`.
