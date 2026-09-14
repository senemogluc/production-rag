# V1 retrieval notes

Observations from running [compare_modes.py](../compare_modes.py) against `data/pytorch.pdf`
(2,853 pages) across the three retrieval modes and three chunk sizes. All examples below are
real output captured while building V1, not illustrative/fabricated numbers.

## Dense vs. hybrid vs. hybrid + reranker

**Dense retrieval** (cosine similarity on BGE embeddings) is a strong semantic baseline — for
"How does autograd compute gradients?" it surfaces the right chapter (p.984, "AUTOMATIC
DIFFERENTIATION PACKAGE") at rank 1 with score 0.84+. Its main weakness shows up on exact-API
queries: for `torch.nn.Conv2d` at the default chunk size, dense-only's top hits are all about the
*quantized* `Conv2d` variant (`torch.ao.nn.quantized.Conv2d`, p.2501) rather than the plain
`torch.nn.Conv2d` class — semantically close, but not the literal symbol asked for.

**Hybrid** (Qdrant-native RRF fusion of the dense vector and a `Qdrant/bm25` sparse vector, both
stored as named vectors in the same collection) consistently pulls the exact-term match back to
rank 1. For `torch.nn.Conv2d` it promotes p.567 (`torch.nn` module reference) and, at 400-token
chunks, the actual `class torch.nn.Conv2d(in_channels, out_channels, ...)` signature (p.553) —
neither of which dense-only ranked in its top 5. RRF scores compress into a small set of discrete
values (1/rank sums), so hybrid scores are far less "spread out" than raw cosine similarity —
useful for combining two incompatible scales, but not informative as a standalone confidence
number.

**Hybrid + reranker** (cross-encoder `ms-marco-MiniLM-L-6-v2` over the top-20 RRF candidates,
cut to top-5) reliably reorders those 20 candidates toward the passage that actually *answers*
the question rather than just mentioning the terms. For "How does autograd compute gradients?" it
promoted p.46 ("AUTOGRAD MECHANICS... gradients using the chain rule") to rank 1 — a page neither
dense nor hybrid had ranked first. This came at a real cost: the reranker only reorders whatever
the 20-candidate RRF pool contains, so if a genuinely relevant page never enters the candidate
pool, no amount of reranking recovers it. In this corpus, that pool was large enough (`CANDIDATE_K
= 20`) that the answer was consistently present.

**Trade-off in one sentence:** dense retrieval optimizes for topical similarity, sparse (BM25)
retrieval optimizes for exact terms, and the reranker optimizes ordering *within* whatever the
first two stages already found — it cannot invent recall the earlier stages didn't provide.

## Chunk size (400 / 800 / 1600 tokens, tested via `CHUNK_SIZE`/`CHUNK_OVERLAP`)

Re-ingesting the same PDF at three chunk sizes produced very different corpus sizes:

| Chunk size | Overlap | Chunks indexed |
|---|---|---|
| 400 | 60 | 16,447 |
| 800 (default) | 120 | 8,813 |
| 1600 | 240 | 4,836 |

- **400-token chunks** gave the most *precise* exact-match hits — for `torch.nn.Conv2d`, this was
  the only chunk size where the literal class signature (`class torch.nn.Conv2d(in_channels,
  out_channels, kernel_size, ...)`, p.553) appeared directly in the dense/hybrid results, because
  the signature wasn't diluted by surrounding prose in the same chunk.
- **1600-token chunks** produced the *worst* result diversity: with a large `chunk_overlap` (240)
  relative to fewer, bigger chunks, the same underlying passage was frequently split across two
  overlapping chunks that both scored high, so top-5 results for `CUDA_ERROR_OUT_OF_MEMORY` and
  `torch.nn.Conv2d` contained literal duplicate/near-duplicate entries (e.g. p.1113 or p.2495
  appearing twice in the same top-5). Larger chunks also picked up more surrounding boilerplate
  (page headers like "PyTorch Documentation, Release main"), which is wasted context passed to the
  LLM.
- **800-token chunks (the default)** sat in between: fewer duplicate-passage artifacts than 1600,
  and still specific enough for good exact-match retrieval once hybrid/reranking is applied,
  without the ~2x index size of the 400-token setting.

**Takeaway:** smaller chunks trade index size and per-chunk context for retrieval precision;
larger chunks trade precision for richer context per hit, but overlap needs to shrink
proportionally or it manufactures near-duplicate top-K results. 800/120 is kept as the default for
this corpus; 400/60 would be the next thing to formally A/B once V2's evaluation harness exists
(this is a qualitative read, not a Recall@K/nDCG comparison — that's explicitly deferred to V2).

## What this doesn't tell us yet

None of the above is backed by labeled relevance judgments — it's manual inspection of
`compare_modes.py` output on a handful of queries, which is exactly what the roadmap's V1 section
expects ("Compare the same queries across retrieval modes... Document retrieval trade-offs")
before V2 turns it into Recall@K / MRR / nDCG numbers over a real evaluation set.
