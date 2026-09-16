# V1 retrieval notes

Observations from running `compare_modes.py` against `data/pytorch.pdf` (2,853 pages), across the
three retrieval modes and three chunk sizes. All numbers below are real output captured while
building V1, not illustrative.

## Dense vs hybrid vs hybrid + reranker

**Dense** (cosine similarity on BGE embeddings) is a strong semantic baseline. For "How does
autograd compute gradients?" it correctly ranks the autograd chapter (p.984) first with score
0.84+. Its weak spot is exact terms: for `torch.nn.Conv2d`, dense-only's top hits were all about
the *quantized* Conv2d variant, not the plain class asked for.

**Hybrid** (Qdrant-native RRF fusion of the dense vector and a `Qdrant/bm25` sparse vector) fixes
that. For `torch.nn.Conv2d` it promotes the actual `torch.nn` module reference to rank 1, and at
smaller chunk sizes surfaces the literal class signature, neither of which dense-only found in its
top 5. One side effect: RRF scores only take a handful of discrete values (they're sums of
`1/rank`), so they're not a useful confidence number on their own, just a fusion ranking signal.

**Hybrid + reranker** (cross-encoder over the top 20 fused candidates, cut to top 5) reorders
those 20 toward the passage that actually answers the question, not just mentions the terms. For
the autograd question it promoted p.46 ("gradients using the chain rule") to rank 1, a page neither
dense nor hybrid had ranked first. Limitation: it can only reorder what's already in the
candidate pool. If a relevant page never makes the top 20 from hybrid search, reranking can't
recover it.

**In one line:** dense retrieval finds topically similar text, BM25 finds exact terms, and the
reranker improves ordering within whatever the first two stages already found. It can't add recall
they didn't provide.

## Chunk size (400 / 800 / 1600 tokens)

Re-ingesting the same PDF at three chunk sizes:

| Chunk size | Overlap | Chunks indexed |
|---|---|---|
| 400 | 60 | 16,447 |
| 800 (default) | 120 | 8,813 |
| 1600 | 240 | 4,836 |

- **400 tokens** gave the most precise exact-match hits. For `torch.nn.Conv2d`, this was the only
  size where the literal class signature appeared directly in results, since it wasn't diluted by
  surrounding prose in the same chunk.
- **1600 tokens** gave the worst result diversity. With overlap scaled up alongside chunk size
  (240), the same passage often got split into two overlapping chunks that both scored high, so
  top-5 results sometimes contained duplicate or near-duplicate entries (e.g. the same page
  appearing twice for `CUDA_ERROR_OUT_OF_MEMORY`). Bigger chunks also pulled in more boilerplate
  (page headers), wasting LLM context.
- **800 tokens (default)** sits in between: fewer duplicate-passage artifacts than 1600, still
  specific enough for good exact-match retrieval once hybrid/reranking is applied, at about half
  the index size of 400.

**Takeaway:** smaller chunks trade index size for precision, larger chunks trade precision for
richer per-chunk context, and overlap needs to shrink proportionally with chunk size or it starts
manufacturing near-duplicate top-K results. 800/120 stays the default here.

## What this doesn't tell us yet

This is manual inspection of a handful of queries, not a labeled benchmark. It's what the roadmap's
V1 section asks for (compare modes, test semantic and exact-keyword queries, document trade-offs).
Turning it into real numbers (Recall@K, MRR, nDCG) is V2's job.
