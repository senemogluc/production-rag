# V0 retrieval notes

What dense-only retrieval looked like before V1 added BM25, fusion, and reranking. Kept as a
baseline reference.

## Setup

- Corpus: `data/pytorch.pdf`, 2,853 pages (a full LaTeX-compiled PyTorch reference, much bigger
  than it looks from the filename).
- Chunking: 800 tokens, 120 overlap, giving 8,813 chunks.
- Retrieval: cosine similarity search over BGE dense embeddings only, top 5.
- Generation: local `Qwen/Qwen2.5-3B-Instruct`, answering strictly from the retrieved chunks
  (per the system prompt in `llm.py`).

## What worked well

On-topic, semantically phrased questions retrieved the right pages reliably. For example, "What is
a tensor in PyTorch?" pulled in the tensor documentation pages and produced an accurate,
well-cited answer on the first try. Dense embeddings are good at this: the question and the answer
don't need to share exact wording, just meaning.

## Where it broke down

Two real limitations showed up during manual testing:

1. **Exact-term queries.** Asking about a specific API name (e.g. `torch.nn.Conv2d`) sometimes
   retrieved semantically related but not literally matching content, such as the *quantized*
   Conv2d variant instead of the plain one. Dense embeddings compare meaning, not exact tokens, so
   this is expected. This is exactly the gap V1's BM25 hybrid search closes.

2. **No grounding check.** Asked an out-of-domain question ("What is the capital of France?"),
   the model answered "Paris" instead of saying the retrieved context didn't cover it, even though
   the system prompt says to only use the provided context. The retriever did its job (it returned
   *something*, just irrelevant chunks), but the LLM fell back on its own pretrained knowledge
   instead of admitting it. This is a faithfulness problem, not a retrieval problem, and it's
   explicitly out of scope for V0 and V1. The roadmap defers measuring and fixing this to V2's
   evaluation stage (faithfulness scoring).

## Why this mattered for V1

These two gaps, exact-term recall and unmeasured faithfulness, are the concrete reasons the
roadmap introduces hybrid retrieval and reranking next. V0 proved the pipeline works end to end;
it didn't prove the retrieved chunks were the *best* chunks available. See
`docs/v1-retrieval-notes.md` for how hybrid search and reranking changed these results.
