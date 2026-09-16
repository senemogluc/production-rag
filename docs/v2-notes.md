# V2 evaluation notes

Real results from `evaluation/run_eval.py`, not illustrative numbers. See
`evaluation/dataset.json` for the 30 questions and `evaluation/results.csv` for the raw table.

## How the dataset was built

Hand-writing 30-50 accurate question/page pairs against a 2,853-page PDF isn't reliable to do by
hand. Instead, `evaluation/build_dataset.py` samples 30 chunks at random (seeded, reproducible)
from the indexed Qdrant collection, filters out short/boilerplate ones, and asks the local LLM to
write one question answerable from each chunk plus a short reference answer. The sampled chunk's
`(source, page)` becomes the gold label automatically. This avoids hallucinated citations, but has
a real side effect: some generated questions retain phrases like "in the given example" or "in the
given code snippet" instead of being fully self-contained, since a 3B model doesn't perfectly
follow that instruction. Spot-checked a handful against their gold page and they're genuinely
answerable from it.

A retrieved chunk counts as relevant if it matches the gold chunk's `(source, page)`, not only the
exact chunk id, since chunk boundaries are an arbitrary 800-character windowing artifact.

## Results

| Mode | Recall@5 | MRR | nDCG@5 | Context precision | Faithfulness | Relevancy | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.733 | 0.536 | 0.586 | 0.200 | 4.00 | 4.13 | 1007 |
| hybrid | **0.833** | **0.623** | **0.676** | **0.240** | **4.30** | **4.33** | **991** |
| hybrid_reranker | 0.733 | 0.625 | 0.652 | 0.233 | 4.10 | 4.27 | 1985 |

(Faithfulness and relevancy are LLM-judged, 1-5. Latency is retrieval only, not full
generation.)

## What this confirms

**Hybrid beats dense on every retrieval and generation metric**, not just the handful of queries
eyeballed in `docs/v1-notes.md`. Recall@5 goes from 0.733 to 0.833, meaning hybrid finds
the correct page in its top 5 for 25 of 30 questions versus 22 of 30 for dense. Faithfulness and
relevancy both improve too: better retrieval gives the LLM better context to answer from. Hybrid is
also marginally *faster* than dense here (991ms vs 1007ms), within noise.

## What this contradicts

**The reranker does not help on this evaluation set**, and this is the actual point of doing real
measurement instead of trusting intuition. `hybrid_reranker`'s Recall@5 drops back to dense's level
(0.733, down from hybrid's 0.833), and faithfulness/relevancy both drop slightly below plain
hybrid too, while latency roughly doubles (1985ms vs 991ms) from the extra cross-encoder pass.

Two likely reasons, not mutually exclusive:

1. **Small eval set.** 30 questions is enough to see a real signal (hybrid's improvement over dense
   is consistent across every column), but not enough to fully trust a difference this size between
   hybrid and hybrid_reranker. Some of this could be noise.
2. **Synthetic questions may favor lexical/semantic overlap over cross-encoder relevance.** The
   questions were generated directly from their gold chunk's text, so they tend to share vocabulary
   with that exact chunk, which is exactly what dense/BM25/RRF fusion are good at matching. The
   cross-encoder was trained on general passage-relevance judgments (MS MARCO), not on
   "which chunk literally produced this question", so it sometimes reorders toward a different,
   also-plausible chunk instead of the one true source. That's a property of this evaluation
   method, not necessarily of the reranker in general.

Either way: the roadmap explicitly warns not to assume a more sophisticated pipeline stage is
automatically better ("the retriever aims for high recall, the reranker aims for better precision
and ordering"). Here it didn't improve ordering *for this task*, and adding it cost real latency.

## Default configuration

**hybrid** is the recommended default: best Recall@5, MRR, nDCG, context precision, faithfulness,
and relevancy of the three, and the lowest latency. `hybrid_reranker` stays available as a mode
(`ask.py --mode hybrid_reranker`) since a larger evaluation set or a different reranker model could
change this conclusion, but it is not the default given current evidence.
`config.RETRIEVAL_MODE` stays `dense` as the safe baseline unless explicitly overridden; changing
the repo-wide default to `hybrid` is a follow-up, not bundled into this evaluation change.

## Known limitations of this evaluation

- **Faithfulness and relevancy are graded by the same local model that generates the answers**
  (`Qwen/Qwen2.5-3B-Instruct` as both generator and judge). This is a real self-grading bias risk,
  not a rigorous claim-decomposition faithfulness score like RAGAS computes. It's the "smallest
  useful version" of LLM-as-judge, given this project runs fully local with no hosted judge API.
- **Context recall and Recall@5 are the same number here.** With exactly one gold-relevant page
  per question, "was the relevant page retrieved at all" and "was it in the top 5" collapse to the
  same computation. Both are reported in `results.csv` for completeness, not because they're
  independent signals in this setup.
- **30 questions**, the low end of the roadmap's suggested 30-50 range, chosen to bound total
  evaluation runtime (~90 generations + 90 judge calls across three modes). A larger set would
  tighten confidence in the hybrid vs hybrid_reranker gap specifically.
