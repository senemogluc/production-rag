from functools import lru_cache

from sentence_transformers import CrossEncoder

from rag import config, tracing
from rag.types import RetrievedChunk


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(config.RERANKER_MODEL)


def rerank(question: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if not chunks:
        return []

    with tracing.observation(
        "rerank", as_type="span", metadata={"n_candidates": len(chunks), "top_k": top_k}
    ) as obs:
        model = get_reranker()
        pairs = [(question, chunk.text) for chunk in chunks]
        scores = model.predict(pairs)

        reranked = [
            RetrievedChunk(text=c.text, source=c.source, page=c.page, score=float(s))
            for c, s in zip(chunks, scores)
        ]
        reranked.sort(key=lambda c: c.score, reverse=True)
        reranked = reranked[:top_k]

        if obs:
            obs.update(
                output=[
                    {"source": c.source, "page": c.page, "score": c.score} for c in reranked
                ]
            )
        return reranked
