from functools import lru_cache

from sentence_transformers import CrossEncoder

from rag import config
from rag.types import RetrievedChunk


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(config.RERANKER_MODEL)


def rerank(question: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if not chunks:
        return []

    model = get_reranker()
    pairs = [(question, chunk.text) for chunk in chunks]
    scores = model.predict(pairs)

    reranked = [
        RetrievedChunk(text=c.text, source=c.source, page=c.page, score=float(s))
        for c, s in zip(chunks, scores)
    ]
    reranked.sort(key=lambda c: c.score, reverse=True)
    return reranked[:top_k]
