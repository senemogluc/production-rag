from qdrant_client.models import Fusion, FusionQuery, Prefetch

from rag import config, tracing
from rag.embeddings import embed_query
from rag.indexing import get_qdrant_client
from rag.reranker import rerank
from rag.sparse import embed_sparse_query
from rag.types import RetrievedChunk

VALID_MODES = ("dense", "hybrid", "hybrid_reranker")


def _to_chunk(point) -> RetrievedChunk:
    return RetrievedChunk(
        text=point.payload["text"],
        source=point.payload["source"],
        page=point.payload["page"],
        score=point.score,
    )


def _retrieve_dense(question: str, top_k: int) -> list[RetrievedChunk]:
    client = get_qdrant_client()
    query_vector = embed_query(question)

    results = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=query_vector,
        using="dense",
        limit=top_k,
    ).points

    return [_to_chunk(point) for point in results]


def _retrieve_hybrid(question: str, candidate_k: int) -> list[RetrievedChunk]:
    client = get_qdrant_client()
    dense_vector = embed_query(question)
    sparse_vector = embed_sparse_query(question)

    results = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        prefetch=[
            Prefetch(query=dense_vector, using="dense", limit=candidate_k),
            Prefetch(query=sparse_vector, using="sparse", limit=candidate_k),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=candidate_k,
    ).points

    return [_to_chunk(point) for point in results]


def retrieve(
    question: str,
    mode: str = config.RETRIEVAL_MODE,
    top_k: int = config.TOP_K,
) -> list[RetrievedChunk]:
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown retrieval mode '{mode}'. Expected one of {VALID_MODES}.")

    with tracing.observation(
        "retrieval",
        as_type="retriever",
        input=question,
        metadata={"mode": mode, "top_k": top_k},
    ) as obs:
        if mode == "dense":
            results = _retrieve_dense(question, top_k)
        elif mode == "hybrid":
            results = _retrieve_hybrid(question, top_k)
        else:
            # hybrid_reranker
            candidates = _retrieve_hybrid(question, config.CANDIDATE_K)
            results = rerank(question, candidates, top_k)

        if obs:
            obs.update(
                output=[
                    {"source": c.source, "page": c.page, "score": c.score} for c in results
                ]
            )
        return results
