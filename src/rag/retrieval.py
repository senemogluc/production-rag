from dataclasses import dataclass

from rag import config
from rag.embeddings import embed_query
from rag.ingest import get_qdrant_client


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float


def retrieve(question: str, top_k: int = config.TOP_K) -> list[RetrievedChunk]:
    client = get_qdrant_client()
    query_vector = embed_query(question)

    results = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
    ).points

    return [
        RetrievedChunk(
            text=point.payload["text"],
            source=point.payload["source"],
            page=point.payload["page"],
            score=point.score,
        )
        for point in results
    ]
