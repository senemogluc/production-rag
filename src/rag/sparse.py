from functools import lru_cache

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

from rag import config


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(config.SPARSE_MODEL)


def embed_sparse_texts(texts: list[str]) -> list[SparseVector]:
    model = get_sparse_model()
    return [
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in model.embed(texts)
    ]


def embed_sparse_query(text: str) -> SparseVector:
    model = get_sparse_model()
    e = next(model.query_embed(text))
    return SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
