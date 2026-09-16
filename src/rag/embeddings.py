from functools import lru_cache

from sentence_transformers import SentenceTransformer

from rag import config, tracing


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    with tracing.observation(
        "dense-embedding",
        as_type="embedding",
        model=config.EMBEDDING_MODEL,
        metadata={"n_texts": len(texts)},
    ):
        model = get_embedding_model()
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def embedding_dimension() -> int:
    return get_embedding_model().get_embedding_dimension()
