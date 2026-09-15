import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from rag import config
from rag.embeddings import embed_texts, embedding_dimension
from rag.sparse import embed_sparse_texts

POINT_ID_NAMESPACE = uuid.UUID("f6a7f6a0-1c1a-4b3f-9c9b-6f5f8f7a2b10")


def load_document(path: Path | str):
    path = Path(path)
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    for page in pages:
        page.metadata["source"] = path.name
        page.metadata["page"] = page.metadata.get("page", 0) + 1
    return pages


def load_documents(data_dir: str = config.DATA_DIR):
    documents = []
    for pdf_path in sorted(Path(data_dir).glob("*.pdf")):
        documents.extend(load_document(pdf_path))
    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(path=config.QDRANT_PATH)


def ensure_collection(client: QdrantClient, recreate: bool = True):
    exists = client.collection_exists(config.QDRANT_COLLECTION)
    if recreate and exists:
        client.delete_collection(config.QDRANT_COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config={
                "dense": VectorParams(size=embedding_dimension(), distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(modifier=Modifier.IDF),
            },
        )


def _point_id(document_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(POINT_ID_NAMESPACE, f"{document_id}:{chunk_index}"))


def index_chunks(client: QdrantClient, chunks, document_id: str) -> int:
    if not chunks:
        return 0

    texts = [chunk.page_content for chunk in chunks]
    dense_vectors = embed_texts(texts)
    sparse_vectors = embed_sparse_texts(texts)

    points = [
        PointStruct(
            id=_point_id(document_id, i),
            vector={"dense": dense_vector, "sparse": sparse_vector},
            payload={
                "text": chunk.page_content,
                "source": chunk.metadata.get("source"),
                "page": chunk.metadata.get("page"),
                "document_id": document_id,
            },
        )
        for i, (chunk, dense_vector, sparse_vector) in enumerate(
            zip(chunks, dense_vectors, sparse_vectors)
        )
    ]
    client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
    return len(points)


def index_documents(data_dir: str = config.DATA_DIR) -> int:
    """Bulk (re)index every PDF under data_dir. Wipes and rebuilds the whole collection."""
    client = get_qdrant_client()
    ensure_collection(client, recreate=True)

    total = 0
    for pdf_path in sorted(Path(data_dir).glob("*.pdf")):
        chunks = chunk_documents(load_document(pdf_path))
        total += index_chunks(client, chunks, document_id=pdf_path.stem)
    return total


def index_document_file(path: Path | str, document_id: str) -> int:
    """Incrementally index a single PDF without touching the rest of the collection."""
    client = get_qdrant_client()
    ensure_collection(client, recreate=False)

    chunks = chunk_documents(load_document(path))
    return index_chunks(client, chunks, document_id=document_id)


def delete_document(document_id: str):
    client = get_qdrant_client()
    client.delete(
        collection_name=config.QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        ),
    )


if __name__ == "__main__":
    count = index_documents()
    print(f"Indexed {count} chunks into Qdrant collection '{config.QDRANT_COLLECTION}'.")
