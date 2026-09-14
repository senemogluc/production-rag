from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag import config
from rag.embeddings import embed_texts, embedding_dimension


def load_documents(data_dir: str = config.DATA_DIR):
    documents = []
    for pdf_path in sorted(Path(data_dir).glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = pdf_path.name
            page.metadata["page"] = page.metadata.get("page", 0) + 1
        documents.extend(pages)
    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(path=config.QDRANT_PATH)


def ensure_collection(client: QdrantClient):
    if client.collection_exists(config.QDRANT_COLLECTION):
        client.delete_collection(config.QDRANT_COLLECTION)
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=VectorParams(size=embedding_dimension(), distance=Distance.COSINE),
    )


def index_documents(data_dir: str = config.DATA_DIR) -> int:
    documents = load_documents(data_dir)
    chunks = chunk_documents(documents)
    if not chunks:
        return 0

    vectors = embed_texts([chunk.page_content for chunk in chunks])

    client = get_qdrant_client()
    ensure_collection(client)

    points = [
        PointStruct(
            id=i,
            vector=vector,
            payload={
                "text": chunk.page_content,
                "source": chunk.metadata.get("source"),
                "page": chunk.metadata.get("page"),
            },
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
    return len(points)


if __name__ == "__main__":
    count = index_documents()
    print(f"Indexed {count} chunks into Qdrant collection '{config.QDRANT_COLLECTION}'.")
