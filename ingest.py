from rag.indexing import index_documents
from rag import config

if __name__ == "__main__":
    count = index_documents()
    print(f"Indexed {count} chunks into Qdrant collection '{config.QDRANT_COLLECTION}'.")
