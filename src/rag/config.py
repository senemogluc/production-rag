import os

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

TOP_K = int(os.getenv("TOP_K", "5"))

QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
DATA_DIR = os.getenv("DATA_DIR", "./data")
