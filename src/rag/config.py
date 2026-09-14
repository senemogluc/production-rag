import os

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

TOP_K = int(os.getenv("TOP_K", "5"))
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))

RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "dense")
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
DATA_DIR = os.getenv("DATA_DIR", "./data")

EVAL_NUM_QUESTIONS = int(os.getenv("EVAL_NUM_QUESTIONS", "30"))
EVAL_SEED = int(os.getenv("EVAL_SEED", "42"))
EVAL_DATASET_PATH = os.getenv("EVAL_DATASET_PATH", "./evaluation/dataset.json")
EVAL_RESULTS_PATH = os.getenv("EVAL_RESULTS_PATH", "./evaluation/results.csv")
