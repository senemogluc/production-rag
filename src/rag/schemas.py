from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from rag import config


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    status: str
    chunk_count: int
    error_message: str | None
    created_at: datetime


class QueryRequest(BaseModel):
    question: str
    top_k: int = config.TOP_K
    mode: Literal["dense", "hybrid", "hybrid_reranker"] = config.RETRIEVAL_MODE


class SourceOut(BaseModel):
    document: str
    page: int
    score: float


class QueryResponse(BaseModel):
    query_id: int
    answer: str
    sources: list[SourceOut]
    latency_ms: int


class FeedbackRequest(BaseModel):
    query_id: int | None = None
    rating: int
    comment: str | None = None


class FeedbackOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    query_id: int | None
    rating: int
    comment: str | None
    created_at: datetime


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    database: bool
