import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, UploadFile
from sqlalchemy import text
from sqlalchemy.orm import Session

from rag import config, indexing, llm
from rag.db import SessionLocal, get_db, init_db
from rag.db_models import DocumentRecord, FeedbackRecord, QueryLog
from rag.embeddings import get_embedding_model
from rag.pipeline import answer_question
from rag.schemas import (
    DocumentOut,
    FeedbackOut,
    FeedbackRequest,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceOut,
)
from rag.sparse import get_sparse_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    Path(config.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    # Warm up models once here instead of paying the load cost on the first request.
    get_embedding_model()
    get_sparse_model()
    llm.warm_up()
    yield


app = FastAPI(
    title="production-rag",
    description="V3 production API for the RAG pipeline.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        indexing.get_qdrant_client().collection_exists(config.QDRANT_COLLECTION)
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        db_ok = True
    except Exception:
        db_ok = False

    return HealthResponse(status="ok", qdrant=qdrant_ok, database=db_ok)


def _process_document(document_id: int, path: str):
    db = SessionLocal()
    try:
        record = db.get(DocumentRecord, document_id)
        record.status = "processing"
        db.commit()

        count = indexing.index_document_file(path, document_id=str(document_id))

        record.status = "ready"
        record.chunk_count = count
        db.commit()
    except Exception as exc:
        record = db.get(DocumentRecord, document_id)
        record.status = "failed"
        record.error_message = str(exc)
        db.commit()
    finally:
        db.close()


@app.post("/documents", response_model=DocumentOut, status_code=202)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    record = DocumentRecord(filename=file.filename, status="pending")
    db.add(record)
    db.commit()
    db.refresh(record)

    dest = Path(config.UPLOAD_DIR) / f"{record.id}_{file.filename}"
    dest.write_bytes(file.file.read())

    background_tasks.add_task(_process_document, record.id, str(dest))
    return record


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    return db.query(DocumentRecord).order_by(DocumentRecord.id).all()


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    record = db.get(DocumentRecord, document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")

    indexing.delete_document(str(document_id))

    for uploaded in Path(config.UPLOAD_DIR).glob(f"{document_id}_*"):
        uploaded.unlink(missing_ok=True)

    db.delete(record)
    db.commit()


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, db: Session = Depends(get_db)):
    start = time.perf_counter()
    result = answer_question(req.question, mode=req.mode, top_k=req.top_k)
    latency_ms = int((time.perf_counter() - start) * 1000)

    log = QueryLog(
        question=req.question,
        mode=req.mode,
        answer=result.answer,
        latency_ms=latency_ms,
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    return QueryResponse(
        query_id=log.id,
        answer=result.answer,
        sources=[SourceOut(document=c.source, page=c.page, score=c.score) for c in result.sources],
        latency_ms=latency_ms,
    )


@app.post("/feedback", response_model=FeedbackOut, status_code=201)
def feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    if req.query_id is not None and db.get(QueryLog, req.query_id) is None:
        raise HTTPException(status_code=404, detail="query_id not found")

    record = FeedbackRecord(query_id=req.query_id, rating=req.rating, comment=req.comment)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
