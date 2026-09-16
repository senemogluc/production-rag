import time


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["qdrant"] is True
    assert body["database"] is True


def _wait_until_ready(client, document_id: int, timeout: float = 60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get("/documents")
        assert response.status_code == 200
        for doc in response.json():
            if doc["id"] == document_id:
                if doc["status"] in ("ready", "failed"):
                    return doc
                break
        time.sleep(0.5)
    raise TimeoutError(f"document {document_id} did not finish processing in {timeout}s")


def test_document_upload_query_feedback_delete(client, sample_pdf_path):
    with sample_pdf_path.open("rb") as f:
        response = client.post(
            "/documents",
            files={"file": ("sample.pdf", f, "application/pdf")},
        )
    assert response.status_code == 202
    document = response.json()
    assert document["status"] == "pending"
    document_id = document["id"]

    ready = _wait_until_ready(client, document_id)
    assert ready["status"] == "ready", ready
    assert ready["chunk_count"] > 0

    response = client.post(
        "/query",
        json={"question": "What is described in this document?", "mode": "hybrid"},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["answer"]
    assert result["sources"]
    query_id = result["query_id"]

    response = client.post(
        "/feedback",
        json={"query_id": query_id, "rating": 5, "comment": "looks right"},
    )
    assert response.status_code == 201
    assert response.json()["query_id"] == query_id

    response = client.delete(f"/documents/{document_id}")
    assert response.status_code == 204

    response = client.get("/documents")
    remaining_ids = [doc["id"] for doc in response.json()]
    assert document_id not in remaining_ids


def test_feedback_unknown_query_id_returns_404(client):
    response = client.post("/feedback", json={"query_id": 999999, "rating": 3})
    assert response.status_code == 404


def test_delete_unknown_document_returns_404(client):
    response = client.delete("/documents/999999")
    assert response.status_code == 404
