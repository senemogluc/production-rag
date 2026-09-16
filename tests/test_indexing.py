from langchain_core.documents import Document

from rag.indexing import _point_id, chunk_documents


def test_point_id_is_deterministic():
    a = _point_id("doc-1", 0)
    b = _point_id("doc-1", 0)
    assert a == b


def test_point_id_differs_by_chunk_index():
    a = _point_id("doc-1", 0)
    b = _point_id("doc-1", 1)
    assert a != b


def test_point_id_differs_by_document_id():
    a = _point_id("doc-1", 0)
    b = _point_id("doc-2", 0)
    assert a != b


def test_point_id_no_collisions_across_a_batch():
    ids = [_point_id(f"doc-{d}", i) for d in range(5) for i in range(20)]
    assert len(ids) == len(set(ids))


def test_chunk_documents_splits_long_text():
    long_text = "word " * 1000  # 5000 chars, well over the 800-char default chunk size
    doc = Document(page_content=long_text, metadata={"source": "test.pdf", "page": 1})

    chunks = chunk_documents([doc])

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 1


def test_chunk_documents_keeps_short_text_as_one_chunk():
    doc = Document(page_content="A short passage.", metadata={"source": "test.pdf", "page": 1})

    chunks = chunk_documents([doc])

    assert len(chunks) == 1
    assert chunks[0].page_content == "A short passage."
