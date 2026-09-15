import os
import tempfile
from pathlib import Path

# Isolate the test run from the real qdrant_data/ and app.db BEFORE rag.config (and anything
# that imports it) is ever imported. rag.config reads these via os.getenv() at import time, so
# this must happen at module load, not inside a fixture.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="production_rag_test_"))
os.environ["QDRANT_PATH"] = str(_TEST_ROOT / "qdrant_data")
os.environ["QDRANT_COLLECTION"] = "test_documents"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_ROOT / 'test_app.db'}"
os.environ["DATA_DIR"] = str(_TEST_ROOT / "data")
os.environ["UPLOAD_DIR"] = str(_TEST_ROOT / "data" / "uploads")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pypdf import PdfReader, PdfWriter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def client():
    from rag.api import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def sample_pdf_path(tmp_path_factory):
    """A tiny 2-page slice of data/pytorch.pdf, fast to index for upload tests."""
    reader = PdfReader(str(REPO_ROOT / "data" / "pytorch.pdf"))
    writer = PdfWriter()
    for page in reader.pages[:2]:
        writer.add_page(page)

    out_dir = tmp_path_factory.mktemp("sample_pdf")
    out_path = out_dir / "sample.pdf"
    with out_path.open("wb") as f:
        writer.write(f)
    return out_path
