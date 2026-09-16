import os
import tempfile
from pathlib import Path

# Isolate the API test run from the real qdrant_data/ and app.db BEFORE rag.config (and anything
# that imports it) is ever imported. rag.config reads these via os.getenv() at import time, so
# this must happen at module load, not inside a fixture.
#
# IMPORTANT: because of that caching, this isolation is process-wide, not just "scoped to
# tests/api/" in any enforced sense. tests/test_eval_regression.py needs the real, fully-indexed
# corpus, and if it's collected in the *same* pytest process as this conftest.py, it silently
# inherits this isolated (empty) QDRANT_PATH too. Always run tests/api/ and the rest of tests/ as
# separate pytest invocations, see tests/test_eval_regression.py's own warning and
# CLAUDE.md/README.md's "Running it" section.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="production_rag_test_"))
os.environ["QDRANT_URL"] = ""  # force embedded mode, no real Qdrant server needed for tests
os.environ["QDRANT_PATH"] = str(_TEST_ROOT / "qdrant_data")
os.environ["QDRANT_COLLECTION"] = "test_documents"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_ROOT / 'test_app.db'}"
os.environ["DATA_DIR"] = str(_TEST_ROOT / "data")
os.environ["UPLOAD_DIR"] = str(_TEST_ROOT / "data" / "uploads")
os.environ["LANGFUSE_PUBLIC_KEY"] = ""  # disable tracing, tests shouldn't call out to Langfuse
os.environ["LANGFUSE_SECRET_KEY"] = ""

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pypdf import PdfReader, PdfWriter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# CI has no GPU, so a real Qwen2.5-3B generation call (loading the model + producing up to 400
# tokens) is the slowest single thing in the whole suite. Retrieval (BGE-small, BM25, the
# reranker) stays real either way, it's fast on CPU too. Set MOCK_LLM=1 (CI only, see
# .github/workflows/ci.yml) to swap in a canned response instead. Unset locally, so `pytest
# tests/api/` on a dev machine always exercises the real model, same as before this existed.
MOCK_LLM = os.environ.get("MOCK_LLM") == "1"


@pytest.fixture(scope="session")
def client():
    from rag import llm

    if MOCK_LLM:
        llm.generate = lambda system_prompt, user_prompt, max_new_tokens=400: (
            "Mocked answer (MOCK_LLM=1): real generation is skipped in CI, see tests/api/conftest.py."
        )
        llm.warm_up = lambda: None

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
