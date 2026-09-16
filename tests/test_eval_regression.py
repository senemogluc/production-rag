# IMPORTANT: run this file separately from tests/api/ (e.g. `pytest tests/ --ignore=tests/api`),
# never in the same pytest invocation. tests/api/conftest.py points QDRANT_PATH at an isolated,
# empty temp directory via os.environ *at module import time*; since rag.config reads env vars
# once and caches them as module constants, that isolation leaks into this file too if both are
# collected in one process, silently pointing this test at an empty index (Recall@5 = 0.000
# instead of a real regression signal). Different pytest invocations are different processes, so
# running these separately is the actual fix, not just a suggestion.

import json
from pathlib import Path

from evaluation.retrieval_eval import evaluate_retrieval

from rag import config

# Committed baseline from evaluation/results.csv (V2), see docs/v2-notes.md. This test isn't
# meant to reproduce that exact number (the LLM-generated dataset and a live index both have some
# variance), it's meant to catch a *large* regression, e.g. a chunking or retrieval change that
# tanks quality while everything still runs without raising an exception. See the roadmap's own
# example: "Recall@5 dropped from 0.88 to 0.61, tests still passed."
BASELINE_RECALL_AT_5 = 0.833
BASELINE_MRR = 0.623
TOLERANCE = 0.15


def _load_dataset() -> list[dict]:
    path = Path(config.EVAL_DATASET_PATH)
    return json.loads(path.read_text(encoding="utf-8"))


def test_hybrid_retrieval_does_not_regress_below_v2_baseline():
    dataset = _load_dataset()
    result = evaluate_retrieval(dataset, mode="hybrid")

    assert result["recall_at_k"] >= BASELINE_RECALL_AT_5 - TOLERANCE, (
        f"Recall@5 is {result['recall_at_k']:.3f}, more than {TOLERANCE} below the V2 baseline "
        f"of {BASELINE_RECALL_AT_5}. This means retrieval quality regressed, even if nothing "
        f"raised an exception."
    )
    assert result["mrr"] >= BASELINE_MRR - TOLERANCE, (
        f"MRR is {result['mrr']:.3f}, more than {TOLERANCE} below the V2 baseline of "
        f"{BASELINE_MRR}."
    )
