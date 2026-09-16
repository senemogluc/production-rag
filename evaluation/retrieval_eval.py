import json
import time
from pathlib import Path

from evaluation import metrics

from rag import config
from rag.retrieval import VALID_MODES, retrieve


def load_dataset(path: str = config.EVAL_DATASET_PATH) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_retrieval(dataset: list[dict], mode: str, top_k: int = config.TOP_K) -> dict:
    recalls, rrs, ndcgs, precisions, latencies_ms = [], [], [], [], []

    for item in dataset:
        gold = (item["gold_source"], item["gold_page"])

        start = time.perf_counter()
        results = retrieve(item["question"], mode=mode, top_k=top_k)
        latencies_ms.append((time.perf_counter() - start) * 1000)

        retrieved_pairs = [(c.source, c.page) for c in results]
        rank = metrics.hit_rank(retrieved_pairs, gold)

        recalls.append(metrics.recall_at_k(rank, top_k))
        rrs.append(metrics.reciprocal_rank(rank))
        ndcgs.append(metrics.ndcg_at_k(rank, top_k))
        precisions.append(metrics.context_precision(retrieved_pairs, gold))

    n = len(dataset)
    return {
        "recall_at_k": sum(recalls) / n,
        "mrr": sum(rrs) / n,
        "ndcg_at_k": sum(ndcgs) / n,
        "context_precision": sum(precisions) / n,
        # Single gold page per question, so "was it retrieved at all" (context recall)
        # is the same computation as Recall@K. See docs/v2-notes.md.
        "context_recall": sum(recalls) / n,
        "avg_retrieval_latency_ms": sum(latencies_ms) / n,
    }


def main():
    dataset = load_dataset()
    for mode in VALID_MODES:
        result = evaluate_retrieval(dataset, mode)
        print(mode, result)


if __name__ == "__main__":
    main()
