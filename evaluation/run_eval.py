import csv
from pathlib import Path

from generation_eval import evaluate_generation
from retrieval_eval import evaluate_retrieval, load_dataset

from rag import config
from rag.retrieval import VALID_MODES


def run():
    dataset = load_dataset()
    rows = []

    for mode in VALID_MODES:
        print(f"=== {mode}: retrieval ===")
        retrieval_result = evaluate_retrieval(dataset, mode)
        print(retrieval_result)

        print(f"=== {mode}: generation ===")
        generation_result = evaluate_generation(dataset, mode)
        print(generation_result)

        faithfulness = generation_result["avg_faithfulness"]
        relevancy = generation_result["avg_answer_relevancy"]

        rows.append(
            {
                "mode": mode,
                "recall_at_5": round(retrieval_result["recall_at_k"], 4),
                "mrr": round(retrieval_result["mrr"], 4),
                "ndcg_at_5": round(retrieval_result["ndcg_at_k"], 4),
                "context_precision": round(retrieval_result["context_precision"], 4),
                "context_recall": round(retrieval_result["context_recall"], 4),
                "avg_retrieval_latency_ms": round(retrieval_result["avg_retrieval_latency_ms"], 1),
                "avg_faithfulness": round(faithfulness, 2) if faithfulness is not None else "",
                "avg_answer_relevancy": round(relevancy, 2) if relevancy is not None else "",
                "judged_count": generation_result["judged_count"],
            }
        )

    out_path = Path(config.EVAL_RESULTS_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote results for {len(rows)} modes to {out_path}")


if __name__ == "__main__":
    run()
