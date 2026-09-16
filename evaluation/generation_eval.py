import re

from evaluation.retrieval_eval import load_dataset

from rag import config
from rag.llm import generate
from rag.pipeline import answer_question
from rag.retrieval import VALID_MODES

JUDGE_SYSTEM_PROMPT = (
    "You are evaluating an AI assistant's answer to a question, given the context it was "
    "given. Score two things from 1 (worst) to 5 (best):\n"
    "Faithfulness: is every claim in the answer supported by the context? 5 means fully "
    "supported, 1 means the answer contradicts the context or invents information not in it.\n"
    "Relevancy: does the answer actually address the question asked? 5 means it directly and "
    "completely answers the question, 1 means it does not address the question at all.\n"
    "Respond in exactly this format:\n"
    "Faithfulness: <1-5>\n"
    "Relevancy: <1-5>"
)


def _extract_number(line: str) -> float | None:
    match = re.search(r"(\d+(\.\d+)?)", line)
    return float(match.group(1)) if match else None


def _parse_scores(text: str) -> tuple[float, float] | None:
    faithfulness = None
    relevancy = None
    for line in text.splitlines():
        line = line.strip()
        if faithfulness is None and line.lower().startswith("faithfulness:"):
            faithfulness = _extract_number(line)
        elif relevancy is None and line.lower().startswith("relevancy:"):
            relevancy = _extract_number(line)
    if faithfulness is not None and relevancy is not None:
        return faithfulness, relevancy
    return None


def judge(question: str, context: str, answer: str) -> tuple[float, float] | None:
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer: {answer}"
    response = generate(JUDGE_SYSTEM_PROMPT, user_prompt, max_new_tokens=50)
    return _parse_scores(response)


def evaluate_generation(dataset: list[dict], mode: str) -> dict:
    faithfulness_scores, relevancy_scores = [], []

    for item in dataset:
        result = answer_question(item["question"], mode=mode, top_k=config.TOP_K)
        context = "\n\n".join(f"[{c.source}, page {c.page}]\n{c.text}" for c in result.sources)

        scores = judge(item["question"], context, result.answer)
        if scores is None:
            continue
        faithfulness, relevancy = scores
        faithfulness_scores.append(faithfulness)
        relevancy_scores.append(relevancy)

    n = len(faithfulness_scores)
    if n == 0:
        return {"avg_faithfulness": None, "avg_answer_relevancy": None, "judged_count": 0}

    return {
        "avg_faithfulness": sum(faithfulness_scores) / n,
        "avg_answer_relevancy": sum(relevancy_scores) / n,
        "judged_count": n,
    }


def main():
    dataset = load_dataset()
    for mode in VALID_MODES:
        result = evaluate_generation(dataset, mode)
        print(mode, result)


if __name__ == "__main__":
    main()
