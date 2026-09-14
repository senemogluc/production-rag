import json
import random
from pathlib import Path

from rag import config
from rag.indexing import get_qdrant_client
from rag.llm import generate

DATASET_SYSTEM_PROMPT = (
    "You are creating an evaluation dataset for a question-answering system. "
    "Given a passage from technical documentation, write ONE specific, self-contained "
    'question that is fully answered by the passage, and a short reference answer '
    "(1-2 sentences) using only information in the passage. Do not refer to \"the "
    "passage\" or \"the context\" in the question. Respond in exactly this format:\n"
    "Question: <question>\n"
    "Answer: <answer>"
)

MIN_CHUNK_LENGTH = 400


def _scroll_all_chunks(client) -> list[dict]:
    chunks = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=500,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        chunks.extend(point.payload for point in points)
        if offset is None:
            break
    return chunks


def _parse_question_answer(text: str) -> tuple[str, str] | None:
    question = None
    answer = None
    for line in text.splitlines():
        line = line.strip()
        if question is None and line.lower().startswith("question:"):
            question = line.split(":", 1)[1].strip()
        elif answer is None and line.lower().startswith("answer:"):
            answer = line.split(":", 1)[1].strip()
    if question and answer:
        return question, answer
    return None


def build_dataset(n: int = config.EVAL_NUM_QUESTIONS, seed: int = config.EVAL_SEED) -> list[dict]:
    client = get_qdrant_client()
    all_chunks = _scroll_all_chunks(client)

    candidates = [c for c in all_chunks if len(c["text"].strip()) >= MIN_CHUNK_LENGTH]
    rng = random.Random(seed)
    rng.shuffle(candidates)

    dataset = []
    for chunk in candidates:
        if len(dataset) >= n:
            break

        response = generate(
            DATASET_SYSTEM_PROMPT,
            f"Passage:\n{chunk['text']}",
            max_new_tokens=200,
        )
        parsed = _parse_question_answer(response)
        if parsed is None:
            continue

        question, answer = parsed
        dataset.append(
            {
                "question": question,
                "reference_answer": answer,
                "gold_source": chunk["source"],
                "gold_page": chunk["page"],
                "gold_chunk_text": chunk["text"],
            }
        )
        print(f"[{len(dataset)}/{n}] {question}")

    return dataset


def main():
    dataset = build_dataset()
    out_path = Path(config.EVAL_DATASET_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(dataset)} evaluation questions to {out_path}")


if __name__ == "__main__":
    main()
