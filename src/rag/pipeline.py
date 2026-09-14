from dataclasses import dataclass

from rag import config
from rag.llm import generate_answer
from rag.retrieval import RetrievedChunk, retrieve


@dataclass
class Answer:
    answer: str
    sources: list[RetrievedChunk]


def answer_question(question: str, top_k: int = config.TOP_K) -> Answer:
    chunks = retrieve(question, top_k=top_k)
    context = "\n\n".join(f"[{c.source}, page {c.page}]\n{c.text}" for c in chunks)

    text = generate_answer(question, context)
    return Answer(answer=text, sources=chunks)
