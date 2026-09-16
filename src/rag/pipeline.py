from dataclasses import dataclass

from rag import config, tracing
from rag.llm import generate_answer
from rag.retrieval import RetrievedChunk, retrieve


@dataclass
class Answer:
    answer: str
    sources: list[RetrievedChunk]


def answer_question(
    question: str,
    mode: str = config.RETRIEVAL_MODE,
    top_k: int = config.TOP_K,
) -> Answer:
    # The root observation of the call, ask.py and api.py's /query both go through this one
    # function, so this is the single place a query becomes a Langfuse trace.
    with tracing.observation(
        "rag.answer_question",
        as_type="chain",
        input=question,
        metadata={"mode": mode, "top_k": top_k},
    ) as obs:
        chunks = retrieve(question, mode=mode, top_k=top_k)
        context = "\n\n".join(f"[{c.source}, page {c.page}]\n{c.text}" for c in chunks)

        text = generate_answer(question, context)

        if obs:
            obs.update(output=text)
        return Answer(answer=text, sources=chunks)
