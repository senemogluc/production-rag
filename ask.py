import argparse

from rag import config, tracing
from rag.pipeline import answer_question


def parse_args():
    parser = argparse.ArgumentParser(description="Ask a question against the indexed documents.")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument(
        "--mode",
        choices=["dense", "hybrid", "hybrid_reranker"],
        default=config.RETRIEVAL_MODE,
        help=f"Retrieval mode (default: {config.RETRIEVAL_MODE})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = answer_question(args.question, mode=args.mode)

    print("Answer:")
    print(result.answer)
    print()
    print(f"Sources (mode={args.mode}):")
    seen = set()
    for chunk in result.sources:
        key = (chunk.source, chunk.page)
        if key in seen:
            continue
        seen.add(key)
        print(f"- {chunk.source}, page {chunk.page} (score={chunk.score:.4f})")

    tracing.flush()


if __name__ == "__main__":
    main()
