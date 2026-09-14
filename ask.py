import sys

from rag.pipeline import answer_question


def main():
    if len(sys.argv) < 2:
        print('Usage: python ask.py "<question>"')
        sys.exit(1)

    question = sys.argv[1]
    result = answer_question(question)

    print("Answer:")
    print(result.answer)
    print()
    print("Sources:")
    seen = set()
    for chunk in result.sources:
        key = (chunk.source, chunk.page)
        if key in seen:
            continue
        seen.add(key)
        print(f"- {chunk.source}, page {chunk.page}")


if __name__ == "__main__":
    main()
