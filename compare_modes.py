from rag.retrieval import VALID_MODES, retrieve

SEMANTIC_QUERIES = [
    "How does autograd compute gradients?",
    "How do you move a model to the GPU?",
    "What is the purpose of a DataLoader?",
]

KEYWORD_QUERIES = [
    "torch.nn.Conv2d",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "torch.optim.Adam",
]


def show(question: str, top_k: int = 5):
    print("=" * 100)
    print(f"Q: {question}")
    for mode in VALID_MODES:
        print(f"\n[{mode}]")
        for chunk in retrieve(question, mode=mode, top_k=top_k):
            snippet = chunk.text[:80].replace("\n", " ")
            print(f"  {chunk.score:>10.4f}  p.{chunk.page:<5}  {snippet}")


def main():
    print("### Semantic queries ###")
    for q in SEMANTIC_QUERIES:
        show(q)

    print("\n### Exact-keyword queries ###")
    for q in KEYWORD_QUERIES:
        show(q)


if __name__ == "__main__":
    main()
