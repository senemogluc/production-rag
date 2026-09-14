import math


def hit_rank(retrieved: list[tuple[str, int]], gold: tuple[str, int]) -> int | None:
    """1-indexed rank of the first retrieved (source, page) matching gold, or None."""
    for i, item in enumerate(retrieved, start=1):
        if item == gold:
            return i
    return None


def recall_at_k(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank is not None else 0.0


def ndcg_at_k(rank: int | None, k: int) -> float:
    """Single relevant item per query, so IDCG = 1 (best case: relevant item at rank 1)."""
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def context_precision(retrieved: list[tuple[str, int]], gold: tuple[str, int]) -> float:
    """Fraction of the retrieved (source, page) pairs that match the gold page."""
    if not retrieved:
        return 0.0
    matches = sum(1 for item in retrieved if item == gold)
    return matches / len(retrieved)
