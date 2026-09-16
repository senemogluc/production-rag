import math

from evaluation.metrics import context_precision, hit_rank, ndcg_at_k, recall_at_k, reciprocal_rank

GOLD = ("pytorch.pdf", 42)
OTHER = ("pytorch.pdf", 99)


def test_hit_rank_found():
    retrieved = [OTHER, GOLD, OTHER]
    assert hit_rank(retrieved, GOLD) == 2


def test_hit_rank_not_found():
    assert hit_rank([OTHER, OTHER], GOLD) is None


def test_hit_rank_first_match_wins():
    retrieved = [GOLD, GOLD]
    assert hit_rank(retrieved, GOLD) == 1


def test_recall_at_k_within_k():
    assert recall_at_k(rank=3, k=5) == 1.0


def test_recall_at_k_beyond_k():
    assert recall_at_k(rank=6, k=5) == 0.0


def test_recall_at_k_not_found():
    assert recall_at_k(rank=None, k=5) == 0.0


def test_reciprocal_rank_found():
    assert reciprocal_rank(4) == 0.25


def test_reciprocal_rank_not_found():
    assert reciprocal_rank(None) == 0.0


def test_ndcg_best_case_rank_one():
    # Single relevant item, ideal rank is 1, so IDCG = DCG = 1.0.
    assert ndcg_at_k(rank=1, k=5) == 1.0


def test_ndcg_decreases_with_rank():
    top = ndcg_at_k(rank=1, k=5)
    mid = ndcg_at_k(rank=3, k=5)
    low = ndcg_at_k(rank=5, k=5)
    assert top > mid > low
    assert mid == 1.0 / math.log2(4)


def test_ndcg_beyond_k_is_zero():
    assert ndcg_at_k(rank=6, k=5) == 0.0


def test_ndcg_not_found_is_zero():
    assert ndcg_at_k(rank=None, k=5) == 0.0


def test_context_precision_all_relevant():
    assert context_precision([GOLD, GOLD], GOLD) == 1.0


def test_context_precision_partial():
    assert context_precision([GOLD, OTHER, GOLD, OTHER], GOLD) == 0.5


def test_context_precision_none_relevant():
    assert context_precision([OTHER, OTHER], GOLD) == 0.0


def test_context_precision_empty_retrieval():
    assert context_precision([], GOLD) == 0.0
