from ads_scholargraph.recsys.eval import ndcg_at_k, recall_at_k


def test_recall_at_k() -> None:
    recommended = ["A", "B", "C"]
    relevant = {"B", "C", "D"}
    assert recall_at_k(recommended, relevant, 2) == 1 / 3


def test_ndcg_at_k_perfect_ranking() -> None:
    recommended = ["A", "B", "C"]
    relevant = {"A", "B"}
    score = ndcg_at_k(recommended, relevant, 2)
    assert round(score, 6) == 1.0


def test_ndcg_at_k_partial_ranking() -> None:
    recommended = ["X", "A", "B"]
    relevant = {"A", "B"}
    score = ndcg_at_k(recommended, relevant, 3)
    assert 0.0 < score < 1.0
