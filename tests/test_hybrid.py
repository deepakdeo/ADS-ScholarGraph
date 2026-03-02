from typing import Any

from ads_scholargraph.recsys.hybrid import HybridRecommender


class _FakeEmbedding:
    def similarity_for_candidates(
        self,
        seed_bibcode: str,
        candidate_bibcodes: list[str],
    ) -> dict[str, float]:
        return {"A": 0.9, "B": 0.1}


def test_hybrid_combines_graph_and_embedding(monkeypatch) -> None:
    def fake_graph(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "bibcode": "A",
                "title": "Paper A",
                "year": 2024,
                "score": 0.8,
                "pagerank": 0.2,
                "reasons": ["Same research cluster", "Shares 3 references"],
            },
            {
                "bibcode": "B",
                "title": "Paper B",
                "year": 2022,
                "score": 0.9,
                "pagerank": 1.0,
                "reasons": ["Shares 1 references"],
            },
        ]

    monkeypatch.setattr("ads_scholargraph.recsys.hybrid.recommend_similar_papers_graph", fake_graph)

    recommender = HybridRecommender(embedding_recommender=_FakeEmbedding())
    recs = recommender.recommend_similar_papers_hybrid("SEED", k=2, candidate_pool=10)

    assert len(recs) == 2
    assert recs[0]["bibcode"] == "A"
    assert any("High abstract similarity" in reason for reason in recs[0]["reasons"])


def test_hybrid_normalize_handles_zero_max() -> None:
    normalized = HybridRecommender._normalize({"A": 0.0, "B": 0.0})
    assert normalized == {"A": 0.0, "B": 0.0}
