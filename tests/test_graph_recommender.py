from typing import Any

from ads_scholargraph.recsys.graph_recommender import recommend_similar_papers_graph


class _FakeGraphClient:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.calls: list[dict[str, Any]] = []

    def run_query(self, query: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append({"query": query, **params})
        return self._rows


def test_graph_recommender_scores_and_reasons() -> None:
    rows = [
        {
            "bibcode": "A",
            "title": "Paper A",
            "year": 2024,
            "pagerank": 0.5,
            "overlap_count": 6,
            "shared_references": 4,
            "co_citation_overlap": 1,
            "same_community": True,
            "community_id": 2,
        },
        {
            "bibcode": "B",
            "title": "Paper B",
            "year": 2023,
            "pagerank": 1.0,
            "overlap_count": 2,
            "shared_references": 1,
            "co_citation_overlap": 1,
            "same_community": False,
            "community_id": None,
        },
    ]
    client = _FakeGraphClient(rows)

    recs = recommend_similar_papers_graph(
        seed_bibcode="SEED",
        k=2,
        candidate_pool=10,
        client=client,
    )

    assert [rec["bibcode"] for rec in recs] == ["A", "B"]
    assert any("Same research cluster" in reason for reason in recs[0]["reasons"])
    assert any("Shares 4 references" in reason for reason in recs[0]["reasons"])
    assert recs[0]["score"] > recs[1]["score"]
    assert recs[0]["pagerank"] == 0.5


def test_graph_recommender_respects_empty_inputs() -> None:
    client = _FakeGraphClient([])

    assert recommend_similar_papers_graph("SEED", k=0, client=client) == []
    assert recommend_similar_papers_graph("SEED", candidate_pool=0, client=client) == []
