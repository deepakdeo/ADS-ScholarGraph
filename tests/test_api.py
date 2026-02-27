from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from ads_scholargraph.api.main import app, get_repository


class _FakeRepository:
    def get_paper(self, bibcode: str) -> dict[str, Any] | None:
        if bibcode == "B1":
            return {
                "bibcode": "B1",
                "title": "Seed Paper",
                "year": 2024,
                "abstract": "Seed abstract",
                "is_seed": True,
                "pagerank": 0.42,
                "community_id": 3,
            }
        return None

    def search_seed_papers(self, q: str, limit: int) -> list[dict[str, Any]]:
        return [{"bibcode": "B1", "title": "Seed Paper", "year": 2024}]


def test_health_endpoint() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_paper_and_search_endpoints() -> None:
    app.dependency_overrides[get_repository] = lambda: _FakeRepository()
    client = TestClient(app)

    try:
        paper = client.get("/paper/B1")
        assert paper.status_code == 200
        assert paper.json()["title"] == "Seed Paper"

        search = client.get("/search", params={"q": "seed", "limit": 5})
        assert search.status_code == 200
        assert search.json()[0]["bibcode"] == "B1"
    finally:
        app.dependency_overrides.clear()


def test_recommend_endpoint_graph_mode(monkeypatch) -> None:
    app.dependency_overrides[get_repository] = lambda: _FakeRepository()

    def fake_graph(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "bibcode": "R1",
                "title": "Rec 1",
                "year": 2023,
                "score": 0.88,
                "reasons": ["Shares 3 references"],
            }
        ]

    monkeypatch.setattr("ads_scholargraph.api.main.recommend_similar_papers_graph", fake_graph)

    client = TestClient(app)
    try:
        resp = client.get("/recommend/paper/B1", params={"k": 5, "mode": "graph"})
        assert resp.status_code == 200
        rows = resp.json()
        assert rows[0]["bibcode"] == "R1"
        assert rows[0]["reasons"] == ["Shares 3 references"]
    finally:
        app.dependency_overrides.clear()
