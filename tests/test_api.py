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
                "citation_count": 27,
                "is_seed": True,
                "pagerank": 0.42,
                "community_id": 3,
            }
        return None

    def search_seed_papers(self, q: str, limit: int) -> list[dict[str, Any]]:
        return [{"bibcode": "B1", "title": "Seed Paper", "year": 2024}]

    def get_citation_edges(self, bibcodes: list[str], limit: int) -> list[dict[str, Any]]:
        return [{"source": "B1", "target": "R1"}]

    def get_keyword_links(self, bibcodes: list[str], limit: int) -> list[dict[str, Any]]:
        return [{"bibcode": "R1", "keyword": "quenching"}]

    def get_similarity_edges(self, bibcodes: list[str], limit: int) -> list[dict[str, Any]]:
        return [{"source": "R1", "target": "R2", "similarity": 0.44}]

    def get_graph_stats_summary(self) -> dict[str, Any]:
        return {
            "paper_count": 3,
            "author_count": 2,
            "keyword_count": 4,
            "venue_count": 1,
            "cites_count": 2,
            "community_count": 1,
        }

    def get_top_papers_by_pagerank(self, limit: int) -> list[dict[str, Any]]:
        return [
            {
                "bibcode": "B1",
                "title": "Seed Paper",
                "pagerank": 0.42,
                "citation_count": 27,
                "year": 2024,
            }
        ]

    def get_community_sizes(self, limit: int) -> list[dict[str, Any]]:
        return [{"community_id": 3, "size": 3}]

    def get_publications_per_year(self) -> list[dict[str, Any]]:
        return [{"year": 2024, "count": 2}, {"year": 2025, "count": 1}]

    def get_citation_counts(self, limit: int) -> list[int]:
        return [27, 14, 5]


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
        assert paper.json()["citation_count"] == 27

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


def test_subgraph_endpoint(monkeypatch) -> None:
    app.dependency_overrides[get_repository] = lambda: _FakeRepository()

    def fake_graph(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "bibcode": "R1",
                "title": "Rec 1",
                "year": 2023,
                "score": 0.91,
                "reasons": ["Same community (2)"],
            },
            {
                "bibcode": "R2",
                "title": "Rec 2",
                "year": 2022,
                "score": 0.73,
                "reasons": ["High TF-IDF similarity"],
            },
        ]

    monkeypatch.setattr("ads_scholargraph.api.main.recommend_similar_papers_graph", fake_graph)

    client = TestClient(app)
    try:
        resp = client.get(
            "/subgraph/paper/B1",
            params={
                "k": 2,
                "mode": "graph",
                "include_citations": True,
                "include_keywords": True,
                "include_similarity": True,
            },
        )
        assert resp.status_code == 200
        payload = resp.json()

        node_ids = {node["id"] for node in payload["nodes"]}
        assert {"B1", "R1", "R2"}.issubset(node_ids)
        assert any(node["type"] == "keyword" for node in payload["nodes"])

        edge_types = {edge["type"] for edge in payload["edges"]}
        assert "RECOMMENDS" in edge_types
        assert "CITES" in edge_types
        assert "HAS_KEYWORD" in edge_types
        assert "SIMILAR_TO" in edge_types
    finally:
        app.dependency_overrides.clear()


def test_stats_overview_endpoint() -> None:
    app.dependency_overrides[get_repository] = lambda: _FakeRepository()
    client = TestClient(app)
    try:
        resp = client.get("/stats/overview")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["summary"]["paper_count"] == 3
        assert payload["summary"]["cites_count"] == 2
        assert payload["top_papers"][0]["bibcode"] == "B1"
        assert payload["community_sizes"][0]["community_id"] == 3
        assert payload["publications_per_year"][0]["year"] == 2024
        assert payload["citation_counts"] == [27, 14, 5]
    finally:
        app.dependency_overrides.clear()
