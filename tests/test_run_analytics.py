from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ads_scholargraph.graph_analytics.run_analytics import run_analytics


@dataclass
class FakeResult:
    records: list[dict[str, Any]] | None = None

    def consume(self) -> None:
        return None

    def single(self) -> dict[str, Any] | None:
        if not self.records:
            return None
        return self.records[0]

    def __iter__(self):
        return iter(self.records or [])


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, query: str, **kwargs: Any) -> FakeResult:
        self.calls.append((query, kwargs))
        return FakeResult()

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class FakeDriver:
    def __init__(self, session: FakeSession) -> None:
        self._session = session
        self.closed = False

    def session(self) -> FakeSession:
        return self._session

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_driver() -> tuple[FakeSession, FakeDriver]:
    session = FakeSession()
    return session, FakeDriver(session)


def test_run_analytics_auto_uses_gds_when_available(monkeypatch, fake_driver) -> None:
    session, driver = fake_driver

    monkeypatch.setattr(
        "ads_scholargraph.graph_analytics.run_analytics._is_gds_available",
        lambda _: True,
    )
    monkeypatch.setattr(
        "ads_scholargraph.graph_analytics.run_analytics._compute_gds_metrics",
        lambda _: {
            "pagerank": [{"bibcode": "P1", "score": 0.42}],
            "communities": [{"bibcode": "P1", "community_id": 1}],
            "betweenness": [{"author_name": "Alice", "score": 0.1}],
        },
    )

    result = run_analytics(mode="auto", batch_size=2, driver=driver)

    assert result["implementation"] == "gds"
    assert result["counts"] == {"pagerank": 1, "communities": 1, "betweenness": 1}
    assert len(session.calls) == 3
    assert any("p.pagerank" in query for query, _ in session.calls)
    assert any("p.community_id" in query for query, _ in session.calls)
    assert any("a.betweenness" in query for query, _ in session.calls)


def test_run_analytics_auto_falls_back_to_networkx(monkeypatch, fake_driver) -> None:
    session, driver = fake_driver

    monkeypatch.setattr(
        "ads_scholargraph.graph_analytics.run_analytics._is_gds_available",
        lambda _: True,
    )

    def fail_gds(_: Any) -> dict[str, list[dict[str, Any]]]:
        raise RuntimeError("gds failed")

    monkeypatch.setattr(
        "ads_scholargraph.graph_analytics.run_analytics._compute_gds_metrics",
        fail_gds,
    )
    monkeypatch.setattr(
        "ads_scholargraph.graph_analytics.run_analytics._compute_networkx_metrics",
        lambda _: {
            "pagerank": [{"bibcode": "P1", "score": 0.2}],
            "communities": [{"bibcode": "P1", "community_id": 0}],
            "betweenness": [{"author_name": "Bob", "score": 0.3}],
        },
    )

    result = run_analytics(mode="auto", batch_size=10, driver=driver)

    assert result["implementation"] == "networkx"
    assert result["counts"]["pagerank"] == 1
    assert len(session.calls) == 3


def test_run_analytics_gds_mode_requires_gds(monkeypatch, fake_driver) -> None:
    _, driver = fake_driver

    monkeypatch.setattr(
        "ads_scholargraph.graph_analytics.run_analytics._is_gds_available",
        lambda _: False,
    )

    with pytest.raises(RuntimeError, match="GDS is not available"):
        run_analytics(mode="gds", driver=driver)
