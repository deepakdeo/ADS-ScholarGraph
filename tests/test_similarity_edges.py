from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("sklearn")

from ads_scholargraph.graph_analytics.add_similarity_edges import (
    _build_documents,
    _compute_similarity_edges,
    add_similarity_edges,
)


@dataclass
class _FakeResult:
    records: list[dict[str, Any]] | None = None

    def consume(self) -> None:
        return None

    def __iter__(self):
        return iter(self.records or [])


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, query: str, **kwargs: Any) -> _FakeResult:
        self.calls.append((query, kwargs))
        if "RETURN p.bibcode AS bibcode" in query:
            return _FakeResult(
                records=[
                    {
                        "bibcode": "A",
                        "title": "Galaxy evolution",
                        "abstract": "stellar mass quenching",
                    },
                    {
                        "bibcode": "B",
                        "title": "Galaxy quenching",
                        "abstract": "stellar populations",
                    },
                    {"bibcode": "C", "title": "Quantum gravity", "abstract": "loop amplitudes"},
                ]
            )
        return _FakeResult()

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeDriver:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session
        self.closed = False

    def session(self) -> _FakeSession:
        return self._session

    def close(self) -> None:
        self.closed = True


def test_build_documents_filters_invalid_rows() -> None:
    rows: list[dict[str, Any]] = [
        {"bibcode": "A", "title": "Title A", "abstract": "Abstract A"},
        {"bibcode": "B", "title": "", "abstract": ""},
        {"bibcode": None, "title": "Missing bibcode", "abstract": "x"},
    ]

    bibcodes, docs = _build_documents(rows)

    assert bibcodes == ["A"]
    assert docs == ["Title A Abstract A"]


def test_compute_similarity_edges_respects_threshold_and_top_k() -> None:
    bibcodes = ["A", "B", "C"]
    docs = [
        "galaxy stellar quenching evolution",
        "stellar quenching in massive galaxy systems",
        "quantum fields and gravity amplitudes",
    ]

    edges = _compute_similarity_edges(
        bibcodes=bibcodes,
        docs=docs,
        threshold=0.2,
        top_k=1,
        max_features=2000,
    )

    assert len(edges) == 1
    edge = edges[0]
    assert {edge["bib_a"], edge["bib_b"]} == {"A", "B"}
    assert edge["similarity"] >= 0.2


def test_add_similarity_edges_writes_rows_and_optionally_wipes() -> None:
    session = _FakeSession()
    driver = _FakeDriver(session)

    result = add_similarity_edges(
        threshold=0.2,
        top_k=2,
        seed_only=True,
        limit=100,
        max_features=2000,
        wipe_existing=True,
        batch_size=1,
        driver=driver,
    )

    assert result["papers_considered"] == 3
    assert result["similarity_edges"] >= 1

    queries = [query for query, _ in session.calls]
    assert any("MATCH ()-[r:SIMILAR_TO]-() DELETE r" in query for query in queries)
    assert any("MERGE (a)-[r:SIMILAR_TO]-(b)" in query for query in queries)
