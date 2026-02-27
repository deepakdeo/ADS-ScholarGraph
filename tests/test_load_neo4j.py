from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ads_scholargraph.pipeline.load_neo4j import CITATION_QUERY, PAPER_QUERY, load_kg


@dataclass
class _FakeResult:
    def consume(self) -> None:
        return None


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, query: str, **kwargs: Any) -> _FakeResult:
        self.calls.append((query, kwargs))
        return _FakeResult()

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeDriver:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    def session(self) -> _FakeSession:
        return self._session

    def __enter__(self) -> "_FakeDriver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeGraphDatabase:
    def __init__(self, driver: _FakeDriver) -> None:
        self._driver = driver

    def driver(self, uri: str, auth: tuple[str, str]) -> _FakeDriver:
        return self._driver


def test_load_kg_writes_seed_metadata_and_seed_flag(monkeypatch, tmp_path: Path) -> None:
    indir = tmp_path / "processed"
    indir.mkdir(parents=True)

    fake_session = _FakeSession()
    fake_driver = _FakeDriver(fake_session)

    monkeypatch.setattr(
        "ads_scholargraph.pipeline.load_neo4j.GraphDatabase",
        _FakeGraphDatabase(fake_driver),
    )
    monkeypatch.setattr(
        "ads_scholargraph.pipeline.load_neo4j.get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "NEO4J_URI": "bolt://localhost:7687",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
            },
        )(),
    )
    monkeypatch.setattr(
        "ads_scholargraph.pipeline.load_neo4j._parse_cypher_statements",
        lambda _: [],
    )
    def fake_load_relationship_rows(
        path: Path,
        *,
        required_strings: tuple[str, ...],
        optional_ints: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        if path.name == "citations.parquet":
            return [
                {
                    "source_bibcode": "B1",
                    "target_bibcode": "PX1",
                    "relation_type": "CITES",
                },
                {
                    "source_bibcode": "B1",
                    "target_bibcode": "PX2",
                    "relation_type": "NOT_CITES",
                },
            ]
        return []

    monkeypatch.setattr(
        "ads_scholargraph.pipeline.load_neo4j._load_relationship_rows",
        fake_load_relationship_rows,
    )

    def fake_read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
        if path.name == "papers.parquet":
            return pd.DataFrame(
                {
                    "bibcode": ["B1", "B2", "B3"],
                    "title": ["T1", "T2", "T3"],
                    "year": ["2025", "bad-year", " 2024 "],
                    "abstract": [None, None, None],
                    "doi": [None, None, None],
                    "citation_count": [1, 2, 3],
                }
            )
        raise AssertionError(f"Unexpected parquet path: {path}")

    monkeypatch.setattr(
        "ads_scholargraph.pipeline.load_neo4j.pd.read_parquet",
        fake_read_parquet,
    )
    (indir / "citations.parquet").touch()

    load_kg(indir=indir, wipe=False, batch_size=1000)

    paper_call = next(call for call in fake_session.calls if call[0] == PAPER_QUERY)
    rows = paper_call[1]["rows"]

    assert rows[0]["title"] == "T1"
    assert rows[0]["year"] == 2025
    assert isinstance(rows[0]["year"], int)
    assert rows[0]["is_seed"] is True
    assert rows[1]["year"] is None
    assert rows[1]["is_seed"] is True
    assert rows[2]["year"] == 2024
    assert isinstance(rows[2]["year"], int)
    assert rows[2]["is_seed"] is True

    citation_call = next(call for call in fake_session.calls if call[0] == CITATION_QUERY)
    citation_rows = citation_call[1]["rows"]
    assert citation_rows == [
        {"source_bibcode": "B1", "target_bibcode": "PX1", "relation_type": "CITES"}
    ]
    assert "ON CREATE SET src.is_seed = false" in CITATION_QUERY
    assert "ON CREATE SET dst.is_seed = false" in CITATION_QUERY
