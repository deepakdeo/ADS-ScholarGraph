from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ads_scholargraph.pipeline.expand_citations import expand_edges, run_expansion

pytest.importorskip("pyarrow")


class FakeADSClient:
    def __init__(self, responses: dict[str, list[dict[str, Any]]]) -> None:
        self._responses = responses
        self.queries: list[str] = []

    def iter_results(
        self,
        q: str,
        *,
        fl: list[str] | tuple[str, ...] | None = None,
        rows: int = 200,
        max_results: int = 2000,
        sort: str | None = None,
    ):
        self.queries.append(q)
        docs = self._responses.get(q, [])
        return iter(docs[:max_results])


def test_expand_edges_references_mode_orients_source_to_target() -> None:
    responses = {
        'references(bibcode:"A")': [{"bibcode": "R1"}, {"bibcode": "R2"}],
        'references(bibcode:"B")': [{"bibcode": "R2"}, {"bibcode": "R3"}],
    }
    client = FakeADSClient(responses)

    edges = expand_edges(
        seed_bibcodes=["A", "B"],
        client=client,
        mode="references",
        max_per_paper=10,
    )

    assert set(edges["source_bibcode"]) == {"A", "B"}
    triples = zip(
        edges["source_bibcode"],
        edges["target_bibcode"],
        edges["relation_type"],
        strict=False,
    )
    assert set(triples) == {
        ("A", "R1", "CITES"),
        ("A", "R2", "CITES"),
        ("B", "R2", "CITES"),
        ("B", "R3", "CITES"),
    }


def test_run_expansion_writes_parquet_and_supports_cache(tmp_path: Path) -> None:
    seed_path = tmp_path / "papers.parquet"
    out_path = tmp_path / "citations.parquet"

    pd.DataFrame({"bibcode": ["A", "B"]}).to_parquet(seed_path, index=False)

    responses = {
        'citations(bibcode:"A")': [{"bibcode": "C1"}],
        'citations(bibcode:"B")': [{"bibcode": "C2"}],
    }
    client = FakeADSClient(responses)

    result = run_expansion(
        seed_path=seed_path,
        out_path=out_path,
        mode="citations",
        max_per_paper=5,
        force=False,
        client=client,
    )

    assert result["skipped"] is False
    assert result["edges"] == 2
    assert out_path.exists()

    citations = pd.read_parquet(out_path)
    citation_pairs = zip(
        citations["source_bibcode"],
        citations["target_bibcode"],
        strict=False,
    )
    assert set(citation_pairs) == {
        ("C1", "A"),
        ("C2", "B"),
    }

    second = run_expansion(
        seed_path=seed_path,
        out_path=out_path,
        mode="citations",
        max_per_paper=5,
        force=False,
        client=client,
    )

    assert second["skipped"] is True
    assert second["edges"] == 0
