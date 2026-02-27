from pathlib import Path

import pytest

from ads_scholargraph.pipeline.transform import load_jsonl, normalize_records, transform_file


def test_normalize_records_produces_expected_relationships() -> None:
    records = load_jsonl(Path("tests/fixtures/ads_sample.jsonl"))
    tables = normalize_records(records)

    assert set(tables["papers"]["bibcode"].tolist()) == {"2024ApJ...0001A", "2023MNRAS.0002B"}
    assert set(tables["authors"]["author_name"].tolist()) == {"Alice A", "Bob B", "Cara C"}
    assert set(tables["keywords"]["keyword"].tolist()) == {"stars", "galaxies", "cosmology"}
    assert set(tables["venues"]["venue"].tolist()) == {"Astrophysical Journal", "MNRAS"}

    first_paper_authors = tables["paper_authors"][
        tables["paper_authors"]["bibcode"] == "2024ApJ...0001A"
    ].sort_values(by="author_order")
    assert first_paper_authors["author_name"].tolist() == ["Alice A", "Bob B"]


def test_transform_file_creates_normalized_tables(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")

    input_path = Path("tests/fixtures/ads_sample.jsonl")
    outdir = tmp_path / "normalized"

    counts = transform_file(input_path=input_path, outdir=outdir)

    assert counts == {
        "papers": 2,
        "authors": 3,
        "paper_authors": 4,
        "keywords": 3,
        "paper_keywords": 4,
        "venues": 2,
        "paper_venues": 2,
    }
    assert (outdir / "papers.parquet").exists()
    assert (outdir / "authors.parquet").exists()
    assert (outdir / "paper_authors.parquet").exists()
    assert (outdir / "keywords.parquet").exists()
    assert (outdir / "paper_keywords.parquet").exists()
    assert (outdir / "venues.parquet").exists()
    assert (outdir / "paper_venues.parquet").exists()
