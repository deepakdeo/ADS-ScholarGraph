"""CLI for loading normalized parquet tables into Neo4j."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from neo4j import Driver, Session
else:
    Driver = Any
    Session = Any

try:
    from neo4j import GraphDatabase
except ModuleNotFoundError:
    GraphDatabase = None

from ads_scholargraph.config import get_settings

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_FILE = REPO_ROOT / "kg" / "schema.cypher"

PAPER_QUERY = """
UNWIND $rows AS row
MERGE (p:Paper {bibcode: row.bibcode})
SET p.title = row.title,
    p.year = row.year,
    p.abstract = row.abstract,
    p.doi = row.doi,
    p.citation_count = row.citation_count,
    p.is_seed = row.is_seed
"""

AUTHOR_QUERY = """
UNWIND $rows AS row
MATCH (p:Paper {bibcode: row.bibcode})
MERGE (a:Author {name: row.author_name})
MERGE (a)-[w:WROTE]->(p)
SET w.author_order = row.author_order
"""

KEYWORD_QUERY = """
UNWIND $rows AS row
MATCH (p:Paper {bibcode: row.bibcode})
MERGE (k:Keyword {name: row.keyword})
MERGE (p)-[:HAS_KEYWORD]->(k)
"""

VENUE_QUERY = """
UNWIND $rows AS row
MATCH (p:Paper {bibcode: row.bibcode})
MERGE (v:Venue {name: row.venue})
MERGE (p)-[:PUBLISHED_IN]->(v)
"""

CITATION_QUERY = """
UNWIND $rows AS row
MERGE (src:Paper {bibcode: row.source_bibcode})
ON CREATE SET src.is_seed = false
MERGE (dst:Paper {bibcode: row.target_bibcode})
ON CREATE SET dst.is_seed = false
MERGE (src)-[:CITES]->(dst)
"""


def _clean_string(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return None


def _clean_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _batched(rows: list[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    for index in range(0, len(rows), batch_size):
        yield rows[index : index + batch_size]


def _parse_cypher_statements(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [statement.strip() for statement in text.split(";") if statement.strip()]


def _to_nullable_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    cleaned = df.where(pd.notnull(df), None)
    return list(cleaned.to_dict(orient="records"))


def _load_papers(indir: Path) -> list[dict[str, Any]]:
    papers_path = indir / "papers.parquet"
    papers_df = pd.read_parquet(
        papers_path,
        columns=["bibcode", "title", "year", "abstract", "doi", "citation_count"],
    )

    rows: list[dict[str, Any]] = []
    for row in _to_nullable_records(papers_df):
        bibcode = _clean_string(row.get("bibcode"))
        if not bibcode:
            continue
        rows.append(
            {
                "bibcode": bibcode,
                "title": _clean_string(row.get("title")),
                "year": _clean_int(row.get("year")),
                "abstract": _clean_string(row.get("abstract")),
                "doi": _clean_string(row.get("doi")),
                "citation_count": _clean_int(row.get("citation_count")),
                "is_seed": True,
            }
        )
    return rows


def _load_relationship_rows(
    path: Path,
    *,
    required_strings: tuple[str, ...],
    optional_ints: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    frame = pd.read_parquet(path)

    rows: list[dict[str, Any]] = []
    for row in _to_nullable_records(frame):
        valid = True
        output: dict[str, Any] = {}

        for key in required_strings:
            value = _clean_string(row.get(key))
            if not value:
                valid = False
                break
            output[key] = value

        if not valid:
            continue

        for key in optional_ints:
            output[key] = _clean_int(row.get(key))

        rows.append(output)

    return rows


def _run_batched_write(
    session: Session,
    query: str,
    rows: list[dict[str, Any]],
    batch_size: int,
) -> None:
    for batch in _batched(rows, batch_size):
        session.run(query, rows=batch).consume()


def load_kg(
    *,
    indir: Path,
    wipe: bool,
    batch_size: int,
) -> dict[str, int]:
    """Load parquet tables from indir into Neo4j."""

    settings = get_settings()
    if GraphDatabase is None:
        raise RuntimeError("neo4j package is not installed. Install dependencies to load Neo4j.")
    driver: Driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )

    papers = _load_papers(indir)
    paper_authors = _load_relationship_rows(
        indir / "paper_authors.parquet",
        required_strings=("bibcode", "author_name"),
        optional_ints=("author_order",),
    )
    paper_keywords = _load_relationship_rows(
        indir / "paper_keywords.parquet",
        required_strings=("bibcode", "keyword"),
    )
    paper_venues = _load_relationship_rows(
        indir / "paper_venues.parquet",
        required_strings=("bibcode", "venue"),
    )

    citations_path = indir / "citations.parquet"
    citations: list[dict[str, Any]] = []
    if citations_path.exists():
        citation_rows = _load_relationship_rows(
            citations_path,
            required_strings=("source_bibcode", "target_bibcode", "relation_type"),
        )
        citations = [row for row in citation_rows if row.get("relation_type") == "CITES"]

    schema_statements = _parse_cypher_statements(SCHEMA_FILE)

    with driver:
        with driver.session() as session:
            if wipe:
                session.run("MATCH (n) DETACH DELETE n").consume()

            for statement in schema_statements:
                session.run(statement).consume()

            _run_batched_write(session, PAPER_QUERY, papers, batch_size)
            _run_batched_write(session, AUTHOR_QUERY, paper_authors, batch_size)
            _run_batched_write(session, KEYWORD_QUERY, paper_keywords, batch_size)
            _run_batched_write(session, VENUE_QUERY, paper_venues, batch_size)
            if citations:
                _run_batched_write(session, CITATION_QUERY, citations, batch_size)

    return {
        "papers": len(papers),
        "paper_authors": len(paper_authors),
        "paper_keywords": len(paper_keywords),
        "paper_venues": len(paper_venues),
        "citations": len(citations),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for Neo4j load CLI."""

    parser = argparse.ArgumentParser(description="Load processed parquet tables into Neo4j.")
    parser.add_argument("--indir", required=True, help="Input processed directory")
    parser.add_argument("--wipe", action="store_true", help="Wipe existing Neo4j data first")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Rows per UNWIND batch",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")

    indir = Path(args.indir)
    if not indir.exists():
        parser.error(f"Input directory does not exist: {indir}")

    counts = load_kg(indir=indir, wipe=args.wipe, batch_size=args.batch_size)
    print(json.dumps({"indir": str(indir), "wipe": args.wipe, "counts": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
