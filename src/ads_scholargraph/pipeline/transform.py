"""CLI for transforming raw ADS JSONL into normalized parquet tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    output.append(stripped)
        return output
    return []


def _first_string(value: Any) -> str | None:
    values = _as_string_list(value)
    return values[0] if values else None


def load_jsonl(input_path: Path) -> list[dict[str, Any]]:
    """Load JSON Lines input where each line is a single ADS document object."""

    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            records.append(payload)
    return records


def normalize_records(records: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Convert ADS docs into normalized tables for papers/authors/keywords/venues."""

    paper_rows: list[dict[str, Any]] = []
    seen_bibcodes: set[str] = set()

    author_names: list[str] = []
    seen_authors: set[str] = set()
    paper_author_rows: list[dict[str, Any]] = []
    seen_paper_authors: set[tuple[str, str, int]] = set()

    keyword_values: list[str] = []
    seen_keywords: set[str] = set()
    paper_keyword_rows: list[dict[str, str]] = []
    seen_paper_keywords: set[tuple[str, str]] = set()

    venue_values: list[str] = []
    seen_venues: set[str] = set()
    paper_venue_rows: list[dict[str, str]] = []
    seen_paper_venues: set[tuple[str, str]] = set()

    for record in records:
        bibcode = record.get("bibcode")
        if not isinstance(bibcode, str) or not bibcode.strip():
            continue
        bibcode = bibcode.strip()

        if bibcode not in seen_bibcodes:
            seen_bibcodes.add(bibcode)
            abstract = record.get("abstract")
            paper_rows.append(
                {
                    "bibcode": bibcode,
                    "title": _first_string(record.get("title")),
                    "year": record.get("year"),
                    "abstract": abstract if isinstance(abstract, str) else None,
                    "doi": _first_string(record.get("doi")),
                    "citation_count": record.get("citation_count"),
                }
            )

        paper_authors = _as_string_list(record.get("author"))
        for order, author_name in enumerate(paper_authors, start=1):
            if author_name not in seen_authors:
                seen_authors.add(author_name)
                author_names.append(author_name)

            paper_author_key = (bibcode, author_name, order)
            if paper_author_key not in seen_paper_authors:
                seen_paper_authors.add(paper_author_key)
                paper_author_rows.append(
                    {
                        "bibcode": bibcode,
                        "author_name": author_name,
                        "author_order": order,
                    }
                )

        keywords = _as_string_list(record.get("keyword"))
        for keyword in keywords:
            if keyword not in seen_keywords:
                seen_keywords.add(keyword)
                keyword_values.append(keyword)

            paper_keyword_key = (bibcode, keyword)
            if paper_keyword_key not in seen_paper_keywords:
                seen_paper_keywords.add(paper_keyword_key)
                paper_keyword_rows.append(
                    {
                        "bibcode": bibcode,
                        "keyword": keyword,
                    }
                )

        venue = _first_string(record.get("pub"))
        if venue:
            if venue not in seen_venues:
                seen_venues.add(venue)
                venue_values.append(venue)

            paper_venue_key = (bibcode, venue)
            if paper_venue_key not in seen_paper_venues:
                seen_paper_venues.add(paper_venue_key)
                paper_venue_rows.append(
                    {
                        "bibcode": bibcode,
                        "venue": venue,
                    }
                )

    author_dimension = [
        {"author_id": index, "author_name": author_name}
        for index, author_name in enumerate(author_names, start=1)
    ]
    keyword_dimension = [
        {"keyword_id": index, "keyword": keyword}
        for index, keyword in enumerate(keyword_values, start=1)
    ]
    venue_dimension = [
        {"venue_id": index, "venue": venue}
        for index, venue in enumerate(venue_values, start=1)
    ]

    return {
        "papers": pd.DataFrame(
            paper_rows,
            columns=["bibcode", "title", "year", "abstract", "doi", "citation_count"],
        ),
        "authors": pd.DataFrame(author_dimension, columns=["author_id", "author_name"]),
        "paper_authors": pd.DataFrame(
            paper_author_rows,
            columns=["bibcode", "author_name", "author_order"],
        ),
        "keywords": pd.DataFrame(keyword_dimension, columns=["keyword_id", "keyword"]),
        "paper_keywords": pd.DataFrame(paper_keyword_rows, columns=["bibcode", "keyword"]),
        "venues": pd.DataFrame(venue_dimension, columns=["venue_id", "venue"]),
        "paper_venues": pd.DataFrame(paper_venue_rows, columns=["bibcode", "venue"]),
    }


def write_tables(tables: dict[str, pd.DataFrame], outdir: Path) -> dict[str, int]:
    """Write normalized tables to parquet files and return row counts."""

    outdir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for table_name, table in tables.items():
        table_path = outdir / f"{table_name}.parquet"
        table.to_parquet(table_path, index=False)
        counts[table_name] = len(table)
    return counts


def transform_file(input_path: Path, outdir: Path) -> dict[str, int]:
    """Load raw JSONL, normalize records, and write parquet tables."""

    records = load_jsonl(input_path)
    tables = normalize_records(records)
    return write_tables(tables, outdir)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for transform step."""

    parser = argparse.ArgumentParser(
        description="Normalize ADS JSONL records into parquet tables."
    )
    parser.add_argument("--in", dest="input_path", required=True, help="Input raw JSONL path")
    parser.add_argument("--outdir", required=True, help="Output directory for parquet tables")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input_path)
    if not input_path.exists():
        parser.error(f"Input file does not exist: {input_path}")

    outdir = Path(args.outdir)
    counts = transform_file(input_path, outdir)
    print(
        json.dumps(
            {"input": str(input_path), "outdir": str(outdir), "counts": counts},
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
