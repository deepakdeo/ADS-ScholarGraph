"""CLI for fetching ADS search results to raw JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ads_scholargraph.ads.client import DEFAULT_FIELDS, ADSClient


def parse_fields(raw_fields: str | None) -> list[str] | None:
    """Parse a comma-separated field string into a list of ADS fields."""

    if raw_fields is None:
        return None

    fields = [field.strip() for field in raw_fields.split(",") if field.strip()]
    return fields or None


def extract_ads_results(
    *,
    client: ADSClient,
    query: str,
    out_path: Path,
    rows: int,
    max_results: int,
    sort: str | None,
    fields: list[str] | None,
    force: bool,
) -> int:
    """Fetch and write ADS records as one JSON object per line."""

    if out_path.exists() and not force:
        print(f"Raw file exists at {out_path}; skipping fetch (use --force to refetch).")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for doc in client.iter_results(
            query,
            fl=fields,
            rows=rows,
            max_results=max_results,
            sort=sort,
        ):
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1

    return written


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for ADS extraction."""

    parser = argparse.ArgumentParser(description="Fetch ADS search results into raw JSONL format.")
    parser.add_argument("--query", required=True, help="ADS query string")
    parser.add_argument("--rows", type=int, default=200, help="Rows per ADS request page")
    parser.add_argument(
        "--max-results",
        type=int,
        default=2000,
        help="Maximum total records to fetch",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL path (for example: data/raw/foo.jsonl)",
    )
    parser.add_argument("--sort", default=None, help="Optional ADS sort expression")
    parser.add_argument(
        "--fl",
        default=",".join(DEFAULT_FIELDS),
        help="Comma-separated ADS fields to request",
    )
    parser.add_argument("--force", action="store_true", help="Refetch even if output file exists")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.rows <= 0:
        parser.error("--rows must be greater than 0")
    if args.max_results <= 0:
        parser.error("--max-results must be greater than 0")

    client = ADSClient.from_settings()
    out_path = Path(args.out)
    fields = parse_fields(args.fl)

    written = extract_ads_results(
        client=client,
        query=args.query,
        out_path=out_path,
        rows=args.rows,
        max_results=args.max_results,
        sort=args.sort,
        fields=fields,
        force=args.force,
    )

    status = {
        "query": args.query,
        "rows": args.rows,
        "max_results": args.max_results,
        "out": str(out_path),
        "written": written,
    }
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
