"""CLI for ADS citation/reference expansion from seed papers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, Protocol

import pandas as pd

from ads_scholargraph.ads.client import ADSClient

ExpansionMode = Literal["references", "citations"]


class ADSResultClient(Protocol):
    """Protocol for ADS clients that can stream query results."""

    def iter_results(
        self,
        q: str,
        *,
        fl: list[str] | tuple[str, ...] | None = None,
        rows: int = 200,
        max_results: int = 2000,
        sort: str | None = None,
    ) -> Any:
        """Yield dict-like ADS docs for a query."""


def _as_bibcode(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    return stripped
    return None


def _build_operator_query(mode: ExpansionMode, bibcode: str) -> str:
    escaped = bibcode.replace("\\", "\\\\").replace('"', '\\"')
    return f'{mode}(bibcode:"{escaped}")'


def load_seed_bibcodes(seed_path: Path) -> list[str]:
    """Load unique seed bibcodes from a papers parquet file."""

    if not seed_path.exists():
        raise FileNotFoundError(f"Seed parquet does not exist: {seed_path}")

    papers = pd.read_parquet(seed_path, columns=["bibcode"])
    if "bibcode" not in papers.columns:
        raise ValueError("Seed parquet must include a 'bibcode' column")

    seen: set[str] = set()
    ordered: list[str] = []
    for raw_bibcode in papers["bibcode"].tolist():
        bibcode = _as_bibcode(raw_bibcode)
        if bibcode and bibcode not in seen:
            seen.add(bibcode)
            ordered.append(bibcode)
    return ordered


def expand_edges(
    *,
    seed_bibcodes: list[str],
    client: ADSResultClient,
    mode: ExpansionMode,
    max_per_paper: int,
) -> pd.DataFrame:
    """Expand citation edges for seed bibcodes and return a normalized edge table."""

    edge_set: set[tuple[str, str, str]] = set()
    page_size = min(200, max_per_paper)

    for seed_bibcode in seed_bibcodes:
        query = _build_operator_query(mode, seed_bibcode)
        for doc in client.iter_results(
            query,
            fl=["bibcode"],
            rows=page_size,
            max_results=max_per_paper,
        ):
            if not isinstance(doc, dict):
                continue

            related_bibcode = _as_bibcode(doc.get("bibcode"))
            if not related_bibcode:
                continue

            if mode == "references":
                source_bibcode, target_bibcode = seed_bibcode, related_bibcode
            else:
                source_bibcode, target_bibcode = related_bibcode, seed_bibcode

            edge_set.add((source_bibcode, target_bibcode, "CITES"))

    edges = sorted(edge_set)
    return pd.DataFrame(
        edges,
        columns=["source_bibcode", "target_bibcode", "relation_type"],
    )


def run_expansion(
    *,
    seed_path: Path,
    out_path: Path,
    mode: ExpansionMode,
    max_per_paper: int,
    force: bool,
    client: ADSResultClient | None = None,
) -> dict[str, Any]:
    """Run citation expansion and write citations parquet."""

    if out_path.exists() and not force:
        return {
            "seed": str(seed_path),
            "out": str(out_path),
            "mode": mode,
            "max_per_paper": max_per_paper,
            "edges": 0,
            "skipped": True,
        }

    resolved_client = client or ADSClient.from_settings()

    seed_bibcodes = load_seed_bibcodes(seed_path)
    edges = expand_edges(
        seed_bibcodes=seed_bibcodes,
        client=resolved_client,
        mode=mode,
        max_per_paper=max_per_paper,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(out_path, index=False)

    return {
        "seed": str(seed_path),
        "out": str(out_path),
        "mode": mode,
        "max_per_paper": max_per_paper,
        "seed_papers": len(seed_bibcodes),
        "edges": len(edges),
        "skipped": False,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for citation expansion."""

    parser = argparse.ArgumentParser(
        description="Expand citations/reference edges from seed papers parquet."
    )
    parser.add_argument("--seed", required=True, help="Path to seed papers.parquet")
    parser.add_argument(
        "--mode",
        choices=["references", "citations"],
        default="references",
        help="ADS expansion operator",
    )
    parser.add_argument(
        "--max-per-paper",
        type=int,
        default=200,
        help="Maximum related bibcodes fetched per seed paper",
    )
    parser.add_argument("--out", required=True, help="Output citations parquet path")
    parser.add_argument("--force", action="store_true", help="Recompute even if output exists")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_per_paper <= 0:
        parser.error("--max-per-paper must be greater than 0")

    result = run_expansion(
        seed_path=Path(args.seed),
        out_path=Path(args.out),
        mode=args.mode,
        max_per_paper=args.max_per_paper,
        force=args.force,
    )

    if result["skipped"]:
        print(f"Citations parquet exists at {args.out}; skipping (use --force to recompute).")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
