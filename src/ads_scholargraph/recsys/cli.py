"""CLI entrypoint for graph, embedding, and hybrid recommendations."""

from __future__ import annotations

import argparse
from typing import Any

from ads_scholargraph.recsys.embedding_recommender import EmbeddingRecommender
from ads_scholargraph.recsys.graph_recommender import recommend_similar_papers_graph
from ads_scholargraph.recsys.hybrid import HybridRecommender


def _format_table(rows: list[dict[str, Any]]) -> str:
    headers = ["bibcode", "year", "title", "score", "reasons"]

    formatted_rows: list[list[str]] = []
    for row in rows:
        formatted_rows.append(
            [
                str(row.get("bibcode", "")),
                str(row.get("year", "")) if row.get("year") is not None else "",
                str(row.get("title", "") or ""),
                f"{float(row.get('score', 0.0)):.4f}",
                "; ".join(row.get("reasons", [])),
            ]
        )

    widths = [len(header) for header in headers]
    for formatted in formatted_rows:
        for idx, value in enumerate(formatted):
            widths[idx] = max(widths[idx], len(value))

    def render_line(values: list[str]) -> str:
        padded = [value.ljust(widths[idx]) for idx, value in enumerate(values)]
        return " | ".join(padded)

    separator = "-+-".join("-" * width for width in widths)
    lines = [render_line(headers), separator]
    lines.extend(render_line(row) for row in formatted_rows)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate paper recommendations.")
    parser.add_argument("--bibcode", required=True, help="Seed paper bibcode")
    parser.add_argument("--k", type=int, default=10, help="Number of recommendations")
    parser.add_argument(
        "--mode",
        choices=["graph", "embed", "hybrid"],
        default="hybrid",
        help="Recommendation mode",
    )
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=200,
        help="Graph candidate pool size (graph/hybrid)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.k <= 0:
        parser.error("--k must be greater than 0")

    if args.mode == "graph":
        rows = recommend_similar_papers_graph(
            seed_bibcode=args.bibcode,
            k=args.k,
            candidate_pool=args.candidate_pool,
        )
    elif args.mode == "embed":
        embedding_recommender = EmbeddingRecommender()
        rows = embedding_recommender.recommend_similar_papers_embedding(args.bibcode, k=args.k)
    else:
        hybrid_recommender = HybridRecommender()
        rows = hybrid_recommender.recommend_similar_papers_hybrid(
            seed_bibcode=args.bibcode,
            k=args.k,
            candidate_pool=args.candidate_pool,
        )

    print(_format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
