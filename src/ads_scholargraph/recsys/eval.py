"""Offline evaluation for graph/embed/hybrid recommenders."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd

from ads_scholargraph.recsys.embedding_recommender import EmbeddingRecommender
from ads_scholargraph.recsys.graph_recommender import recommend_similar_papers_graph
from ads_scholargraph.recsys.hybrid import HybridRecommender


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    hits = len(set(recommended[:k]).intersection(relevant))
    return hits / float(len(relevant))


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0

    dcg = 0.0
    for idx, bibcode in enumerate(recommended[:k], start=1):
        if bibcode in relevant:
            dcg += 1.0 / math.log2(idx + 1)

    ideal_hits = min(k, len(relevant))
    idcg = 0.0
    for idx in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(idx + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _build_holdout(
    citations_df: pd.DataFrame,
    holdout_fraction: float,
    min_heldout: int,
    seed: int,
) -> dict[str, set[str]]:
    rng = random.Random(seed)

    grouped: dict[str, list[str]] = {}
    for _, row in citations_df.iterrows():
        source = row.get("source_bibcode")
        target = row.get("target_bibcode")
        if isinstance(source, str) and isinstance(target, str):
            grouped.setdefault(source, []).append(target)

    holdout: dict[str, set[str]] = {}
    for source, targets in grouped.items():
        unique_targets = sorted(set(targets))
        if len(unique_targets) < max(2, min_heldout):
            continue

        holdout_n = max(1, int(len(unique_targets) * holdout_fraction))
        if holdout_n < min_heldout:
            continue
        if holdout_n >= len(unique_targets):
            holdout_n = len(unique_targets) - 1
        if holdout_n <= 0:
            continue

        sampled = set(rng.sample(unique_targets, holdout_n))
        if len(sampled) >= min_heldout:
            holdout[source] = sampled

    return holdout


def _recommend(
    mode: str,
    seed_bibcode: str,
    k: int,
    candidate_pool: int,
    hybrid: HybridRecommender,
    embed: EmbeddingRecommender,
) -> list[dict[str, Any]]:
    if mode == "graph":
        return recommend_similar_papers_graph(
            seed_bibcode=seed_bibcode,
            k=k,
            candidate_pool=candidate_pool,
        )
    if mode == "embed":
        return embed.recommend_similar_papers_embedding(seed_bibcode, k=k)
    return hybrid.recommend_similar_papers_hybrid(
        seed_bibcode=seed_bibcode,
        k=k,
        candidate_pool=candidate_pool,
    )


def run_evaluation(
    *,
    citations_path: Path,
    seed_papers_path: Path | None,
    k: int,
    holdout_fraction: float,
    min_heldout: int,
    candidate_pool: int,
    random_seed: int,
) -> dict[str, dict[str, float]]:
    """Evaluate graph/embed/hybrid modes using held-out citation targets."""

    citations_df = pd.read_parquet(citations_path)
    citations_df = citations_df[citations_df["relation_type"] == "CITES"]

    resolved_seed_papers_path = seed_papers_path or (citations_path.parent / "papers.parquet")
    if resolved_seed_papers_path.exists():
        seed_papers_df = pd.read_parquet(resolved_seed_papers_path, columns=["bibcode"])
        seed_bibcodes = {
            bibcode
            for bibcode in seed_papers_df["bibcode"].tolist()
            if isinstance(bibcode, str) and bibcode
        }
        citations_df = citations_df[citations_df["source_bibcode"].isin(seed_bibcodes)]

    holdout = _build_holdout(citations_df, holdout_fraction, min_heldout, random_seed)

    embed = EmbeddingRecommender()
    hybrid = HybridRecommender(embedding_recommender=embed)

    report: dict[str, dict[str, float]] = {}
    modes = ["graph", "embed", "hybrid"]
    for mode in modes:
        recalls: list[float] = []
        ndcgs: list[float] = []

        for seed_bibcode, relevant in holdout.items():
            recs = _recommend(mode, seed_bibcode, k, candidate_pool, hybrid, embed)
            predicted = [row["bibcode"] for row in recs if isinstance(row.get("bibcode"), str)]
            if not predicted:
                recalls.append(0.0)
                ndcgs.append(0.0)
                continue

            recalls.append(recall_at_k(predicted, relevant, k))
            ndcgs.append(ndcg_at_k(predicted, relevant, k))

        queries = len(holdout)
        report[mode] = {
            "queries": float(queries),
            f"recall@{k}": sum(recalls) / len(recalls) if recalls else 0.0,
            f"ndcg@{k}": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        }

    return report


def write_report(report: dict[str, dict[str, float]], out_path: Path, k: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Offline Recommendation Evaluation",
        "",
        f"k = {k}",
        "",
        "| Mode | Queries | Recall@k | NDCG@k |",
        "|------|---------|----------|--------|",
    ]

    for mode in ["graph", "embed", "hybrid"]:
        metrics = report.get(mode, {})
        queries = int(metrics.get("queries", 0.0))
        recall = metrics.get(f"recall@{k}", 0.0)
        ndcg = metrics.get(f"ndcg@{k}", 0.0)
        lines.append(f"| {mode} | {queries} | {recall:.4f} | {ndcg:.4f} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline recommendation evaluation.")
    parser.add_argument(
        "--citations",
        default="data/processed/ads_star/citations.parquet",
        help="Path to citations.parquet",
    )
    parser.add_argument(
        "--seed-papers",
        default=None,
        help="Optional path to papers.parquet used to restrict sources to seed papers",
    )
    parser.add_argument("--k", type=int, default=10, help="Evaluation cutoff")
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help="Fraction of citations to hold out",
    )
    parser.add_argument(
        "--min-heldout",
        type=int,
        default=2,
        help="Minimum held-out citations per seed paper",
    )
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=200,
        help="Candidate pool for graph/hybrid",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", default="docs/eval_report.md", help="Markdown report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.k <= 0:
        parser.error("--k must be greater than 0")
    if not (0.0 < args.holdout_fraction < 1.0):
        parser.error("--holdout-fraction must be between 0 and 1")

    report = run_evaluation(
        citations_path=Path(args.citations),
        seed_papers_path=Path(args.seed_papers) if args.seed_papers else None,
        k=args.k,
        holdout_fraction=args.holdout_fraction,
        min_heldout=args.min_heldout,
        candidate_pool=args.candidate_pool,
        random_seed=args.random_seed,
    )
    write_report(report, Path(args.out), args.k)

    print(f"Wrote evaluation report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
