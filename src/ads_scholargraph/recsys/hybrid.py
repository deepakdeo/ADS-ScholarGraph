"""Hybrid recommender combining graph and embedding signals."""

from __future__ import annotations

from typing import Any, Protocol

from ads_scholargraph.recsys.embedding_recommender import EmbeddingRecommender
from ads_scholargraph.recsys.graph_recommender import recommend_similar_papers_graph


class EmbeddingScorer(Protocol):
    """Protocol for embedding similarity scoring used by hybrid reranker."""

    def similarity_for_candidates(
        self,
        seed_bibcode: str,
        candidate_bibcodes: list[str],
    ) -> dict[str, float]:
        """Return similarity values keyed by candidate bibcode."""


class HybridRecommender:
    """Hybrid recommendation with graph candidate generation and embedding rerank."""

    def __init__(self, embedding_recommender: EmbeddingScorer | None = None) -> None:
        self._embedding = embedding_recommender or EmbeddingRecommender()

    @staticmethod
    def _normalize(values: dict[str, float]) -> dict[str, float]:
        if not values:
            return {}
        max_value = max(values.values())
        if max_value <= 0.0:
            return {key: 0.0 for key in values}
        return {key: value / max_value for key, value in values.items()}

    def recommend_similar_papers_hybrid(
        self,
        seed_bibcode: str,
        k: int = 10,
        candidate_pool: int = 200,
    ) -> list[dict[str, Any]]:
        """Return hybrid recommendations with human-readable reasons."""

        if k <= 0:
            return []

        graph_candidates = recommend_similar_papers_graph(
            seed_bibcode=seed_bibcode,
            k=candidate_pool,
            candidate_pool=candidate_pool,
        )
        if not graph_candidates:
            return []

        candidate_bibcodes = [candidate["bibcode"] for candidate in graph_candidates]
        embed_scores = self._embedding.similarity_for_candidates(seed_bibcode, candidate_bibcodes)

        graph_scores = {
            candidate["bibcode"]: float(candidate.get("score", 0.0))
            for candidate in graph_candidates
        }
        pagerank_scores = {
            candidate["bibcode"]: float(candidate.get("pagerank", 0.0))
            for candidate in graph_candidates
        }

        graph_norm = self._normalize(graph_scores)
        embed_norm = self._normalize(embed_scores)
        pagerank_norm = self._normalize(pagerank_scores)

        ranked: list[dict[str, Any]] = []
        for candidate in graph_candidates:
            bibcode = candidate["bibcode"]
            graph_component = graph_norm.get(bibcode, 0.0)
            embed_component = embed_norm.get(bibcode, 0.0)
            pagerank_component = pagerank_norm.get(bibcode, 0.0)

            score = 0.55 * graph_component + 0.4 * embed_component + 0.05 * pagerank_component

            reasons = list(candidate.get("reasons", []))
            if embed_component > 0.0:
                reasons.append("High abstract similarity")

            ranked.append(
                {
                    "bibcode": bibcode,
                    "title": candidate.get("title"),
                    "year": candidate.get("year"),
                    "score": round(score, 6),
                    "reasons": reasons,
                }
            )

        ranked.sort(key=lambda row: float(row["score"]), reverse=True)
        return ranked[:k]
