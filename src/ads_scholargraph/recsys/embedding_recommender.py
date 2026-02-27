"""Embedding-based recommendations using TF-IDF over paper title/abstract."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ads_scholargraph.config import get_settings

if TYPE_CHECKING:
    from neo4j import Driver
else:
    Driver = Any

try:
    from neo4j import GraphDatabase
except ModuleNotFoundError:
    GraphDatabase = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError:
    TfidfVectorizer = None
    cosine_similarity = None

SEED_PAPER_QUERY = """
MATCH (p:Paper {is_seed: true})
RETURN p.bibcode AS bibcode,
       p.title AS title,
       p.abstract AS abstract,
       p.year AS year,
       coalesce(p.pagerank, 0.0) AS pagerank
"""


@dataclass
class _EmbeddingIndex:
    bibcodes: list[str]
    metadata: dict[str, dict[str, Any]]
    vectorizer: Any
    matrix: Any


class EmbeddingRecommender:
    """TF-IDF recommender with on-disk cache for seed-paper embeddings."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        if TfidfVectorizer is None or cosine_similarity is None:
            raise RuntimeError(
                "scikit-learn is required for embedding recommender. Install dependencies."
            )

        self._cache_dir = cache_dir or Path(".cache/ads_scholargraph/recsys")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path = self._cache_dir / "tfidf_index.pkl"
        self._index: _EmbeddingIndex | None = None

    def _fetch_seed_papers(self) -> list[dict[str, Any]]:
        settings = get_settings()
        if GraphDatabase is None:
            raise RuntimeError(
                "neo4j package is not installed. Install dependencies to query Neo4j."
            )

        driver: Driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        with driver:
            with driver.session() as session:
                return [dict(record) for record in session.run(SEED_PAPER_QUERY)]

    def _build_index(self, rows: list[dict[str, Any]]) -> _EmbeddingIndex:
        bibcodes: list[str] = []
        docs: list[str] = []
        metadata: dict[str, dict[str, Any]] = {}

        for row in rows:
            bibcode = row.get("bibcode")
            if not isinstance(bibcode, str):
                continue

            title = row.get("title") if isinstance(row.get("title"), str) else ""
            abstract = row.get("abstract") if isinstance(row.get("abstract"), str) else ""
            year = row.get("year") if isinstance(row.get("year"), int) else None
            raw_pagerank = row.get("pagerank")
            pagerank = float(raw_pagerank) if isinstance(raw_pagerank, (int, float)) else 0.0

            doc = f"{title} {abstract}".strip()
            if not doc:
                doc = title or bibcode

            bibcodes.append(bibcode)
            docs.append(doc)
            metadata[bibcode] = {
                "title": title or None,
                "year": year,
                "pagerank": pagerank,
            }

        vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
        matrix = vectorizer.fit_transform(docs)

        return _EmbeddingIndex(
            bibcodes=bibcodes,
            metadata=metadata,
            vectorizer=vectorizer,
            matrix=matrix,
        )

    def _load_or_build_index(self) -> _EmbeddingIndex:
        if self._index is not None:
            return self._index

        if self._cache_path.exists():
            with self._cache_path.open("rb") as handle:
                self._index = pickle.load(handle)
            return self._index

        rows = self._fetch_seed_papers()
        index = self._build_index(rows)
        with self._cache_path.open("wb") as handle:
            pickle.dump(index, handle)
        self._index = index
        return index

    def similarity_for_candidates(
        self,
        seed_bibcode: str,
        candidate_bibcodes: list[str],
    ) -> dict[str, float]:
        """Return cosine similarity scores from seed to provided candidate bibcodes."""

        index = self._load_or_build_index()
        if seed_bibcode not in index.metadata:
            return {bibcode: 0.0 for bibcode in candidate_bibcodes}

        bibcode_to_idx = {bibcode: idx for idx, bibcode in enumerate(index.bibcodes)}
        seed_idx = bibcode_to_idx.get(seed_bibcode)
        if seed_idx is None:
            return {bibcode: 0.0 for bibcode in candidate_bibcodes}

        similarities = cosine_similarity(index.matrix[seed_idx], index.matrix).ravel()

        scores: dict[str, float] = {}
        for bibcode in candidate_bibcodes:
            idx = bibcode_to_idx.get(bibcode)
            score = float(similarities[idx]) if idx is not None else 0.0
            scores[bibcode] = score
        return scores

    def recommend_similar_papers_embedding(
        self,
        seed_bibcode: str,
        k: int = 10,
    ) -> list[dict[str, Any]]:
        """Recommend similar papers using TF-IDF cosine similarity."""

        if k <= 0:
            return []

        index = self._load_or_build_index()
        bibcode_to_idx = {bibcode: idx for idx, bibcode in enumerate(index.bibcodes)}
        seed_idx = bibcode_to_idx.get(seed_bibcode)
        if seed_idx is None:
            return []

        similarities = cosine_similarity(index.matrix[seed_idx], index.matrix).ravel()
        similarities[seed_idx] = -1.0

        ranked_idx = similarities.argsort()[::-1]
        results: list[dict[str, Any]] = []
        for idx in ranked_idx:
            if len(results) >= k:
                break

            bibcode = index.bibcodes[idx]
            if bibcode == seed_bibcode:
                continue

            score = float(similarities[idx])
            if score <= 0.0:
                continue

            meta = index.metadata.get(bibcode, {})
            results.append(
                {
                    "bibcode": bibcode,
                    "title": meta.get("title"),
                    "year": meta.get("year"),
                    "score": round(score, 6),
                    "reasons": ["High TF-IDF similarity"],
                }
            )

        return results
