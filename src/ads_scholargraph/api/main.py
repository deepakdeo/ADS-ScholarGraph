"""FastAPI backend for KG paper search and recommendations."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel

from ads_scholargraph.config import get_settings
from ads_scholargraph.recsys.embedding_recommender import EmbeddingRecommender
from ads_scholargraph.recsys.graph_recommender import recommend_similar_papers_graph
from ads_scholargraph.recsys.hybrid import HybridRecommender

if TYPE_CHECKING:
    from neo4j import Driver
else:
    Driver = Any

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import Neo4jError
except ModuleNotFoundError:
    GraphDatabase = None
    Neo4jError = Exception

app = FastAPI(title="ADS ScholarGraph API", version="0.1.0")


class PaperResponse(BaseModel):
    bibcode: str
    title: str | None
    year: int | None
    abstract: str | None
    is_seed: bool
    pagerank: float | None
    community_id: int | None


class RecommendationResponse(BaseModel):
    bibcode: str
    title: str | None
    year: int | None
    score: float
    reasons: list[str]


class SearchResult(BaseModel):
    bibcode: str
    title: str | None
    year: int | None


class Neo4jRepository:
    """Read-only Neo4j access layer for API routes."""

    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    @classmethod
    def from_settings(cls) -> Neo4jRepository:
        settings = get_settings()
        if GraphDatabase is None:
            raise RuntimeError("neo4j package is not installed. Install dependencies to run API.")

        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        return cls(driver)

    def get_paper(self, bibcode: str) -> dict[str, Any] | None:
        query = """
        MATCH (p:Paper {bibcode: $bibcode})
        RETURN p.bibcode AS bibcode,
               p.title AS title,
               p.year AS year,
               p.abstract AS abstract,
               coalesce(p.is_seed, false) AS is_seed,
               p.pagerank AS pagerank,
               p.community_id AS community_id
        LIMIT 1
        """
        with self._driver.session() as session:
            record = session.run(query, bibcode=bibcode).single()
            return dict(record) if record else None

    def search_seed_papers(self, q: str, limit: int) -> list[dict[str, Any]]:
        fulltext_query = """
        CALL db.index.fulltext.queryNodes('paper_text_idx', $q) YIELD node, score
        WHERE node:Paper AND coalesce(node.is_seed, false) = true
        RETURN node.bibcode AS bibcode,
               node.title AS title,
               node.year AS year,
               score
        ORDER BY score DESC, node.pagerank DESC
        LIMIT $limit
        """

        fallback_query = """
        MATCH (p:Paper)
        WHERE coalesce(p.is_seed, false) = true
          AND (
            toLower(coalesce(p.title, '')) CONTAINS toLower($q)
            OR toLower(coalesce(p.abstract, '')) CONTAINS toLower($q)
          )
        RETURN p.bibcode AS bibcode,
               p.title AS title,
               p.year AS year,
               coalesce(p.pagerank, 0.0) AS score
        ORDER BY score DESC, p.year DESC
        LIMIT $limit
        """

        with self._driver.session() as session:
            try:
                rows = [dict(record) for record in session.run(fulltext_query, q=q, limit=limit)]
                if rows:
                    return rows
            except Neo4jError:
                pass

            return [dict(record) for record in session.run(fallback_query, q=q, limit=limit)]


def _normalize_recommendations(rows: list[dict[str, Any]]) -> list[RecommendationResponse]:
    normalized: list[RecommendationResponse] = []
    for row in rows:
        bibcode = row.get("bibcode")
        if not isinstance(bibcode, str):
            continue

        title = row.get("title") if isinstance(row.get("title"), str) else None
        year = row.get("year") if isinstance(row.get("year"), int) else None
        score_raw = row.get("score")
        score = float(score_raw) if isinstance(score_raw, (int, float)) else 0.0
        reasons_raw = row.get("reasons")
        if isinstance(reasons_raw, list):
            reasons = [reason for reason in reasons_raw if isinstance(reason, str)]
        else:
            reasons = []

        normalized.append(
            RecommendationResponse(
                bibcode=bibcode,
                title=title,
                year=year,
                score=score,
                reasons=reasons,
            )
        )

    return normalized


@lru_cache(maxsize=1)
def _embedding_recommender() -> EmbeddingRecommender:
    return EmbeddingRecommender()


@lru_cache(maxsize=1)
def _hybrid_recommender() -> HybridRecommender:
    return HybridRecommender(embedding_recommender=_embedding_recommender())


def get_repository() -> Neo4jRepository:
    return Neo4jRepository.from_settings()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/paper/{bibcode}", response_model=PaperResponse)
def get_paper_endpoint(
    bibcode: str,
    repository: Neo4jRepository = Depends(get_repository),  # noqa: B008
) -> PaperResponse:
    row = repository.get_paper(bibcode)
    if row is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    row_bibcode = row.get("bibcode")
    if not isinstance(row_bibcode, str):
        raise HTTPException(status_code=500, detail="Paper payload missing bibcode")

    row_pagerank = row.get("pagerank")
    pagerank = float(row_pagerank) if isinstance(row_pagerank, (int, float)) else None

    return PaperResponse(
        bibcode=row_bibcode,
        title=row.get("title") if isinstance(row.get("title"), str) else None,
        year=row.get("year") if isinstance(row.get("year"), int) else None,
        abstract=row.get("abstract") if isinstance(row.get("abstract"), str) else None,
        is_seed=bool(row.get("is_seed")),
        pagerank=pagerank,
        community_id=row.get("community_id") if isinstance(row.get("community_id"), int) else None,
    )


@app.get("/recommend/paper/{bibcode}", response_model=list[RecommendationResponse])
def recommend_paper_endpoint(
    bibcode: str,
    k: int = Query(10, ge=1, le=100),
    mode: Literal["graph", "embed", "hybrid"] = Query("hybrid"),
) -> list[RecommendationResponse]:
    if mode == "graph":
        rows = recommend_similar_papers_graph(
            seed_bibcode=bibcode,
            k=k,
            candidate_pool=max(200, k * 10),
        )
    elif mode == "embed":
        rows = _embedding_recommender().recommend_similar_papers_embedding(
            seed_bibcode=bibcode,
            k=k,
        )
    else:
        rows = _hybrid_recommender().recommend_similar_papers_hybrid(
            seed_bibcode=bibcode,
            k=k,
            candidate_pool=max(200, k * 10),
        )

    return _normalize_recommendations(rows)


@app.get("/search", response_model=list[SearchResult])
def search_endpoint(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    repository: Neo4jRepository = Depends(get_repository),  # noqa: B008
) -> list[SearchResult]:
    rows = repository.search_seed_papers(q=q, limit=limit)

    results: list[SearchResult] = []
    for row in rows:
        bibcode = row.get("bibcode")
        if not isinstance(bibcode, str):
            continue

        results.append(
            SearchResult(
                bibcode=bibcode,
                title=row.get("title") if isinstance(row.get("title"), str) else None,
                year=row.get("year") if isinstance(row.get("year"), int) else None,
            )
        )

    return results
