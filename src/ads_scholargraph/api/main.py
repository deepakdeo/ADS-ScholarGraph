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

_graph_database: Any | None
_neo4j_error: type[Exception]
try:
    from neo4j import GraphDatabase as _neo4j_graph_database
    from neo4j.exceptions import Neo4jError as _Neo4jError
    _graph_database = _neo4j_graph_database
    _neo4j_error = _Neo4jError
except ModuleNotFoundError:
    _graph_database = None
    _neo4j_error = Exception

app = FastAPI(title="ADS ScholarGraph API", version="0.1.0")


class PaperResponse(BaseModel):
    bibcode: str
    title: str | None
    year: int | None
    abstract: str | None
    citation_count: int | None
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


class SubgraphNode(BaseModel):
    id: str
    label: str
    type: Literal["seed", "recommended", "keyword"]
    year: int | None = None
    title: str | None = None


class SubgraphEdge(BaseModel):
    source: str
    target: str
    label: str
    type: Literal["RECOMMENDS", "CITES", "HAS_KEYWORD", "SIMILAR_TO"]
    similarity: float | None = None


class SubgraphResponse(BaseModel):
    nodes: list[SubgraphNode]
    edges: list[SubgraphEdge]


class GraphStatsSummary(BaseModel):
    paper_count: int
    author_count: int
    keyword_count: int
    venue_count: int
    cites_count: int
    avg_degree: float
    community_count: int
    graph_density: float


class GraphStatsTopPaper(BaseModel):
    bibcode: str
    title: str | None
    pagerank: float
    citation_count: int | None
    year: int | None


class GraphStatsCommunity(BaseModel):
    community_id: int
    size: int


class GraphStatsYearCount(BaseModel):
    year: int
    count: int


class GraphStatsOverview(BaseModel):
    summary: GraphStatsSummary
    top_papers: list[GraphStatsTopPaper]
    community_sizes: list[GraphStatsCommunity]
    publications_per_year: list[GraphStatsYearCount]
    citation_counts: list[int]


class Neo4jRepository:
    """Read-only Neo4j access layer for API routes."""

    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    @classmethod
    def from_settings(cls) -> Neo4jRepository:
        settings = get_settings()
        if _graph_database is None:
            raise RuntimeError("neo4j package is not installed. Install dependencies to run API.")

        driver = _graph_database.driver(
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
               p.citation_count AS citation_count,
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
            except _neo4j_error:
                pass

            return [dict(record) for record in session.run(fallback_query, q=q, limit=limit)]

    def get_citation_edges(self, bibcodes: list[str], limit: int) -> list[dict[str, Any]]:
        if not bibcodes:
            return []

        query = """
        MATCH (src:Paper)-[:CITES]->(dst:Paper)
        WHERE src.bibcode IN $bibcodes AND dst.bibcode IN $bibcodes
        RETURN DISTINCT src.bibcode AS source, dst.bibcode AS target
        LIMIT $limit
        """
        with self._driver.session() as session:
            return [
                dict(record)
                for record in session.run(query, bibcodes=bibcodes, limit=limit)
            ]

    def get_keyword_links(self, bibcodes: list[str], limit: int) -> list[dict[str, Any]]:
        if not bibcodes:
            return []

        query = """
        MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
        WHERE p.bibcode IN $bibcodes
        RETURN p.bibcode AS bibcode, k.name AS keyword
        ORDER BY keyword ASC
        LIMIT $limit
        """
        with self._driver.session() as session:
            return [
                dict(record)
                for record in session.run(query, bibcodes=bibcodes, limit=limit)
            ]

    def get_similarity_edges(self, bibcodes: list[str], limit: int) -> list[dict[str, Any]]:
        if not bibcodes:
            return []

        query = """
        MATCH (src:Paper)-[r:SIMILAR_TO]-(dst:Paper)
        WHERE src.bibcode IN $bibcodes
          AND dst.bibcode IN $bibcodes
          AND src.bibcode < dst.bibcode
        RETURN src.bibcode AS source,
               dst.bibcode AS target,
               coalesce(r.similarity, 0.0) AS similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """
        with self._driver.session() as session:
            return [
                dict(record)
                for record in session.run(query, bibcodes=bibcodes, limit=limit)
            ]

    def get_graph_stats_summary(self) -> dict[str, Any]:
        query = """
        MATCH (p:Paper)
        WITH count(p) AS paper_count
        MATCH (a:Author)
        WITH paper_count, count(a) AS author_count
        MATCH (k:Keyword)
        WITH paper_count, author_count, count(k) AS keyword_count
        MATCH (v:Venue)
        WITH paper_count, author_count, keyword_count, count(v) AS venue_count
        MATCH ()-[c:CITES]->()
        WITH paper_count, author_count, keyword_count, venue_count, count(c) AS cites_count
        MATCH (p:Paper)
        WHERE p.community_id IS NOT NULL
        RETURN paper_count,
               author_count,
               keyword_count,
               venue_count,
               cites_count,
               count(DISTINCT p.community_id) AS community_count
        """
        with self._driver.session() as session:
            record = session.run(query).single()
            return dict(record) if record else {}

    def get_top_papers_by_pagerank(self, limit: int) -> list[dict[str, Any]]:
        query = """
        MATCH (p:Paper)
        WHERE p.pagerank IS NOT NULL
        RETURN p.bibcode AS bibcode,
               p.title AS title,
               p.pagerank AS pagerank,
               p.citation_count AS citation_count,
               p.year AS year
        ORDER BY p.pagerank DESC
        LIMIT $limit
        """
        with self._driver.session() as session:
            return [dict(record) for record in session.run(query, limit=limit)]

    def get_community_sizes(self, limit: int) -> list[dict[str, Any]]:
        query = """
        MATCH (p:Paper)
        WHERE p.community_id IS NOT NULL
        RETURN p.community_id AS community_id, count(*) AS size
        ORDER BY size DESC
        LIMIT $limit
        """
        with self._driver.session() as session:
            return [dict(record) for record in session.run(query, limit=limit)]

    def get_publications_per_year(self) -> list[dict[str, Any]]:
        query = """
        MATCH (p:Paper)
        WHERE p.year IS NOT NULL
        RETURN p.year AS year, count(*) AS count
        ORDER BY year ASC
        """
        with self._driver.session() as session:
            return [dict(record) for record in session.run(query)]

    def get_citation_counts(self, limit: int) -> list[int]:
        query = """
        MATCH (p:Paper)
        WHERE p.citation_count IS NOT NULL
        RETURN p.citation_count AS citation_count
        ORDER BY p.citation_count DESC
        LIMIT $limit
        """
        with self._driver.session() as session:
            values: list[int] = []
            for record in session.run(query, limit=limit):
                value = record.get("citation_count")
                if isinstance(value, int):
                    values.append(value)
            return values


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


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    return default


def _recommend_rows(
    *,
    bibcode: str,
    k: int,
    mode: Literal["graph", "embed", "hybrid"],
) -> list[dict[str, Any]]:
    if mode == "graph":
        return recommend_similar_papers_graph(
            seed_bibcode=bibcode,
            k=k,
            candidate_pool=max(200, k * 10),
        )
    if mode == "embed":
        return _embedding_recommender().recommend_similar_papers_embedding(
            seed_bibcode=bibcode,
            k=k,
        )
    return _hybrid_recommender().recommend_similar_papers_hybrid(
        seed_bibcode=bibcode,
        k=k,
        candidate_pool=max(200, k * 10),
    )


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
        citation_count=(
            row.get("citation_count")
            if isinstance(row.get("citation_count"), int)
            else None
        ),
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
    rows = _recommend_rows(bibcode=bibcode, k=k, mode=mode)

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


@app.get("/stats/overview", response_model=GraphStatsOverview)
def stats_overview_endpoint(
    repository: Neo4jRepository = Depends(get_repository),  # noqa: B008
) -> GraphStatsOverview:
    summary_raw = repository.get_graph_stats_summary()
    paper_count = _as_int(summary_raw.get("paper_count"))
    cites_count = _as_int(summary_raw.get("cites_count"))
    avg_degree = (2.0 * float(cites_count) / float(paper_count)) if paper_count > 0 else 0.0
    density = (
        float(cites_count) / float(paper_count * (paper_count - 1))
        if paper_count > 1
        else 0.0
    )

    summary = GraphStatsSummary(
        paper_count=paper_count,
        author_count=_as_int(summary_raw.get("author_count")),
        keyword_count=_as_int(summary_raw.get("keyword_count")),
        venue_count=_as_int(summary_raw.get("venue_count")),
        cites_count=cites_count,
        avg_degree=avg_degree,
        community_count=_as_int(summary_raw.get("community_count")),
        graph_density=density,
    )

    top_papers: list[GraphStatsTopPaper] = []
    for row in repository.get_top_papers_by_pagerank(limit=10):
        bibcode = row.get("bibcode")
        pagerank_raw = row.get("pagerank")
        if not isinstance(bibcode, str) or not isinstance(pagerank_raw, (int, float)):
            continue
        top_papers.append(
            GraphStatsTopPaper(
                bibcode=bibcode,
                title=row.get("title") if isinstance(row.get("title"), str) else None,
                pagerank=float(pagerank_raw),
                citation_count=(
                    row.get("citation_count")
                    if isinstance(row.get("citation_count"), int)
                    else None
                ),
                year=row.get("year") if isinstance(row.get("year"), int) else None,
            )
        )

    community_sizes: list[GraphStatsCommunity] = []
    for row in repository.get_community_sizes(limit=20):
        community_id = row.get("community_id")
        size = row.get("size")
        if isinstance(community_id, int) and isinstance(size, int):
            community_sizes.append(
                GraphStatsCommunity(
                    community_id=community_id,
                    size=size,
                )
            )

    publications_per_year: list[GraphStatsYearCount] = []
    for row in repository.get_publications_per_year():
        year = row.get("year")
        count = row.get("count")
        if isinstance(year, int) and isinstance(count, int):
            publications_per_year.append(GraphStatsYearCount(year=year, count=count))

    citation_counts = repository.get_citation_counts(limit=5000)

    return GraphStatsOverview(
        summary=summary,
        top_papers=top_papers,
        community_sizes=community_sizes,
        publications_per_year=publications_per_year,
        citation_counts=citation_counts,
    )


@app.get("/subgraph/paper/{bibcode}", response_model=SubgraphResponse)
def subgraph_endpoint(
    bibcode: str,
    k: int = Query(10, ge=1, le=15),
    mode: Literal["graph", "embed", "hybrid"] = Query("hybrid"),
    include_citations: bool = Query(False),
    include_keywords: bool = Query(False),
    include_similarity: bool = Query(False),
    repository: Neo4jRepository = Depends(get_repository),  # noqa: B008
) -> SubgraphResponse:
    seed = repository.get_paper(bibcode)
    if seed is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    rec_rows = _recommend_rows(bibcode=bibcode, k=k, mode=mode)[:k]

    nodes: dict[str, SubgraphNode] = {}
    edges: list[SubgraphEdge] = []
    edge_keys: set[tuple[str, str, str]] = set()

    seed_title = seed.get("title") if isinstance(seed.get("title"), str) else bibcode
    seed_year = seed.get("year") if isinstance(seed.get("year"), int) else None
    nodes[bibcode] = SubgraphNode(
        id=bibcode,
        label=seed_title or bibcode,
        type="seed",
        year=seed_year,
        title=seed_title,
    )

    visible_bibcodes = [bibcode]
    for row in rec_rows:
        rec_bibcode = row.get("bibcode")
        if not isinstance(rec_bibcode, str):
            continue
        visible_bibcodes.append(rec_bibcode)

        title = row.get("title") if isinstance(row.get("title"), str) else rec_bibcode
        year = row.get("year") if isinstance(row.get("year"), int) else None
        nodes[rec_bibcode] = SubgraphNode(
            id=rec_bibcode,
            label=title or rec_bibcode,
            type="recommended",
            year=year,
            title=title,
        )

        reasons_raw = row.get("reasons")
        if isinstance(reasons_raw, list):
            reasons = [reason for reason in reasons_raw if isinstance(reason, str)]
        else:
            reasons = []
        reason_text = reasons[0] if reasons else "Recommended"
        edge_key = (bibcode, rec_bibcode, "RECOMMENDS")
        if edge_key not in edge_keys:
            edge_keys.add(edge_key)
            edges.append(
                SubgraphEdge(
                    source=bibcode,
                    target=rec_bibcode,
                    label=reason_text,
                    type="RECOMMENDS",
                )
            )

    deduped_bibcodes = sorted(set(visible_bibcodes))

    if include_citations:
        citation_rows = repository.get_citation_edges(deduped_bibcodes, limit=60)
        for row in citation_rows:
            source = row.get("source")
            target = row.get("target")
            if not isinstance(source, str) or not isinstance(target, str):
                continue
            if source not in nodes or target not in nodes:
                continue
            edge_key = (source, target, "CITES")
            if edge_key in edge_keys:
                continue
            edge_keys.add(edge_key)
            edges.append(
                SubgraphEdge(
                    source=source,
                    target=target,
                    label="CITES",
                    type="CITES",
                )
            )

    if include_keywords:
        keyword_rows = repository.get_keyword_links(deduped_bibcodes, limit=120)
        per_paper_count: dict[str, int] = {}
        keyword_nodes_added = 0
        for row in keyword_rows:
            paper_id = row.get("bibcode")
            keyword = row.get("keyword")
            if not isinstance(paper_id, str) or not isinstance(keyword, str):
                continue
            if paper_id not in nodes:
                continue

            count = per_paper_count.get(paper_id, 0)
            if count >= 2:
                continue
            if keyword_nodes_added >= 30:
                break

            keyword_id = f"kw::{keyword}"
            if keyword_id not in nodes:
                nodes[keyword_id] = SubgraphNode(
                    id=keyword_id,
                    label=keyword,
                    type="keyword",
                    title=keyword,
                )
                keyword_nodes_added += 1

            edge_key = (paper_id, keyword_id, "HAS_KEYWORD")
            if edge_key not in edge_keys:
                edge_keys.add(edge_key)
                edges.append(
                    SubgraphEdge(
                        source=paper_id,
                        target=keyword_id,
                        label="HAS_KEYWORD",
                        type="HAS_KEYWORD",
                    )
                )
                per_paper_count[paper_id] = count + 1

    if include_similarity:
        similarity_rows = repository.get_similarity_edges(deduped_bibcodes, limit=80)
        for row in similarity_rows:
            source = row.get("source")
            target = row.get("target")
            if not isinstance(source, str) or not isinstance(target, str):
                continue
            if source not in nodes or target not in nodes:
                continue

            similarity_raw = row.get("similarity")
            similarity = float(similarity_raw) if isinstance(similarity_raw, (int, float)) else 0.0
            edge_key = (source, target, "SIMILAR_TO")
            if edge_key in edge_keys:
                continue
            edge_keys.add(edge_key)
            edges.append(
                SubgraphEdge(
                    source=source,
                    target=target,
                    label=f"SIMILAR ({similarity:.2f})",
                    type="SIMILAR_TO",
                    similarity=similarity,
                )
            )

    return SubgraphResponse(nodes=list(nodes.values()), edges=edges)
