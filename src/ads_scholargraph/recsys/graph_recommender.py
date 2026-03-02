"""Graph-based paper recommendations from Neo4j citation structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from ads_scholargraph.config import get_settings

if TYPE_CHECKING:
    from neo4j import Driver
else:
    Driver = Any

_graph_database: Any | None
try:
    from neo4j import GraphDatabase as _neo4j_graph_database
    _graph_database = _neo4j_graph_database
except ModuleNotFoundError:
    _graph_database = None


@dataclass
class GraphCandidate:
    bibcode: str
    title: str | None
    year: int | None
    pagerank: float
    overlap_count: int
    shared_references: int
    co_citation_overlap: int
    same_community: bool
    community_id: int | None


class GraphQueryClient(Protocol):
    """Protocol for graph query execution."""

    def run_query(self, query: str, **params: Any) -> list[dict[str, Any]]:
        """Return query rows as dictionaries."""


class Neo4jGraphClient:
    """Thin Neo4j client wrapper for graph recommender queries."""

    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    @classmethod
    def from_settings(cls) -> Neo4jGraphClient:
        settings = get_settings()
        if _graph_database is None:
            raise RuntimeError(
                "neo4j package is not installed. Install dependencies to query Neo4j."
            )
        driver = _graph_database.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        return cls(driver)

    def run_query(self, query: str, **params: Any) -> list[dict[str, Any]]:
        with self._driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]


CANDIDATE_QUERY = """
MATCH (seed:Paper {bibcode: $seed_bibcode, is_seed: true})
OPTIONAL MATCH (seed)-[:CITES]->(seed_ref:Paper)
WITH seed, collect(DISTINCT seed_ref.bibcode) AS seed_refs
OPTIONAL MATCH (seed_citer:Paper)-[:CITES]->(seed)
WITH seed, seed_refs, collect(DISTINCT seed_citer.bibcode) AS seed_citers
MATCH (candidate:Paper {is_seed: true})
WHERE candidate.bibcode <> seed.bibcode
OPTIONAL MATCH (candidate)-[:CITES]->(cand_ref:Paper)
WITH seed, seed_refs, seed_citers, candidate, collect(DISTINCT cand_ref.bibcode) AS cand_refs
OPTIONAL MATCH (cand_citer:Paper)-[:CITES]->(candidate)
WITH seed, candidate, seed_refs, seed_citers, cand_refs,
     collect(DISTINCT cand_citer.bibcode) AS cand_citers
WITH seed, candidate,
     size([x IN cand_refs WHERE x IN seed_refs]) AS shared_references,
     size([x IN cand_citers WHERE x IN seed_citers]) AS co_citation_overlap,
     CASE WHEN seed.community_id IS NOT NULL AND candidate.community_id = seed.community_id
          THEN true ELSE false END AS same_community
WITH candidate, shared_references, co_citation_overlap, same_community,
     (shared_references + co_citation_overlap +
      CASE WHEN same_community THEN 1 ELSE 0 END) AS overlap_count
WHERE overlap_count > 0
RETURN candidate.bibcode AS bibcode,
       candidate.title AS title,
       candidate.year AS year,
       coalesce(candidate.pagerank, 0.0) AS pagerank,
       overlap_count,
       shared_references,
       co_citation_overlap,
       same_community,
       candidate.community_id AS community_id
ORDER BY overlap_count DESC, pagerank DESC
LIMIT $candidate_pool
"""


def _safe_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _safe_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _normalize(value: float, max_value: float) -> float:
    if max_value <= 0.0:
        return 0.0
    return value / max_value


def _score_candidates(candidates: list[GraphCandidate]) -> list[dict[str, Any]]:
    max_overlap = max((candidate.overlap_count for candidate in candidates), default=0)
    max_pagerank = max((candidate.pagerank for candidate in candidates), default=0.0)

    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        overlap_score = _normalize(float(candidate.overlap_count), float(max_overlap))
        pagerank_score = _normalize(candidate.pagerank, max_pagerank)
        final_score = 0.8 * overlap_score + 0.2 * pagerank_score

        reasons: list[str] = []
        if candidate.same_community and candidate.community_id is not None:
            reasons.append("Same research cluster")
        if candidate.shared_references > 0:
            reasons.append(f"Shares {candidate.shared_references} references")
        if candidate.co_citation_overlap > 0:
            reasons.append(f"Co-cited by {candidate.co_citation_overlap} papers")

        ranked.append(
            {
                "bibcode": candidate.bibcode,
                "title": candidate.title,
                "year": candidate.year,
                "score": round(final_score, 6),
                "reasons": reasons,
                "graph_score": final_score,
                "pagerank": candidate.pagerank,
            }
        )

    ranked.sort(key=lambda row: float(row["score"]), reverse=True)
    return ranked


def recommend_similar_papers_graph(
    seed_bibcode: str,
    k: int = 10,
    candidate_pool: int = 200,
    client: GraphQueryClient | None = None,
) -> list[dict[str, Any]]:
    """Recommend similar seed papers from graph overlap and pagerank signals."""

    if k <= 0:
        return []
    if candidate_pool <= 0:
        return []

    resolved_client: GraphQueryClient
    if client is None:
        resolved_client = Neo4jGraphClient.from_settings()
    else:
        resolved_client = client

    rows = resolved_client.run_query(
        CANDIDATE_QUERY,
        seed_bibcode=seed_bibcode,
        candidate_pool=candidate_pool,
    )

    candidates: list[GraphCandidate] = []
    for row in rows:
        bibcode = row.get("bibcode")
        if not isinstance(bibcode, str):
            continue

        candidates.append(
            GraphCandidate(
                bibcode=bibcode,
                title=row.get("title") if isinstance(row.get("title"), str) else None,
                year=_safe_int(row.get("year")),
                pagerank=_safe_float(row.get("pagerank")),
                overlap_count=int(_safe_float(row.get("overlap_count"))),
                shared_references=int(_safe_float(row.get("shared_references"))),
                co_citation_overlap=int(_safe_float(row.get("co_citation_overlap"))),
                same_community=bool(row.get("same_community")),
                community_id=_safe_int(row.get("community_id")),
            )
        )

    ranked = _score_candidates(candidates)
    return ranked[:k]
