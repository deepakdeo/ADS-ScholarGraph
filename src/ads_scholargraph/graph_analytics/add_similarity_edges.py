"""Materialize bounded TF-IDF similarity edges between papers in Neo4j."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, TypeVar

from ads_scholargraph.config import get_settings

if TYPE_CHECKING:
    from neo4j import Session
else:
    Session = Any


class DriverLike(Protocol):
    """Minimal driver interface required by similarity edge pipeline."""

    def session(self) -> Session: ...

    def close(self) -> None: ...

_graph_database: Any | None
try:
    from neo4j import GraphDatabase as _neo4j_graph_database

    _graph_database = _neo4j_graph_database
except ModuleNotFoundError:
    _graph_database = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError:
    TfidfVectorizer = None
    cosine_similarity = None

FETCH_PAPERS_QUERY = """
MATCH (p:Paper)
WHERE ($seed_only = false OR coalesce(p.is_seed, false) = true)
RETURN p.bibcode AS bibcode,
       p.title AS title,
       p.abstract AS abstract
ORDER BY p.bibcode
LIMIT $limit
"""

DELETE_SIMILARITY_QUERY = "MATCH ()-[r:SIMILAR_TO]-() DELETE r"

UPSERT_SIMILARITY_QUERY = """
UNWIND $rows AS row
MATCH (a:Paper {bibcode: row.bib_a})
MATCH (b:Paper {bibcode: row.bib_b})
MERGE (a)-[r:SIMILAR_TO]-(b)
SET r.similarity = row.similarity
"""


class SimilarityEdge(TypedDict):
    """Row payload for SIMILAR_TO relationship writes."""

    bib_a: str
    bib_b: str
    similarity: float


_RowT = TypeVar("_RowT")


def _batched(rows: list[_RowT], batch_size: int) -> Iterator[list[_RowT]]:
    for index in range(0, len(rows), batch_size):
        yield rows[index : index + batch_size]


def _fetch_paper_rows(
    session: Session,
    *,
    seed_only: bool,
    limit: int,
) -> list[dict[str, Any]]:
    return [
        dict(record)
        for record in session.run(
            FETCH_PAPERS_QUERY,
            seed_only=seed_only,
            limit=limit,
        )
    ]


def _build_documents(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    bibcodes: list[str] = []
    docs: list[str] = []

    for row in rows:
        bibcode = row.get("bibcode")
        if not isinstance(bibcode, str):
            continue

        title = row.get("title") if isinstance(row.get("title"), str) else ""
        abstract = row.get("abstract") if isinstance(row.get("abstract"), str) else ""
        doc = f"{title} {abstract}".strip()
        if not doc:
            continue

        bibcodes.append(bibcode)
        docs.append(doc)

    return bibcodes, docs


def _compute_similarity_edges(
    *,
    bibcodes: list[str],
    docs: list[str],
    threshold: float,
    top_k: int,
    max_features: int,
) -> list[SimilarityEdge]:
    if len(bibcodes) != len(docs):
        raise ValueError("bibcodes and docs must have the same length")
    if len(bibcodes) < 2:
        return []
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("threshold must be in (0.0, 1.0)")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if max_features <= 0:
        raise ValueError("max_features must be greater than 0")
    if TfidfVectorizer is None or cosine_similarity is None:
        raise RuntimeError("scikit-learn is required to compute similarity edges.")

    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    matrix = vectorizer.fit_transform(docs)
    similarities = cosine_similarity(matrix, dense_output=False)

    edge_scores: dict[tuple[str, str], float] = {}
    for i, bib_a in enumerate(bibcodes):
        row = similarities.getrow(i)
        candidates: list[tuple[int, float]] = []
        for idx, score in zip(row.indices, row.data, strict=False):
            if int(idx) == i:
                continue
            score_value = float(score)
            if score_value < threshold:
                continue
            candidates.append((int(idx), score_value))

        candidates.sort(key=lambda item: item[1], reverse=True)
        for idx, score_value in candidates[:top_k]:
            bib_b = bibcodes[idx]
            edge_key = (bib_a, bib_b) if bib_a < bib_b else (bib_b, bib_a)
            previous_score = edge_scores.get(edge_key)
            if previous_score is None or score_value > previous_score:
                edge_scores[edge_key] = score_value

    rows: list[SimilarityEdge] = [
        SimilarityEdge(bib_a=bib_a, bib_b=bib_b, similarity=round(score, 6))
        for (bib_a, bib_b), score in edge_scores.items()
    ]
    rows.sort(key=lambda row: row["similarity"], reverse=True)
    return rows


def _write_similarity_edges(
    session: Session,
    *,
    rows: list[SimilarityEdge],
    batch_size: int,
    wipe_existing: bool,
) -> None:
    if wipe_existing:
        session.run(DELETE_SIMILARITY_QUERY).consume()

    for batch in _batched(rows, batch_size):
        session.run(UPSERT_SIMILARITY_QUERY, rows=batch).consume()


def add_similarity_edges(
    *,
    threshold: float = 0.3,
    top_k: int = 5,
    seed_only: bool = True,
    limit: int = 5000,
    max_features: int = 50000,
    wipe_existing: bool = False,
    batch_size: int = 1000,
    driver: DriverLike | None = None,
) -> dict[str, Any]:
    """Compute and write bounded `SIMILAR_TO` edges to Neo4j."""

    if limit <= 1:
        raise ValueError("limit must be greater than 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    settings = get_settings()
    managed_driver = driver is None
    if driver is None:
        if _graph_database is None:
            raise RuntimeError(
                "neo4j package is not installed. Install dependencies to add similarity edges."
            )
        resolved_driver = _graph_database.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
    else:
        resolved_driver = driver

    try:
        with resolved_driver.session() as session:
            raw_rows = _fetch_paper_rows(session, seed_only=seed_only, limit=limit)
            bibcodes, docs = _build_documents(raw_rows)
            edge_rows = _compute_similarity_edges(
                bibcodes=bibcodes,
                docs=docs,
                threshold=threshold,
                top_k=top_k,
                max_features=max_features,
            )
            _write_similarity_edges(
                session,
                rows=edge_rows,
                batch_size=batch_size,
                wipe_existing=wipe_existing,
            )
    finally:
        if managed_driver:
            resolved_driver.close()

    return {
        "papers_considered": len(bibcodes),
        "similarity_edges": len(edge_rows),
        "threshold": threshold,
        "top_k": top_k,
        "seed_only": seed_only,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for similarity edge materialization."""

    parser = argparse.ArgumentParser(
        description="Add bounded TF-IDF SIMILAR_TO edges to Neo4j papers."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Similarity threshold in (0,1)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Max similar neighbors kept per paper",
    )
    parser.add_argument(
        "--seed-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only seed papers (default true).",
    )
    parser.add_argument("--limit", type=int, default=5000, help="Max papers to consider")
    parser.add_argument("--max-features", type=int, default=50000, help="TF-IDF vocabulary cap")
    parser.add_argument("--batch-size", type=int, default=1000, help="Write batch size")
    parser.add_argument(
        "--wipe-existing",
        action="store_true",
        help="Delete existing SIMILAR_TO edges before writing new ones.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for similarity edge materialization."""

    parser = build_parser()
    args = parser.parse_args(argv)

    result = add_similarity_edges(
        threshold=args.threshold,
        top_k=args.top_k,
        seed_only=args.seed_only,
        limit=args.limit,
        max_features=args.max_features,
        wipe_existing=args.wipe_existing,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
