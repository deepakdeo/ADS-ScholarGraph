"""Run graph analytics and write features back to Neo4j."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from neo4j import Driver, Session
else:
    Driver = Any
    Session = Any

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import Neo4jError as _Neo4jDriverError
    Neo4jDriverError = _Neo4jDriverError
except ModuleNotFoundError:
    GraphDatabase = None
    Neo4jDriverError = Exception

from ads_scholargraph.config import get_settings

AnalyticsMode = Literal["auto", "gds", "networkx"]

PAGERANK_WRITE_QUERY = """
UNWIND $rows AS row
MATCH (p:Paper {bibcode: row.bibcode})
SET p.pagerank = row.score
"""

COMMUNITY_WRITE_QUERY = """
UNWIND $rows AS row
MATCH (p:Paper {bibcode: row.bibcode})
SET p.community_id = row.community_id
"""

BETWEENNESS_WRITE_QUERY = """
UNWIND $rows AS row
MATCH (a:Author {name: row.author_name})
SET a.betweenness = row.score
"""


def _batched(rows: list[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    for index in range(0, len(rows), batch_size):
        yield rows[index : index + batch_size]


def _write_rows(
    session: Session,
    query: str,
    rows: list[dict[str, Any]],
    batch_size: int,
) -> int:
    if not rows:
        return 0

    for batch in _batched(rows, batch_size):
        session.run(query, rows=batch).consume()
    return len(rows)


def _is_gds_available(session: Session) -> bool:
    try:
        record = session.run("CALL gds.version() YIELD version RETURN version LIMIT 1").single()
    except Neo4jDriverError:
        return False
    return record is not None


def _gds_drop_graph_if_exists(session: Session, graph_name: str) -> None:
    record = session.run(
        "CALL gds.graph.exists($graph_name) YIELD exists RETURN exists",
        graph_name=graph_name,
    ).single()
    exists = bool(record and record.get("exists"))
    if exists:
        session.run("CALL gds.graph.drop($graph_name)", graph_name=graph_name).consume()


def _collect_records(result: Any, key_map: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in result:
        row: dict[str, Any] = {}
        valid = True
        for output_key, record_key in key_map.items():
            value = record.get(record_key)
            if value is None:
                valid = False
                break
            row[output_key] = value
        if valid:
            rows.append(row)
    return rows


def _compute_gds_metrics(session: Session) -> dict[str, list[dict[str, Any]]]:
    citation_graph = "citation_graph_analytics"
    author_graph = "author_collab_graph_analytics"

    _gds_drop_graph_if_exists(session, citation_graph)
    _gds_drop_graph_if_exists(session, author_graph)

    try:
        session.run(
            """
            CALL gds.graph.project(
                $graph_name,
                'Paper',
                {CITES: {orientation: 'NATURAL'}}
            )
            """,
            graph_name=citation_graph,
        ).consume()

        pagerank_rows = _collect_records(
            session.run(
                """
                CALL gds.pageRank.stream($graph_name)
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).bibcode AS bibcode, score
                """,
                graph_name=citation_graph,
            ),
            {"bibcode": "bibcode", "score": "score"},
        )

        try:
            community_result = session.run(
                """
                CALL gds.louvain.stream($graph_name)
                YIELD nodeId, communityId
                RETURN gds.util.asNode(nodeId).bibcode AS bibcode,
                       toInteger(communityId) AS community_id
                """,
                graph_name=citation_graph,
            )
        except Neo4jDriverError:
            community_result = session.run(
                """
                CALL gds.leiden.stream($graph_name)
                YIELD nodeId, communityId
                RETURN gds.util.asNode(nodeId).bibcode AS bibcode,
                       toInteger(communityId) AS community_id
                """,
                graph_name=citation_graph,
            )

        community_rows = _collect_records(
            community_result,
            {"bibcode": "bibcode", "community_id": "community_id"},
        )

        session.run(
            """
            CALL gds.graph.project.cypher(
                $graph_name,
                'MATCH (a:Author) RETURN id(a) AS id',
                '
                MATCH (a1:Author)-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
                WHERE id(a1) < id(a2)
                WITH id(a1) AS src, id(a2) AS dst, count(DISTINCT p) AS weight
                RETURN src AS source, dst AS target, weight
                UNION
                MATCH (a1:Author)-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
                WHERE id(a1) < id(a2)
                WITH id(a1) AS src, id(a2) AS dst, count(DISTINCT p) AS weight
                RETURN dst AS source, src AS target, weight
                '
            )
            """,
            graph_name=author_graph,
        ).consume()

        betweenness_rows = _collect_records(
            session.run(
                """
                CALL gds.betweenness.stream($graph_name)
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).name AS author_name, score
                """,
                graph_name=author_graph,
            ),
            {"author_name": "author_name", "score": "score"},
        )
    finally:
        _gds_drop_graph_if_exists(session, citation_graph)
        _gds_drop_graph_if_exists(session, author_graph)

    return {
        "pagerank": pagerank_rows,
        "communities": community_rows,
        "betweenness": betweenness_rows,
    }


def _compute_networkx_metrics(session: Session) -> dict[str, list[dict[str, Any]]]:
    try:
        import networkx as nx
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "NetworkX is required for fallback analytics mode. Install dependency 'networkx'."
        ) from exc

    paper_nodes = [
        record.get("bibcode")
        for record in session.run("MATCH (p:Paper) RETURN p.bibcode AS bibcode")
        if isinstance(record.get("bibcode"), str)
    ]

    citation_graph = nx.DiGraph()
    citation_graph.add_nodes_from(paper_nodes)

    for record in session.run(
        """
        MATCH (src:Paper)-[:CITES]->(dst:Paper)
        RETURN src.bibcode AS source_bibcode, dst.bibcode AS target_bibcode
        """
    ):
        source = record.get("source_bibcode")
        target = record.get("target_bibcode")
        if isinstance(source, str) and isinstance(target, str):
            citation_graph.add_edge(source, target)

    pagerank_scores = nx.pagerank(citation_graph)
    pagerank_rows = [
        {"bibcode": bibcode, "score": float(score)}
        for bibcode, score in pagerank_scores.items()
    ]

    undirected_citation = citation_graph.to_undirected()
    if undirected_citation.number_of_nodes() == 0:
        communities: list[set[str]] = []
    elif hasattr(nx.community, "louvain_communities"):
        communities = [
            {str(node) for node in community}
            for community in nx.community.louvain_communities(undirected_citation, seed=42)
        ]
    else:
        communities = [
            {str(node) for node in community}
            for community in nx.community.greedy_modularity_communities(undirected_citation)
        ]

    communities_sorted = sorted(
        communities,
        key=lambda community: (-len(community), sorted(community)[0] if community else ""),
    )

    community_rows: list[dict[str, Any]] = []
    for community_id, members in enumerate(communities_sorted):
        for bibcode in members:
            community_rows.append({"bibcode": bibcode, "community_id": community_id})

    author_nodes = [
        record.get("author_name")
        for record in session.run("MATCH (a:Author) RETURN a.name AS author_name")
        if isinstance(record.get("author_name"), str)
    ]

    author_graph = nx.Graph()
    author_graph.add_nodes_from(author_nodes)

    for record in session.run(
        """
        MATCH (a1:Author)-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
        WHERE a1.name < a2.name
        RETURN a1.name AS author_1, a2.name AS author_2, count(DISTINCT p) AS weight
        """
    ):
        author_1 = record.get("author_1")
        author_2 = record.get("author_2")
        weight = record.get("weight")
        if isinstance(author_1, str) and isinstance(author_2, str):
            edge_weight = int(weight) if isinstance(weight, int) else 1
            author_graph.add_edge(author_1, author_2, weight=edge_weight)

    betweenness_scores = nx.betweenness_centrality(author_graph, weight="weight", normalized=True)
    betweenness_rows = [
        {"author_name": author_name, "score": float(score)}
        for author_name, score in betweenness_scores.items()
    ]

    return {
        "pagerank": pagerank_rows,
        "communities": community_rows,
        "betweenness": betweenness_rows,
    }


def run_analytics(
    *,
    mode: AnalyticsMode,
    batch_size: int = 1000,
    driver: Driver | None = None,
) -> dict[str, Any]:
    """Run graph analytics using GDS or NetworkX fallback and write node properties."""

    settings = get_settings()
    managed_driver = driver is None
    if driver is None:
        if GraphDatabase is None:
            raise RuntimeError(
                "neo4j package is not installed. Install dependencies to run analytics."
            )
        resolved_driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
    else:
        resolved_driver = driver

    try:
        with resolved_driver.session() as session:
            implementation: str

            if mode == "gds":
                if not _is_gds_available(session):
                    raise RuntimeError("GDS is not available in this Neo4j instance.")
                metrics = _compute_gds_metrics(session)
                implementation = "gds"
            elif mode == "networkx":
                metrics = _compute_networkx_metrics(session)
                implementation = "networkx"
            else:
                if _is_gds_available(session):
                    try:
                        metrics = _compute_gds_metrics(session)
                        implementation = "gds"
                    except Exception:
                        metrics = _compute_networkx_metrics(session)
                        implementation = "networkx"
                else:
                    metrics = _compute_networkx_metrics(session)
                    implementation = "networkx"

            pagerank_count = _write_rows(
                session,
                PAGERANK_WRITE_QUERY,
                metrics["pagerank"],
                batch_size,
            )
            community_count = _write_rows(
                session,
                COMMUNITY_WRITE_QUERY,
                metrics["communities"],
                batch_size,
            )
            betweenness_count = _write_rows(
                session,
                BETWEENNESS_WRITE_QUERY,
                metrics["betweenness"],
                batch_size,
            )
    finally:
        if managed_driver:
            resolved_driver.close()

    return {
        "mode": mode,
        "implementation": implementation,
        "counts": {
            "pagerank": pagerank_count,
            "communities": community_count,
            "betweenness": betweenness_count,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Build parser for graph analytics CLI."""

    parser = argparse.ArgumentParser(
        description="Compute graph analytics and write Neo4j properties."
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "gds", "networkx"],
        default="auto",
        help="Execution mode: prefer GDS, force GDS, or force NetworkX.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Rows per property write batch",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")

    result = run_analytics(mode=args.mode, batch_size=args.batch_size)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
