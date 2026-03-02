"""Streamlit dashboard for seed search and explainable recommendations."""

from __future__ import annotations

import json
import os
from typing import Any

import networkx as nx
import pandas as pd
import requests
import streamlit as st
from pyvis.network import Network
from streamlit.components.v1 import html as components_html

from ads_scholargraph.ui.rendering import build_tooltip, sanitize_ads_html, shorten_title
from ads_scholargraph.utils.schema_diagram import SchemaStats, render_schema_svg

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MAX_GRAPH_RECS = 15
GRAPH_HEIGHT_PX = 680


def _api_get(path: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _paper_label(paper: dict[str, Any]) -> str:
    title = paper.get("title") or "(untitled)"
    year = paper.get("year") if isinstance(paper.get("year"), int) else ""
    bibcode = paper.get("bibcode") or ""
    return f"{title} ({year}) [{bibcode}]"


def _render_abstract(abstract: str | None) -> None:
    if not abstract:
        st.markdown("No abstract available.")
        return
    sanitized = sanitize_ads_html(abstract)
    st.markdown(sanitized, unsafe_allow_html=True)


def _node_display_label(
    node: dict[str, Any],
    *,
    show_full_titles: bool,
    max_label_len: int,
) -> str:
    raw_title = node.get("title") if isinstance(node.get("title"), str) else ""
    fallback = node.get("label") if isinstance(node.get("label"), str) else ""
    title = raw_title or fallback or "(untitled)"
    year = node.get("year") if isinstance(node.get("year"), int) else None

    if show_full_titles:
        short_title = " ".join(title.split())
    else:
        short_title = shorten_title(title, max_label_len)

    return f"{year} — {short_title}" if year is not None else short_title


def _build_graph_details(
    *,
    seed_detail: dict[str, Any],
    recs: list[dict[str, Any]],
    graph_nodes: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}

    seed_bibcode = seed_detail.get("bibcode")
    if isinstance(seed_bibcode, str):
        details[seed_bibcode] = dict(seed_detail)

    for rec in recs:
        bibcode = rec.get("bibcode")
        if isinstance(bibcode, str):
            details[bibcode] = dict(rec)

    for node in graph_nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue
        detail = details.get(node_id, {})
        if not isinstance(detail.get("title"), str) and isinstance(node.get("title"), str):
            detail["title"] = node.get("title")
        if not isinstance(detail.get("year"), int) and isinstance(node.get("year"), int):
            detail["year"] = node.get("year")
        detail["type"] = node.get("type")
        details[node_id] = detail

    return details


def _enrich_paper_details(details_by_id: dict[str, dict[str, Any]], bibcodes: list[str]) -> None:
    for bibcode in bibcodes:
        detail = details_by_id.get(bibcode, {})
        has_abstract = isinstance(detail.get("abstract"), str)
        has_pagerank = isinstance(detail.get("pagerank"), (int, float))
        if has_abstract and has_pagerank:
            continue
        try:
            payload = _api_get(f"/paper/{bibcode}")
        except requests.RequestException:
            continue
        if not isinstance(payload, dict):
            continue
        merged = dict(detail)
        merged.update(payload)
        details_by_id[bibcode] = merged


def _safe_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _filter_subgraph(
    subgraph: dict[str, Any],
    *,
    details_by_id: dict[str, dict[str, Any]],
    show_papers: bool,
    show_keywords: bool,
    year_range: tuple[int, int],
    min_citations: int,
    min_pagerank: float,
    communities: set[int],
) -> dict[str, list[dict[str, Any]]]:
    nodes_raw = subgraph.get("nodes")
    edges_raw = subgraph.get("edges")
    nodes = nodes_raw if isinstance(nodes_raw, list) else []
    edges = edges_raw if isinstance(edges_raw, list) else []

    keep_ids: set[str] = set()
    kept_nodes: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue

        node_type = node.get("type")
        if node_type in {"seed", "recommended"}:
            if not show_papers:
                continue

            detail = details_by_id.get(node_id, {})
            year = _safe_int(detail.get("year")) or _safe_int(node.get("year"))
            if year is not None and (year < year_range[0] or year > year_range[1]):
                if node_type != "seed":
                    continue

            citations = _safe_int(detail.get("citation_count"))
            if citations is not None and citations < min_citations and node_type != "seed":
                continue

            pagerank = _safe_float(detail.get("pagerank"))
            if pagerank is not None and pagerank < min_pagerank and node_type != "seed":
                continue

            community_id = _safe_int(detail.get("community_id"))
            if communities and community_id not in communities and node_type != "seed":
                continue

        elif node_type == "keyword":
            if not show_keywords:
                continue

        keep_ids.add(node_id)
        kept_nodes.append(node)

    kept_edges: list[dict[str, Any]] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        target = edge.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        if source in keep_ids and target in keep_ids:
            kept_edges.append(edge)

    return {"nodes": kept_nodes, "edges": kept_edges}


def _matching_node_ids(
    nodes: list[dict[str, Any]],
    details_by_id: dict[str, dict[str, Any]],
    search_term: str,
) -> set[str]:
    query = search_term.strip().lower()
    if not query:
        return set()

    matches: set[str] = set()
    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue

        haystack: list[str] = []
        for field in ("label", "title"):
            value = node.get(field)
            if isinstance(value, str):
                haystack.append(value.lower())

        detail = details_by_id.get(node_id, {})
        for field in ("title", "abstract", "bibcode"):
            value = detail.get(field)
            if isinstance(value, str):
                haystack.append(value.lower())

        reasons = detail.get("reasons")
        if isinstance(reasons, list):
            haystack.extend(reason.lower() for reason in reasons if isinstance(reason, str))

        if query in node_id.lower() or any(query in text for text in haystack):
            matches.add(node_id)

    return matches


def _compute_shortest_path_edges(
    subgraph: dict[str, list[dict[str, Any]]],
    source_id: str,
    target_id: str,
) -> tuple[list[str], set[tuple[str, str]]]:
    graph = nx.Graph()
    for edge in subgraph.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        if isinstance(source, str) and isinstance(target, str):
            graph.add_edge(source, target)

    path_nodes = nx.shortest_path(graph, source=source_id, target=target_id)
    highlighted_edges: set[tuple[str, str]] = set()
    for index in range(len(path_nodes) - 1):
        a = path_nodes[index]
        b = path_nodes[index + 1]
        highlighted_edges.add((a, b))
        highlighted_edges.add((b, a))
    return path_nodes, highlighted_edges


def _build_citation_histogram(citation_counts: list[int]) -> pd.DataFrame:
    if not citation_counts:
        return pd.DataFrame(columns=["bucket", "count"])

    max_value = max(citation_counts)
    if max_value <= 10:
        bins = [0, 1, 2, 3, 4, 5, 10]
    else:
        step = max(10, int(max_value / 8))
        bins = list(range(0, max_value + step, step))
    if bins[-1] <= max_value:
        bins.append(max_value + 1)

    series = pd.Series(citation_counts, name="citation_count")
    buckets = pd.cut(series, bins=bins, right=False, include_lowest=True)
    counts = buckets.value_counts().sort_index()
    frame = counts.rename("count").reset_index()
    frame.columns = ["bucket", "count"]
    frame["bucket"] = frame["bucket"].astype(str)
    return frame


def _render_graph_stats_tab(stats_overview: dict[str, Any]) -> None:
    summary = stats_overview.get("summary")
    if not isinstance(summary, dict):
        st.warning("Stats payload is missing summary data.")
        return

    st.subheader("Knowledge Graph Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Papers", summary.get("paper_count", 0))
    col2.metric("Authors", summary.get("author_count", 0))
    col3.metric("Keywords", summary.get("keyword_count", 0))
    col4.metric("Venues", summary.get("venue_count", 0))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Citation Edges", summary.get("cites_count", 0))
    col6.metric("Avg Degree", f"{float(summary.get('avg_degree', 0.0)):.2f}")
    col7.metric("Communities", summary.get("community_count", 0))
    col8.metric("Graph Density", f"{float(summary.get('graph_density', 0.0)):.6f}")

    top_papers = stats_overview.get("top_papers")
    if isinstance(top_papers, list) and top_papers:
        top_df = pd.DataFrame(top_papers)
        chart_df = top_df[["bibcode", "pagerank"]].set_index("bibcode")
        st.markdown("**Top Papers by PageRank**")
        st.bar_chart(chart_df, use_container_width=True)

    communities = stats_overview.get("community_sizes")
    if isinstance(communities, list) and communities:
        community_df = pd.DataFrame(communities)
        chart_df = community_df[["community_id", "size"]].set_index("community_id")
        st.markdown("**Community Size Distribution**")
        st.bar_chart(chart_df, use_container_width=True)

    pubs_per_year = stats_overview.get("publications_per_year")
    if isinstance(pubs_per_year, list) and pubs_per_year:
        year_df = pd.DataFrame(pubs_per_year).sort_values("year")
        chart_df = year_df[["year", "count"]].set_index("year")
        st.markdown("**Publications Per Year**")
        st.line_chart(chart_df, use_container_width=True)

    citation_counts = stats_overview.get("citation_counts")
    if isinstance(citation_counts, list):
        numeric_citations = [value for value in citation_counts if isinstance(value, int)]
        hist_df = _build_citation_histogram(numeric_citations)
        if not hist_df.empty:
            st.markdown("**Citation Count Distribution**")
            st.bar_chart(hist_df.set_index("bucket"), use_container_width=True)


def _render_subgraph(
    subgraph: dict[str, Any],
    *,
    detail_by_id: dict[str, dict[str, Any]],
    show_full_titles: bool,
    max_label_len: int,
    enable_physics: bool,
    stabilization_iterations: int,
    show_edge_labels: bool,
    highlighted_nodes: set[str] | None = None,
    highlighted_edges: set[tuple[str, str]] | None = None,
) -> None:
    net = Network(height=f"{GRAPH_HEIGHT_PX}px", width="100%", directed=True, bgcolor="#ffffff")
    highlighted_nodes = highlighted_nodes or set()
    highlighted_edges = highlighted_edges or set()

    color_map = {
        "seed": "#f97316",
        "recommended": "#3b82f6",
        "keyword": "#94a3b8",
    }
    shape_map = {
        "seed": "star",
        "recommended": "dot",
        "keyword": "box",
    }
    size_map = {
        "seed": 42,
        "recommended": 24,
        "keyword": 16,
    }
    font_map = {
        "seed": 17,
        "recommended": 14,
        "keyword": 12,
    }

    nodes = subgraph.get("nodes", [])
    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue

        node_type = node.get("type") if isinstance(node.get("type"), str) else "recommended"
        label = _node_display_label(
            node,
            show_full_titles=show_full_titles,
            max_label_len=max_label_len,
        )
        tooltip = build_tooltip(node, detail_by_id.get(node_id))
        is_highlighted = node_id in highlighted_nodes
        border_width = 4 if is_highlighted else 2

        net.add_node(
            node_id,
            label=label,
            title=tooltip,
            color=("#facc15" if is_highlighted else color_map.get(node_type, "#64748b")),
            shape=shape_map.get(node_type, "dot"),
            size=(
                size_map.get(node_type, 22) * 1.5
                if is_highlighted
                else size_map.get(node_type, 22)
            ),
            borderWidth=border_width,
            font={"size": font_map.get(node_type, 14), "face": "Helvetica"},
        )

    edges = subgraph.get("edges", [])
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            continue

        edge_type = edge.get("type") if isinstance(edge.get("type"), str) else "RECOMMENDS"
        label = edge.get("label") if isinstance(edge.get("label"), str) else edge_type
        color = "#64748b"
        width = 2.0
        if edge_type == "CITES":
            color = "#94a3b8"
            width = 1.0
        elif edge_type == "HAS_KEYWORD":
            color = "#9ca3af"
            width = 1.2
        elif edge_type == "RECOMMENDS":
            color = "#f59e0b"
            width = 2.2
        dashes = edge_type == "HAS_KEYWORD"
        if (source, target) in highlighted_edges:
            color = "#ef4444"
            width = 3.2
            dashes = False

        net.add_edge(
            source,
            target,
            label=label if show_edge_labels else "",
            title=label,
            color=color,
            width=width,
            dashes=dashes,
        )

    options = {
        "layout": {"improvedLayout": True},
        "interaction": {
            "hover": True,
            "navigationButtons": True,
            "keyboard": True,
            "tooltipDelay": 80,
            "zoomView": True,
            "dragView": True,
        },
        "physics": {
            "enabled": enable_physics,
            "barnesHut": {
                "gravitationalConstant": -12000,
                "centralGravity": 0.2,
                "springLength": 220,
                "springConstant": 0.02,
                "damping": 0.82,
                "avoidOverlap": 0.35,
            },
            "stabilization": {
                "enabled": enable_physics,
                "iterations": stabilization_iterations,
                "fit": True,
            },
        },
        "edges": {
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.7}},
            "font": {"size": 11, "align": "middle", "face": "Helvetica"},
            "smooth": {"enabled": True, "type": "dynamic"},
        },
    }
    net.set_options(json.dumps(options))
    components_html(net.generate_html(), height=GRAPH_HEIGHT_PX + 40, scrolling=True)


def main() -> None:
    st.set_page_config(page_title="ADS ScholarGraph Recommender", layout="wide")

    st.title("ADS ScholarGraph Recommender")
    st.caption("Search seed papers and generate explainable recommendations from the local KG.")

    st.sidebar.header("Recommendation Settings")
    mode = st.sidebar.selectbox("Mode", options=["hybrid", "graph", "embed"], index=0)
    k = st.sidebar.slider("Top-K", min_value=3, max_value=30, value=10, step=1)
    max_graph_recs = st.sidebar.slider(
        "Max nodes in graph (recommendations)",
        min_value=3,
        max_value=MAX_GRAPH_RECS,
        value=10,
        step=1,
    )
    show_full_titles = st.sidebar.checkbox("Show full titles on graph nodes", value=False)
    max_label_len = st.sidebar.slider(
        "Max label length",
        min_value=30,
        max_value=100,
        value=60,
        step=5,
    )
    enable_physics = st.sidebar.checkbox("Enable physics", value=True)
    stabilization_iterations = st.sidebar.slider(
        "Stabilize iterations",
        min_value=50,
        max_value=500,
        value=180,
        step=10,
        disabled=not enable_physics,
    )
    show_edge_labels = st.sidebar.checkbox("Show edge labels", value=False)
    st.sidebar.markdown("### Graph enrichments")
    include_citations = st.sidebar.checkbox("Include citation edges", value=False)
    include_keywords = st.sidebar.checkbox("Include keyword nodes", value=False)

    query = st.text_input("Search seed papers (title/abstract keywords)", value="")

    matches: list[dict[str, Any]] = []
    if query.strip():
        try:
            matches = _api_get("/search", params={"q": query.strip(), "limit": 20})
        except requests.RequestException as exc:
            st.error(f"Search failed: {exc}")

    if not matches:
        st.info("Enter keywords to search seed papers.")
        return

    paper_by_label = {_paper_label(paper): paper for paper in matches}
    selected_label = st.selectbox("Select a seed paper", options=list(paper_by_label.keys()))
    selected = paper_by_label[selected_label]
    selected_bibcode = selected.get("bibcode")

    if not isinstance(selected_bibcode, str):
        st.error("Selected paper has an invalid bibcode.")
        return

    trigger = st.button("Recommend", type="primary")

    if not trigger:
        return

    graph_k = min(k, max_graph_recs, MAX_GRAPH_RECS)

    try:
        seed_detail = _api_get(f"/paper/{selected_bibcode}")
        recs = _api_get(
            f"/recommend/paper/{selected_bibcode}",
            params={"k": k, "mode": mode},
        )
        graph_data = _api_get(
            f"/subgraph/paper/{selected_bibcode}",
            params={
                "k": graph_k,
                "mode": mode,
                "include_citations": include_citations,
                "include_keywords": include_keywords,
            },
        )
    except requests.RequestException as exc:
        st.error(f"Recommendation request failed: {exc}")
        return

    stats_overview: dict[str, Any] = {}
    stats_error: str | None = None
    try:
        stats_payload = _api_get("/stats/overview")
        if isinstance(stats_payload, dict):
            stats_overview = stats_payload
    except requests.RequestException as exc:
        stats_error = str(exc)

    with st.expander("Seed paper abstract", expanded=False):
        _render_abstract(seed_detail.get("abstract"))

    if not recs:
        st.warning("No recommendations returned for this seed paper.")
        return

    table_rows: list[dict[str, Any]] = []
    for rec in recs:
        table_rows.append(
            {
                "bibcode": rec.get("bibcode"),
                "year": rec.get("year"),
                "title": rec.get("title"),
                "score": rec.get("score"),
                "reasons": "; ".join(rec.get("reasons", [])),
            }
        )

    graph_nodes = graph_data.get("nodes")
    if isinstance(graph_nodes, list):
        nodes_list = [node for node in graph_nodes if isinstance(node, dict)]
    else:
        nodes_list = []
    details_by_id = _build_graph_details(
        seed_detail=seed_detail,
        recs=recs if isinstance(recs, list) else [],
        graph_nodes=nodes_list,
    )

    rec_bibcodes = [
        rec.get("bibcode")
        for rec in recs
        if isinstance(rec, dict) and isinstance(rec.get("bibcode"), str)
    ]
    _enrich_paper_details(
        details_by_id,
        [selected_bibcode, *[bibcode for bibcode in rec_bibcodes if isinstance(bibcode, str)]],
    )

    paper_node_ids: list[str] = []
    for node in nodes_list:
        node_id = node.get("id")
        node_type = node.get("type")
        if isinstance(node_id, str) and node_type in {"seed", "recommended"}:
            paper_node_ids.append(node_id)
    years = [
        year
        for paper_id in paper_node_ids
        for year in [_safe_int(details_by_id.get(paper_id, {}).get("year"))]
        if year is not None
    ]
    citations = [
        count
        for paper_id in paper_node_ids
        for count in [_safe_int(details_by_id.get(paper_id, {}).get("citation_count"))]
        if count is not None
    ]
    pageranks = [
        value
        for paper_id in paper_node_ids
        for value in [_safe_float(details_by_id.get(paper_id, {}).get("pagerank"))]
        if value is not None
    ]
    available_communities = sorted(
        {
            community_id
            for paper_id in paper_node_ids
            for community_id in [_safe_int(details_by_id.get(paper_id, {}).get("community_id"))]
            if community_id is not None
        }
    )

    st.sidebar.markdown("### Graph Filters")
    show_papers = st.sidebar.checkbox("Show papers", value=True)
    show_keywords = st.sidebar.checkbox("Show keywords", value=include_keywords)
    if years:
        year_range = st.sidebar.slider(
            "Publication year",
            min_value=min(years),
            max_value=max(years),
            value=(min(years), max(years)),
        )
    else:
        year_range = (1900, 2100)

    max_citations = max(citations) if citations else 0
    min_citations = st.sidebar.slider(
        "Min citations",
        min_value=0,
        max_value=max(1, max_citations),
        value=0,
    )
    if max_citations == 0:
        min_citations = 0

    max_pagerank = max(pageranks) if pageranks else 0.0
    min_pagerank = st.sidebar.slider(
        "Min PageRank",
        min_value=0.0,
        max_value=max(0.0001, max_pagerank),
        value=0.0,
        step=0.0001,
    )
    if max_pagerank == 0.0:
        min_pagerank = 0.0

    selected_communities = st.sidebar.multiselect(
        "Communities",
        options=available_communities,
        default=[],
    )

    filtered_graph = _filter_subgraph(
        graph_data,
        details_by_id=details_by_id,
        show_papers=show_papers,
        show_keywords=show_keywords,
        year_range=(int(year_range[0]), int(year_range[1])),
        min_citations=min_citations,
        min_pagerank=min_pagerank,
        communities={int(value) for value in selected_communities},
    )

    rec_tab, graph_tab, stats_tab, schema_tab = st.tabs(
        ["Recommendations", "Graph View", "Graph Stats", "Schema View"]
    )

    with rec_tab:
        st.subheader("Recommendations")
        st.dataframe(table_rows, use_container_width=True)

        st.subheader("Recommendation Details")
        for rec in recs:
            bibcode = rec.get("bibcode")
            if not isinstance(bibcode, str):
                continue

            with st.expander(f"{rec.get('title') or '(untitled)'} [{bibcode}]", expanded=False):
                st.markdown(f"**Score:** {rec.get('score')}")
                reasons = rec.get("reasons", [])
                if isinstance(reasons, list) and reasons:
                    st.markdown("**Reasons:**")
                    for reason in reasons:
                        st.markdown(f"- {reason}")

                try:
                    paper = _api_get(f"/paper/{bibcode}")
                    st.markdown("**Abstract:**")
                    _render_abstract(paper.get("abstract"))
                except requests.RequestException:
                    st.markdown("Abstract unavailable.")

    with graph_tab:
        st.subheader("Knowledge Graph View")
        st.markdown(
            """
            <div style="display:flex;gap:14px;flex-wrap:wrap;padding:10px 12px;
                        border:1px solid #e5e7eb;border-radius:8px;background:#f8fafc;">
              <span><b style="color:#f97316;">★</b> Seed paper</span>
              <span><b style="color:#3b82f6;">●</b> Recommended paper</span>
              <span><b style="color:#94a3b8;">■</b> Keyword node</span>
              <span><b style="color:#f59e0b;">━</b> Recommendation edge</span>
              <span><b style="color:#94a3b8;">━</b> CITES edge</span>
              <span><b style="color:#9ca3af;">┈</b> HAS_KEYWORD edge</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Tip: drag to pan, scroll to zoom. Reload the page to reset camera position.")

        filtered_nodes = filtered_graph.get("nodes", [])
        filtered_node_dicts = [node for node in filtered_nodes if isinstance(node, dict)]
        filtered_edges = filtered_graph.get("edges", [])

        if not filtered_node_dicts:
            st.warning("No nodes match the active filters. Relax filters to view the graph.")
            return

        search_term = st.text_input("Search nodes (title, bibcode, abstract, reasons)", value="")
        highlighted_nodes = _matching_node_ids(filtered_node_dicts, details_by_id, search_term)
        highlighted_edges: set[tuple[str, str]] = set()

        paper_choices: dict[str, str] = {}
        for node in filtered_node_dicts:
            if node.get("type") not in {"seed", "recommended"}:
                continue
            node_id = node.get("id")
            if not isinstance(node_id, str):
                continue
            label = _node_display_label(
                node,
                show_full_titles=True,
                max_label_len=max_label_len,
            )
            paper_choices[f"{label} [{node_id}]"] = node_id

        with st.expander("Find shortest path", expanded=False):
            if len(paper_choices) < 2:
                st.caption("Need at least two visible paper nodes for path exploration.")
            else:
                sorted_paper_labels = sorted(paper_choices.keys())
                paper_a = st.selectbox("From paper", options=sorted_paper_labels, key="path_from")
                paper_b = st.selectbox("To paper", options=sorted_paper_labels, key="path_to")
                if st.button("Find Path", key="find_path_btn"):
                    source_id = paper_choices[paper_a]
                    target_id = paper_choices[paper_b]
                    try:
                        filtered_edge_dicts = [
                            edge for edge in filtered_edges if isinstance(edge, dict)
                        ]
                        path_nodes, path_edges = _compute_shortest_path_edges(
                            {"nodes": filtered_node_dicts, "edges": filtered_edge_dicts},
                            source_id=source_id,
                            target_id=target_id,
                        )
                        highlighted_nodes.update(path_nodes)
                        highlighted_edges.update(path_edges)
                        st.success("Path found: " + " -> ".join(path_nodes))
                    except nx.NetworkXNoPath:
                        st.warning("No path found between the selected papers in the current view.")
                    except nx.NodeNotFound:
                        st.warning("Selected nodes are not available in the current graph.")

        _render_subgraph(
            filtered_graph,
            detail_by_id=details_by_id,
            show_full_titles=show_full_titles,
            max_label_len=max_label_len,
            enable_physics=enable_physics,
            stabilization_iterations=stabilization_iterations,
            show_edge_labels=show_edge_labels,
            highlighted_nodes=highlighted_nodes,
            highlighted_edges=highlighted_edges,
        )

        node_choices: dict[str, str] = {}
        for node in filtered_node_dicts:
            node_id = node.get("id")
            if not isinstance(node_id, str):
                continue
            label = _node_display_label(
                node,
                show_full_titles=True,
                max_label_len=max_label_len,
            )
            node_choices[f"{label} [{node_id}]"] = node_id

        if node_choices:
            selected_node_label = st.selectbox(
                "Inspect node details",
                options=sorted(node_choices.keys()),
            )
            selected_node_id = node_choices[selected_node_label]
            selected_detail = details_by_id.get(selected_node_id, {})
            st.markdown(f"**Bibcode/ID:** `{selected_node_id}`")
            st.markdown(f"**Title:** {selected_detail.get('title') or '(untitled)'}")
            st.markdown(f"**Year:** {selected_detail.get('year') or 'Unknown'}")
            citation_count = selected_detail.get("citation_count")
            citation_display = citation_count if citation_count is not None else "Unknown"
            st.markdown(
                f"**Citations:** "
                f"{citation_display}"
            )
            pagerank_value = selected_detail.get("pagerank")
            pagerank_display = pagerank_value if pagerank_value is not None else "Unknown"
            st.markdown(
                f"**PageRank:** "
                f"{pagerank_display}"
            )

            score = selected_detail.get("score")
            if isinstance(score, (int, float)):
                st.markdown(f"**Score:** {float(score):.4f}")

            reasons_raw = selected_detail.get("reasons")
            if isinstance(reasons_raw, list):
                reasons = [reason for reason in reasons_raw if isinstance(reason, str)]
                if reasons:
                    st.markdown("**Reasons:**")
                    for reason in reasons:
                        st.markdown(f"- {reason}")

            node_type = selected_detail.get("type")
            if node_type in {"seed", "recommended"}:
                paper_payload = selected_detail
                if not isinstance(paper_payload.get("abstract"), str):
                    try:
                        paper_payload = _api_get(f"/paper/{selected_node_id}")
                    except requests.RequestException:
                        paper_payload = selected_detail
                st.markdown("**Abstract:**")
                _render_abstract(paper_payload.get("abstract"))

    with stats_tab:
        if stats_error:
            st.warning(f"Stats endpoint unavailable: {stats_error}")
        elif not stats_overview:
            st.warning("No graph stats available.")
        else:
            _render_graph_stats_tab(stats_overview)

    with schema_tab:
        st.subheader("Knowledge Graph Schema")
        summary = stats_overview.get("summary") if isinstance(stats_overview, dict) else None
        if isinstance(summary, dict):
            schema_stats = SchemaStats(
                paper_count=_safe_int(summary.get("paper_count")) or 0,
                author_count=_safe_int(summary.get("author_count")) or 0,
                keyword_count=_safe_int(summary.get("keyword_count")) or 0,
                venue_count=_safe_int(summary.get("venue_count")) or 0,
                cites_count=_safe_int(summary.get("cites_count")) or 0,
            )
        else:
            schema_stats = SchemaStats(
                paper_count=0,
                author_count=0,
                keyword_count=0,
                venue_count=0,
                cites_count=0,
            )

        schema_svg = render_schema_svg(schema_stats)
        components_html(schema_svg, height=460, scrolling=False)
        st.caption(
            "Entity counts are from /stats/overview. "
            "Relationships shown: CITES, WROTE, HAS_KEYWORD, PUBLISHED_IN."
        )


if __name__ == "__main__":
    main()
