"""Streamlit dashboard for seed search and explainable recommendations."""

from __future__ import annotations

import json
import os
from typing import Any

import requests
import streamlit as st
from pyvis.network import Network
from streamlit.components.v1 import html as components_html

from ads_scholargraph.ui.rendering import build_tooltip, sanitize_ads_html, shorten_title

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


def _render_subgraph(
    subgraph: dict[str, Any],
    *,
    detail_by_id: dict[str, dict[str, Any]],
    show_full_titles: bool,
    max_label_len: int,
    enable_physics: bool,
    stabilization_iterations: int,
    show_edge_labels: bool,
) -> None:
    net = Network(height=f"{GRAPH_HEIGHT_PX}px", width="100%", directed=True, bgcolor="#ffffff")

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

        net.add_node(
            node_id,
            label=label,
            title=tooltip,
            color=color_map.get(node_type, "#64748b"),
            shape=shape_map.get(node_type, "dot"),
            size=size_map.get(node_type, 22),
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

        net.add_edge(
            source,
            target,
            label=label if show_edge_labels else "",
            title=label,
            color=color,
            width=width,
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

    rec_tab, graph_tab = st.tabs(["Recommendations", "Graph View"])

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
            **Legend**
            - Orange star: Seed paper
            - Blue circle: Recommended paper
            - Gray box: Keyword node (optional)
            - Orange edge: Recommendation rationale
            - Gray edge: Citation link (`CITES`)
            """
        )
        st.caption("Tip: drag to pan, scroll to zoom. Reload the page to reset camera position.")

        graph_nodes = graph_data.get("nodes")
        nodes_list = graph_nodes if isinstance(graph_nodes, list) else []
        details_by_id = _build_graph_details(
            seed_detail=seed_detail,
            recs=recs if isinstance(recs, list) else [],
            graph_nodes=[node for node in nodes_list if isinstance(node, dict)],
        )

        _render_subgraph(
            graph_data,
            detail_by_id=details_by_id,
            show_full_titles=show_full_titles,
            max_label_len=max_label_len,
            enable_physics=enable_physics,
            stabilization_iterations=stabilization_iterations,
            show_edge_labels=show_edge_labels,
        )

        node_choices: dict[str, str] = {}
        for node in nodes_list:
            if not isinstance(node, dict):
                continue
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


if __name__ == "__main__":
    main()
