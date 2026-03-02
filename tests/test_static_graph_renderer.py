from __future__ import annotations

import networkx as nx

from ads_scholargraph.utils.static_graph_renderer import (
    graph_to_gexf_bytes,
    render_static_graph_svg,
)


def _sample_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("B1", label="2024 — Seed", type="seed", highlighted=True)
    graph.add_node("R1", label="2023 — Rec", type="recommended", highlighted=False)
    graph.add_edge("B1", "R1", type="RECOMMENDS", highlighted=True)
    return graph


def test_render_static_graph_svg_contains_nodes_and_edges() -> None:
    svg = render_static_graph_svg(_sample_graph(), title="Demo Graph")

    assert "<svg" in svg
    assert "Demo Graph" in svg
    assert "2024 — Seed" in svg
    assert "2023 — Rec" in svg
    assert "line" in svg
    assert "circle" in svg


def test_graph_to_gexf_bytes_serializes_graph() -> None:
    payload = graph_to_gexf_bytes(_sample_graph())
    text = payload.decode("utf-8")

    assert "<gexf" in text
    assert "B1" in text
    assert "R1" in text
