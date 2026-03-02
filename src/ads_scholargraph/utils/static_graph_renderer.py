"""Static SVG graph rendering utilities for export workflows."""

from __future__ import annotations

from html import escape

import networkx as nx


def _scaled_positions(
    graph: nx.Graph,
    *,
    width: int,
    height: int,
    margin: int,
) -> dict[str, tuple[float, float]]:
    if graph.number_of_nodes() == 0:
        return {}

    positions_raw = nx.spring_layout(graph, seed=42)
    xs = [float(pos[0]) for pos in positions_raw.values()]
    ys = [float(pos[1]) for pos in positions_raw.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    usable_w = max(width - (2 * margin), 1)
    usable_h = max(height - (2 * margin), 1)

    scaled: dict[str, tuple[float, float]] = {}
    for node_id, (x_val, y_val) in positions_raw.items():
        x = margin + ((float(x_val) - min_x) / span_x) * usable_w
        y = margin + ((float(y_val) - min_y) / span_y) * usable_h
        scaled[str(node_id)] = (x, y)
    return scaled


def render_static_graph_svg(
    graph: nx.Graph,
    *,
    width: int = 1200,
    height: int = 820,
    title: str = "ADS ScholarGraph Export",
) -> str:
    """Render a networkx graph as a standalone SVG string."""

    margin = 60
    pos = _scaled_positions(graph, width=width, height=height, margin=margin)

    color_by_type = {
        "seed": "#f97316",
        "recommended": "#3b82f6",
        "keyword": "#94a3b8",
    }

    edge_lines: list[str] = []
    for source, target, data in graph.edges(data=True):
        source_pos = pos.get(str(source))
        target_pos = pos.get(str(target))
        if source_pos is None or target_pos is None:
            continue

        highlighted = bool(data.get("highlighted"))
        edge_color = "#ef4444" if highlighted else "#94a3b8"
        edge_width = "3.0" if highlighted else "1.4"
        edge_lines.append(
            (
                f'<line x1="{source_pos[0]:.1f}" y1="{source_pos[1]:.1f}" '
                f'x2="{target_pos[0]:.1f}" y2="{target_pos[1]:.1f}" '
                f'stroke="{edge_color}" stroke-width="{edge_width}" opacity="0.85" />'
            )
        )

    node_shapes: list[str] = []
    label_lines: list[str] = []
    for node_id, data in graph.nodes(data=True):
        node_key = str(node_id)
        node_pos = pos.get(node_key)
        if node_pos is None:
            continue

        node_type = data.get("type") if isinstance(data.get("type"), str) else "recommended"
        highlighted = bool(data.get("highlighted"))
        color = "#facc15" if highlighted else color_by_type.get(node_type, "#64748b")
        radius = 16 if node_type == "seed" else 12
        if node_type == "keyword":
            radius = 9
        if highlighted:
            radius += 5

        node_shapes.append(
            (
                f'<circle cx="{node_pos[0]:.1f}" cy="{node_pos[1]:.1f}" r="{radius}" '
                f'fill="{color}" stroke="#ffffff" stroke-width="1.5" opacity="0.95" />'
            )
        )
        label = data.get("label")
        label_text = str(label) if isinstance(label, str) else node_key
        label_lines.append(
            (
                f'<text x="{node_pos[0]:.1f}" y="{node_pos[1] + radius + 14:.1f}" '
                f'font-size="11" fill="#e5e7eb" text-anchor="middle">{escape(label_text)}</text>'
            )
        )

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect x="0" y="0" width="100%" height="100%" fill="#0b1220"/>',
            (
                f'<text x="24" y="34" font-size="18" fill="#e5e7eb" '
                f'font-weight="700">{escape(title)}</text>'
            ),
            *edge_lines,
            *node_shapes,
            *label_lines,
            "</svg>",
        ]
    )


def graph_to_gexf_bytes(graph: nx.Graph) -> bytes:
    """Serialize a graph to GEXF bytes for download/export."""

    return "\n".join(nx.generate_gexf(graph, prettyprint=True)).encode("utf-8")
