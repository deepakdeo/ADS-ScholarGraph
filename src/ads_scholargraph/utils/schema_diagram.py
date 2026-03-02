"""Schema diagram helpers for rendering a lightweight KG overview SVG."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape


@dataclass(frozen=True)
class SchemaStats:
    """Counts used to annotate the schema diagram."""

    paper_count: int
    author_count: int
    keyword_count: int
    venue_count: int
    cites_count: int


def render_schema_svg(stats: SchemaStats) -> str:
    """Render an inline SVG schema diagram with node and edge counts."""

    paper_color = "#3b82f6"
    author_color = "#e5e7eb"
    keyword_color = "#f59e0b"
    venue_color = "#60a5fa"
    edge_color = "#94a3b8"
    text_color = "#0f172a"
    bg_color = "#f8fafc"

    return f"""
<svg xmlns="http://www.w3.org/2000/svg" width="980" height="420" viewBox="0 0 980 420">
  <rect x="0" y="0" width="980" height="420" fill="{bg_color}" />
  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
      <polygon points="0 0, 8 4, 0 8" fill="{edge_color}" />
    </marker>
  </defs>

  <rect x="70" y="140" width="250" height="140" rx="12" fill="{paper_color}" opacity="0.92" />
  <text x="90" y="170" font-size="18" fill="white" font-weight="700">Paper</text>
  <text x="90" y="196" font-size="13" fill="white">count: {stats.paper_count}</text>
  <text x="90" y="220" font-size="12" fill="white">bibcode, title, year</text>
  <text x="90" y="240" font-size="12" fill="white">abstract, citation_count</text>
  <text x="90" y="260" font-size="12" fill="white">pagerank, community_id</text>

  <rect x="390" y="50" width="220" height="120" rx="12" fill="{author_color}" opacity="0.95" />
  <text x="410" y="80" font-size="17" fill="{text_color}" font-weight="700">Author</text>
  <text x="410" y="106" font-size="13" fill="{text_color}">count: {stats.author_count}</text>
  <text x="410" y="128" font-size="12" fill="{text_color}">name, betweenness</text>

  <rect x="390" y="250" width="220" height="120" rx="12" fill="{keyword_color}" opacity="0.92" />
  <text x="410" y="280" font-size="17" fill="{text_color}" font-weight="700">Keyword</text>
  <text x="410" y="306" font-size="13" fill="{text_color}">count: {stats.keyword_count}</text>
  <text x="410" y="328" font-size="12" fill="{text_color}">name</text>

  <rect x="700" y="140" width="210" height="120" rx="12" fill="{venue_color}" opacity="0.92" />
  <text x="720" y="170" font-size="17" fill="white" font-weight="700">Venue</text>
  <text x="720" y="196" font-size="13" fill="white">count: {stats.venue_count}</text>
  <text x="720" y="218" font-size="12" fill="white">name</text>

  <line x1="320" y1="180" x2="390" y2="110"
        stroke="{edge_color}" stroke-width="2.5" marker-end="url(#arrow)" />
  <text x="325" y="120" font-size="12" fill="{text_color}">WROTE</text>

  <line x1="320" y1="220" x2="390" y2="300"
        stroke="{edge_color}" stroke-width="2.5" stroke-dasharray="4,4"
        marker-end="url(#arrow)" />
  <text x="322" y="300" font-size="12" fill="{text_color}">HAS_KEYWORD</text>

  <line x1="610" y1="205" x2="700" y2="200"
        stroke="{edge_color}" stroke-width="2.5" marker-end="url(#arrow)" />
  <text x="620" y="190" font-size="12" fill="{text_color}">PUBLISHED_IN</text>

  <path d="M 160 138 C 130 70, 265 70, 240 138" fill="none"
        stroke="{edge_color}" stroke-width="2.5" marker-end="url(#arrow)" />
  <text x="133" y="72" font-size="12" fill="{text_color}">CITES ({stats.cites_count})</text>

  <text x="30" y="390" font-size="12" fill="{escape(text_color)}">
    ADS ScholarGraph schema overview: entities and core relationships
  </text>
</svg>
""".strip()
