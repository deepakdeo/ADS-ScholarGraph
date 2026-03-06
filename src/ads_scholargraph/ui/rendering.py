"""Safe rendering helpers for ADS abstract HTML snippets."""

from __future__ import annotations

import html
import re
from typing import Any

import bleach

_ALLOWED_TAGS = ["sub", "sup", "b", "i", "em", "strong", "br", "p", "span"]


def sanitize_ads_html(text: str) -> str:
    """Sanitize ADS abstract HTML, preserving only a small safe tag subset."""

    without_scripts = re.sub(r"(?is)<script.*?>.*?</script>", "", text)
    cleaned = bleach.clean(
        without_scripts,
        tags=_ALLOWED_TAGS,
        attributes={},
        strip=True,
    )
    # Normalize any remaining entities and strip null bytes defensively.
    return html.unescape(cleaned).replace("\x00", "")


def shorten_title(title: str, max_len: int) -> str:
    """Truncate title to a readable label length with ellipsis."""

    if max_len <= 0:
        return ""

    compact = " ".join(title.split())
    if len(compact) <= max_len:
        return compact
    if max_len == 1:
        return "…"
    return compact[: max_len - 1].rstrip() + "…"


def build_tooltip(node: dict[str, Any], detail: dict[str, Any] | None = None) -> str:
    """Build HTML tooltip content for graph nodes with full metadata."""

    node_type = str(node.get("type", "recommended"))
    title = node.get("title") if isinstance(node.get("title"), str) else ""
    year = node.get("year") if isinstance(node.get("year"), int) else None

    bibcode = node.get("id") if isinstance(node.get("id"), str) else None
    if isinstance(detail, dict):
        if isinstance(detail.get("bibcode"), str):
            bibcode = detail["bibcode"]
        if isinstance(detail.get("title"), str):
            title = detail["title"]
        if isinstance(detail.get("year"), int):
            year = detail["year"]

    safe_title = html.escape(title or "(untitled)")
    lines: list[str] = [
        (
            "<div style='font-family:Trebuchet MS, Gill Sans, sans-serif;"
            "max-width:320px;padding:10px 12px;border-radius:14px;"
            "background:#172033;color:#f8fafc;box-shadow:0 12px 28px rgba(15,23,42,0.28);'>"
        ),
        (
            "<div style='font-weight:700;font-size:14px;line-height:1.4;margin-bottom:6px;'>"
            f"{safe_title}</div>"
        ),
    ]

    if bibcode:
        lines.append(
            f"<div style='color:#e2e8f0;'>Bibcode: {html.escape(str(bibcode))}</div>"
        )
    if year is not None:
        lines.append(f"<div style='color:#e2e8f0;'>Year: {year}</div>")
    lines.append(f"<div style='color:#e2e8f0;'>Type: {html.escape(node_type)}</div>")

    if isinstance(detail, dict):
        citation_count = detail.get("citation_count")
        if isinstance(citation_count, int):
            lines.append(f"<div style='color:#e2e8f0;'>Citations: {citation_count}</div>")

        pagerank = detail.get("pagerank")
        if isinstance(pagerank, (int, float)):
            lines.append(
                f"<div style='color:#e2e8f0;'>PageRank: {float(pagerank):.6f}</div>"
            )

        community_id = detail.get("community_id")
        if isinstance(community_id, int):
            lines.append(f"<div style='color:#e2e8f0;'>Community: {community_id}</div>")

        score = detail.get("score")
        if isinstance(score, (int, float)):
            lines.append(f"<div style='color:#e2e8f0;'>Score: {float(score):.4f}</div>")
        reasons = detail.get("reasons")
        if isinstance(reasons, list):
            reason_lines = [str(reason) for reason in reasons if isinstance(reason, str)]
            if reason_lines:
                lines.append(
                    "<div style='margin-top:8px;margin-bottom:4px;color:#cbd5e1;'>Reasons:</div>"
                )
                for reason in reason_lines:
                    lines.append(f"<div>&bull; {html.escape(reason)}</div>")

    lines.append("</div>")
    return "".join(lines)
