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
    lines: list[str] = [f"<b>{safe_title}</b>"]

    if bibcode:
        lines.append(f"Bibcode: {html.escape(str(bibcode))}")
    if year is not None:
        lines.append(f"Year: {year}")
    lines.append(f"Type: {html.escape(node_type)}")

    if isinstance(detail, dict):
        score = detail.get("score")
        if isinstance(score, (int, float)):
            lines.append(f"Score: {float(score):.4f}")
        reasons = detail.get("reasons")
        if isinstance(reasons, list):
            reason_lines = [str(reason) for reason in reasons if isinstance(reason, str)]
            if reason_lines:
                lines.append("Reasons:")
                for reason in reason_lines:
                    lines.append(f"&bull; {html.escape(reason)}")

    return "<br>".join(lines)
