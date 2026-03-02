import pytest

pytest.importorskip("bleach")

from ads_scholargraph.ui.rendering import build_tooltip, sanitize_ads_html, shorten_title


def test_sanitize_ads_html_keeps_sub_sup() -> None:
    raw = "CO<SUB>2</SUB> and x<SUP>2</SUP>"
    cleaned = sanitize_ads_html(raw)

    assert "<sub>2</sub>" in cleaned.lower()
    assert "<sup>2</sup>" in cleaned.lower()


def test_sanitize_ads_html_removes_script_and_attributes() -> None:
    raw = '<p onclick="alert(1)">Hello</p><script>alert(2)</script>'
    cleaned = sanitize_ads_html(raw)

    assert "onclick" not in cleaned.lower()
    assert "<script" not in cleaned.lower()
    assert "alert(2)" not in cleaned.lower()
    assert "<p>hello</p>" in cleaned.lower()


def test_shorten_title_with_ellipsis() -> None:
    title = "A very long title that should be truncated for graph labels"
    shortened = shorten_title(title, max_len=24)

    assert len(shortened) <= 24
    assert shortened.endswith("…")


def test_shorten_title_non_positive_length_returns_empty() -> None:
    assert shorten_title("Anything", max_len=0) == ""


def test_build_tooltip_includes_full_details() -> None:
    node = {"id": "B1", "type": "recommended", "title": "Node title", "year": 2024}
    detail = {
        "bibcode": "B1",
        "title": "Full Title",
        "year": 2025,
        "citation_count": 12,
        "pagerank": 0.321,
        "community_id": 7,
        "score": 0.81234,
        "reasons": ["Same community", "High similarity"],
    }

    tooltip = build_tooltip(node, detail)

    assert "Full Title" in tooltip
    assert "Bibcode: B1" in tooltip
    assert "Year: 2025" in tooltip
    assert "Citations: 12" in tooltip
    assert "PageRank: 0.321000" in tooltip
    assert "Community: 7" in tooltip
    assert "Score: 0.8123" in tooltip
    assert "Same community" in tooltip
    assert "&bull;" in tooltip
