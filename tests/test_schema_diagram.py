from ads_scholargraph.utils.schema_diagram import SchemaStats, render_schema_svg


def test_render_schema_svg_includes_counts_and_labels() -> None:
    svg = render_schema_svg(
        SchemaStats(
            paper_count=11,
            author_count=7,
            keyword_count=19,
            venue_count=3,
            cites_count=42,
        )
    )

    assert "<svg" in svg
    assert "Paper" in svg
    assert "Author" in svg
    assert "Keyword" in svg
    assert "Venue" in svg
    assert "CITES (42)" in svg
    assert "count: 11" in svg
    assert "count: 7" in svg
