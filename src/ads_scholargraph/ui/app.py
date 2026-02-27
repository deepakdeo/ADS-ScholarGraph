"""Streamlit dashboard for seed search and explainable recommendations."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def _api_get(path: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _paper_label(paper: dict[str, Any]) -> str:
    title = paper.get("title") or "(untitled)"
    year = paper.get("year") if isinstance(paper.get("year"), int) else ""
    bibcode = paper.get("bibcode") or ""
    return f"{title} ({year}) [{bibcode}]"


def main() -> None:
    st.set_page_config(page_title="ADS ScholarGraph Recommender", layout="wide")

    st.title("ADS ScholarGraph Recommender")
    st.caption("Search seed papers and generate explainable recommendations from the local KG.")

    st.sidebar.header("Recommendation Settings")
    mode = st.sidebar.selectbox("Mode", options=["hybrid", "graph", "embed"], index=0)
    k = st.sidebar.slider("Top-K", min_value=3, max_value=30, value=10, step=1)

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

    col_left, col_right = st.columns([1, 3])
    with col_left:
        trigger = st.button("Recommend", type="primary")

    if not trigger:
        return

    try:
        seed_detail = _api_get(f"/paper/{selected_bibcode}")
        recs = _api_get(
            f"/recommend/paper/{selected_bibcode}",
            params={"k": k, "mode": mode},
        )
    except requests.RequestException as exc:
        st.error(f"Recommendation request failed: {exc}")
        return

    with st.expander("Seed paper abstract", expanded=False):
        st.markdown(seed_detail.get("abstract") or "No abstract available.")

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
                st.markdown(paper.get("abstract") or "No abstract available.")
            except requests.RequestException:
                st.markdown("Abstract unavailable.")


if __name__ == "__main__":
    main()
