"""Client for ADS Search API requests."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import requests

from ads_scholargraph.config import Settings, get_settings

ADS_SEARCH_URL = "https://api.adsabs.harvard.edu/v1/search/query"
DEFAULT_FIELDS: tuple[str, ...] = (
    "bibcode",
    "title",
    "author",
    "aff",
    "pub",
    "year",
    "abstract",
    "doi",
    "citation_count",
    "keyword",
)


class ADSClient:
    """Thin wrapper around ADS search endpoint with paginated iteration."""

    def __init__(
        self,
        api_token: str,
        base_url: str = ADS_SEARCH_URL,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._api_token = api_token
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> ADSClient:
        """Build a client from environment-backed project settings."""

        resolved_settings = settings or get_settings()
        api_token = (resolved_settings.ADS_API_TOKEN or "").strip()
        if not api_token:
            raise RuntimeError(
                "ADS_API_TOKEN is not set. Add ADS_API_TOKEN=<your_token> to your local .env file."
            )
        return cls(api_token=api_token)

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_token}"}

    def search(
        self,
        q: str,
        *,
        fl: Sequence[str] | None = None,
        rows: int = 200,
        start: int = 0,
        sort: str | None = None,
    ) -> dict[str, Any]:
        """Execute a single ADS search request page and return decoded JSON payload."""

        if rows <= 0:
            raise ValueError("rows must be greater than 0")
        if start < 0:
            raise ValueError("start must be >= 0")

        fields = tuple(fl) if fl is not None else DEFAULT_FIELDS
        params: dict[str, Any] = {
            "q": q,
            "fl": ",".join(fields),
            "rows": rows,
            "start": start,
        }
        if sort:
            params["sort"] = sort

        response = requests.get(
            self._base_url,
            headers=self._headers,
            params=params,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()

        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("ADS response payload was not a JSON object")
        return payload

    def iter_results(
        self,
        q: str,
        *,
        fl: Sequence[str] | None = None,
        rows: int = 200,
        max_results: int = 2000,
        sort: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate over ADS docs while paginating via start/rows until max_results."""

        if max_results <= 0:
            return

        start = 0
        remaining = max_results

        while remaining > 0:
            batch_rows = min(rows, remaining)
            payload = self.search(q, fl=fl, rows=batch_rows, start=start, sort=sort)

            response_obj = payload.get("response", {})
            if not isinstance(response_obj, dict):
                raise ValueError("ADS response['response'] must be an object")

            docs = response_obj.get("docs", [])
            if not isinstance(docs, list):
                raise ValueError("ADS response['response']['docs'] must be a list")

            if not docs:
                break

            yielded = 0
            for doc in docs:
                if isinstance(doc, dict):
                    yield doc
                    yielded += 1

            remaining -= yielded
            start += len(docs)

            if len(docs) < batch_rows:
                break
