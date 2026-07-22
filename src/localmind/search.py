from __future__ import annotations

import httpx
import json
import re

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    published_date: str | None = None


class SearxngSearch:
    def __init__(self, base_url: str, timeout_seconds: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, max_results: int = 10) -> str:
        if not query.strip():
            return "Search error: query cannot be empty."

        limit = max(1, min(max_results, 10))
        params = {"q": query, "format": "json"}
        lowered_query = query.lower()
        if re.search(r"\b(?:news|headlines?)\b", lowered_query):
            params["categories"] = "news"
            if re.search(
                r"\b(?:latest|recent|current|today|breaking|updates?)\b",
                lowered_query,
            ):
                params["time_range"] = "month"
        try:
            response = httpx.get(
                f"{self.base_url}/search",
                params=params,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 403:
                return (
                    "Search error: SearXNG rejected JSON output. Enable JSON format in "
                    "SearXNG settings.yml with search.formats including json."
                )
            return f"Search error: SearXNG returned HTTP {exc.response.status_code}."
        except Exception as exc:
            return f"Search error: could not reach SearXNG at {self.base_url}: {exc}"

        results = normalize_searxng_results(payload, limit)
        if not results:
            return "No search results found."
        return json.dumps([result.__dict__ for result in results], ensure_ascii=True)


def normalize_searxng_results(payload: dict[str, Any], max_results: int) -> list[SearchResult]:
    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        return []

    normalized: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("content") or item.get("snippet") or "").strip()
        published_date = item.get("publishedDate") or item.get("published_date")
        if not title or not url:
            continue
        normalized.append(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                published_date=str(published_date) if published_date else None,
            )
        )
        if len(normalized) >= max_results:
            break
    return normalized
