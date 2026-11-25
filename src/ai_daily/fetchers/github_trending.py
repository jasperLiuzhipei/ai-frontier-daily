"""GitHub Search API fetcher for rising AI & Agent Harness open-source repos."""

from __future__ import annotations

import os
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Any

from ai_daily.fetchers.rss import parse_datetime
from ai_daily.http_client import HTTPClient
from ai_daily.models import Article

GITHUB_SEARCH_REPOS = "https://api.github.com/search/repositories"


def fetch_github_trending_source(source: dict[str, Any], http: HTTPClient) -> list[Article]:
    queries = source.get("queries") or ["agent harness", "coding agent LLM"]
    min_stars = int(source.get("min_stars", 50))
    max_items = int(source.get("max_items", 10))
    lookback_days = int(source.get("lookback_days", 14))
    since_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    seen_repos: set[str] = set()
    articles: list[Article] = []

    for q in queries:
        full_q = f"({q}) pushed:>={since_date} stars:>={min_stars}"
        params = urllib.parse.urlencode(
            {"q": full_q, "sort": "updated", "order": "desc", "per_page": 8}
        )
        resp = http.get(f"{GITHUB_SEARCH_REPOS}?{params}", headers=headers)
        data = resp.json()
        for repo in data.get("items", []):
            full_name = repo.get("full_name") or ""
            html_url = repo.get("html_url") or ""
            if not full_name or not html_url or full_name in seen_repos:
                continue
            seen_repos.add(full_name)
            stars = int(repo.get("stargazers_count") or 0)
            lang = repo.get("language") or "Unknown"
            desc = (repo.get("description") or "").strip()
            topics = repo.get("topics") or []
            title = f"{full_name} ({stars}★)"
            summary = f"[{lang}] {desc}"
            if topics:
                summary += f" | Topics: {', '.join(topics[:6])}"
            articles.append(
                Article(
                    source_id=source["id"],
                    source_name=source.get("name", "GitHub AI"),
                    category=source.get("category", "tech_share"),
                    title=title,
                    url=html_url,
                    summary=summary[:420],
                    published_at=parse_datetime(repo.get("pushed_at") or repo.get("updated_at")),
                    author=full_name.split("/")[0],
                    tags=list(source.get("tags", [])),
                    metrics={"stars": stars, "language": lang},
                )
            )

    articles.sort(key=lambda a: int(a.metrics.get("stars", 0)), reverse=True)
    return articles[:max_items]
