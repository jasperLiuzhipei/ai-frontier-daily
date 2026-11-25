"""Hacker News Algolia API fetcher for AI & Agent Harness discussions."""

from __future__ import annotations

import time
import urllib.parse
from typing import Any

from ai_daily.fetchers.rss import parse_datetime, strip_html
from ai_daily.http_client import HTTPClient
from ai_daily.models import Article

HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"


def fetch_hackernews_source(
    source: dict[str, Any], http: HTTPClient, window_hours: int = 36
) -> list[Article]:
    queries = source.get("queries") or ["agent harness", "LLM", "Gemini", "DeepSeek"]
    hn_tags = str(source.get("hn_tags") or "story")
    min_points = int(source.get("min_points", 25))
    max_items = int(source.get("max_items", 18))
    cutoff_ts = int(time.time()) - max(window_hours, 24) * 3600

    seen_ids: set[str] = set()
    articles: list[Article] = []

    for query in queries:
        params = urllib.parse.urlencode(
            {
                "query": query,
                "tags": hn_tags,
                "numericFilters": f"created_at_i>{cutoff_ts},points>={min_points}",
                "hitsPerPage": 10,
            }
        )
        url = f"{HN_SEARCH_URL}?{params}"
        resp = http.get(url)
        data = resp.json()
        for hit in data.get("hits", []):
            oid = str(hit.get("objectID") or "")
            if not oid or oid in seen_ids:
                continue
            seen_ids.add(oid)
            title = (hit.get("title") or "").strip()
            if not title:
                continue
            story_url = (hit.get("url") or "").strip()
            hn_url = f"https://news.ycombinator.com/item?id={oid}"
            target_url = story_url or hn_url
            points = int(hit.get("points") or 0)
            comments = int(hit.get("num_comments") or 0)
            story_text = strip_html(hit.get("story_text") or "", max_chars=280)
            summary = f"HN 热议：{points} 分 / {comments} 条评论（讨论：{hn_url}）"
            if story_text:
                summary = f"{summary} | {story_text}"
            articles.append(
                Article(
                    source_id=source["id"],
                    source_name=source.get("name", "Hacker News"),
                    category=source.get("category", "harness"),
                    title=title,
                    url=target_url,
                    summary=summary,
                    published_at=parse_datetime(hit.get("created_at")),
                    author=str(hit.get("author") or ""),
                    tags=list(source.get("tags", [])),
                    metrics={"points": points, "comments": comments, "hn_url": hn_url},
                )
            )

    articles.sort(key=lambda a: int(a.metrics.get("points", 0)), reverse=True)
    return articles[:max_items]
