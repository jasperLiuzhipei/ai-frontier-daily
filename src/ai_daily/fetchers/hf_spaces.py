"""Hugging Face Trending Spaces JSON API fetcher for fun & interactive AI demos."""

from __future__ import annotations

from typing import Any

from ai_daily.fetchers.rss import parse_datetime
from ai_daily.http_client import HTTPClient
from ai_daily.models import Article

HF_TRENDING_SPACES_URL = (
    "https://huggingface.co/api/spaces?sort=trendingScore&direction=-1&limit=12&full=true"
)


def fetch_hf_spaces_source(source: dict[str, Any], http: HTTPClient) -> list[Article]:
    url = source.get("url") or HF_TRENDING_SPACES_URL
    min_likes = int(source.get("min_likes", 30))
    max_items = int(source.get("max_items", 8))

    resp = http.get(url)
    payload = resp.json()
    if not isinstance(payload, list):
        return []

    articles: list[Article] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        space_id = str(entry.get("id") or "").strip()
        if not space_id or "/" not in space_id:
            continue
        likes = int(entry.get("likes") or 0)
        trending = float(entry.get("trendingScore") or 0.0)
        if likes < min_likes and trending < 10:
            continue
        card = entry.get("cardData") if isinstance(entry.get("cardData"), dict) else {}
        title_text = str(card.get("title") or space_id.split("/", 1)[1]).strip()
        short_desc = str(card.get("short_description") or "").strip()
        sdk = str(entry.get("sdk") or card.get("sdk") or "").strip()
        space_url = f"https://huggingface.co/spaces/{space_id}"
        pub_raw = entry.get("lastModified") or entry.get("createdAt")
        summary_parts = [f"HF Trending Space（{likes} likes, trending {trending:.0f}）"]
        if sdk:
            summary_parts.append(f"SDK: {sdk}")
        if short_desc:
            summary_parts.append(short_desc[:260])
        articles.append(
            Article(
                source_id=source["id"],
                source_name=source.get("name", "Hugging Face Trending Spaces"),
                category=source.get("category", "geek"),
                title=f"HF Space: {title_text} ({space_id})",
                url=space_url,
                summary=" | ".join(summary_parts)[:400],
                published_at=parse_datetime(str(pub_raw)) if pub_raw else None,
                author=space_id.split("/", 1)[0],
                tags=list(source.get("tags", [])),
                metrics={"upvotes": likes, "trending_score": trending},
            )
        )
        if len(articles) >= max_items:
            break

    return articles
