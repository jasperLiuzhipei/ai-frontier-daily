"""Hugging Face Daily Papers JSON API fetcher."""

from __future__ import annotations

from typing import Any

from ai_daily.fetchers.rss import parse_datetime, strip_html
from ai_daily.http_client import HTTPClient
from ai_daily.models import Article

HF_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"


def fetch_hf_papers_source(source: dict[str, Any], http: HTTPClient) -> list[Article]:
    url = source.get("url") or HF_DAILY_PAPERS_URL
    min_upvotes = int(source.get("min_upvotes", 10))
    max_items = int(source.get("max_items", 15))

    resp = http.get(url)
    payload = resp.json()
    if not isinstance(payload, list):
        return []

    articles: list[Article] = []
    for entry in payload:
        paper = entry.get("paper") if isinstance(entry, dict) else None
        if not isinstance(paper, dict):
            continue
        paper_id = str(paper.get("id") or "").strip()
        title = (entry.get("title") or paper.get("title") or "").strip()
        if not paper_id or not title:
            continue
        upvotes = int(paper.get("upvotes") or entry.get("upvotes") or 0)
        if upvotes < min_upvotes:
            continue
        summary_raw = entry.get("summary") or paper.get("summary") or ""
        hf_url = f"https://huggingface.co/papers/{paper_id}"
        pub_raw = entry.get("publishedAt") or paper.get("publishedAt")
        authors_list = paper.get("authors") or []
        author_names = [
            a.get("name", "") for a in authors_list[:3] if isinstance(a, dict) and a.get("name")
        ]
        summary = f"HF Daily Papers（{upvotes} upvotes）| {strip_html(summary_raw, max_chars=360)}"
        articles.append(
            Article(
                source_id=source["id"],
                source_name=source.get("name", "Hugging Face Daily Papers"),
                category=source.get("category", "tech_share"),
                title=title,
                url=hf_url,
                summary=summary,
                published_at=parse_datetime(pub_raw),
                author=", ".join(author_names),
                tags=list(source.get("tags", [])),
                metrics={"upvotes": upvotes, "arxiv_id": paper_id},
            )
        )

    articles.sort(key=lambda a: int(a.metrics.get("upvotes", 0)), reverse=True)
    return articles[:max_items]
