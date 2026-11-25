"""AI Hot (https://aihot.news) public v1 API fetcher with fallback domain and Markdown support."""

from __future__ import annotations

import re
import urllib.parse
from typing import Any

from ai_daily.fetchers.rss import parse_datetime, strip_html
from ai_daily.http_client import HTTPClient
from ai_daily.models import Article

DEFAULT_AIHOT_URLS = [
    "https://aihot.news/api/v1/items",
    "https://aihot.virxact.com/api/v1/items",
]

_MD_LINK_RE = re.compile(r"\[([^\]]{4,200})\]\((https?://[^)\s]+)\)")
_SOURCE_NOISE_RE = re.compile(
    r"(?:[（(]\s*(?:网页|RSS|Feed|Blog|News|官网|官方)\s*[）)]|[:：]\s*(?:News|Blog|Updates|RSS|Feed)\b)",
    re.IGNORECASE,
)


def clean_source_name(name: str) -> str:
    cleaned = _SOURCE_NOISE_RE.sub("", name or "")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ·:-：")
    return cleaned


def _build_query_url(base_url: str, source: dict[str, Any]) -> str:
    if "?" in base_url:
        return base_url
    params: dict[str, Any] = {}
    if "/api/v1/items" in base_url:
        params["mode"] = source.get("mode", "selected")
        params["window"] = source.get("window", "24h")
        params["limit"] = int(source.get("max_items", 20))
        if source.get("query"):
            params["q"] = source["query"]
    if not params:
        return base_url
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def _extract_item_url(item: dict[str, Any]) -> str:
    links = item.get("links")
    if isinstance(links, dict):
        orig = str(links.get("original") or "").strip()
        if orig.startswith(("http://", "https://")):
            return orig
        aihot_link = str(links.get("aihot") or "").strip()
        if aihot_link.startswith(("http://", "https://")):
            return aihot_link
    for key in ("url", "originalUrl", "link", "permalink"):
        val = str(item.get(key) or "").strip()
        if val.startswith(("http://", "https://")):
            return val
    return ""


def _extract_source_label(item: dict[str, Any], default_name: str) -> str:
    src = item.get("source")
    raw = ""
    if isinstance(src, dict):
        raw = str(src.get("name") or src.get("title") or "").strip()
    elif isinstance(src, str):
        raw = src.strip()
    cleaned = clean_source_name(raw)
    if cleaned:
        return f"AI Hot · {cleaned}"
    return default_name


def parse_aihot_json(payload: Any, source: dict[str, Any]) -> list[Article]:
    """Parse AIHOT v1 JSON payload (`items` / `data` / list) into Article objects."""
    if isinstance(payload, dict):
        raw_items = (
            payload.get("items")
            or payload.get("data")
            or payload.get("results")
            or []
        )
        if isinstance(raw_items, dict):
            raw_items = raw_items.get("items") or raw_items.get("list") or []
    elif isinstance(payload, list):
        raw_items = payload
    else:
        return []

    source_id = source["id"]
    source_name = source.get("name", "AI Hot")
    default_cat = source.get("category", "portals")
    base_tags = list(source.get("tags", []))
    max_items = int(source.get("max_items", 20))
    min_score = float(source.get("min_score", 0.0))

    articles: list[Article] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("originalTitle") or item.get("title_en") or "").strip()
        url = _extract_item_url(item)
        if not title or not url:
            continue

        raw_score = item.get("score") or item.get("hotScore") or item.get("qualityScore") or 0
        try:
            score_val = float(raw_score)
        except (TypeError, ValueError):
            score_val = 0.0
        if min_score > 0 and score_val > 0 and score_val < min_score:
            continue

        orig_title = str(item.get("originalTitle") or item.get("title_en") or "").strip()
        summary_raw = str(item.get("summary") or item.get("description") or "").strip()
        summary_parts = []
        if summary_raw:
            summary_parts.append(strip_html(summary_raw, max_chars=340))
        if orig_title and orig_title != title:
            summary_parts.append(f"原标题: {orig_title[:120]}")
        summary = " | ".join(summary_parts)

        pub_raw = (
            item.get("publishedAt")
            or item.get("published_at")
            or item.get("createdAt")
            or item.get("updatedAt")
        )
        item_tags = list(base_tags)
        if item.get("category") and isinstance(item["category"], str):
            item_tags.append(item["category"])
        for t in item.get("tags") or []:
            if isinstance(t, str) and t.strip():
                item_tags.append(t.strip())

        articles.append(
            Article(
                source_id=source_id,
                source_name=_extract_source_label(item, source_name),
                category=default_cat,
                title=title[:220],
                url=url,
                summary=summary[:420],
                published_at=parse_datetime(str(pub_raw)) if pub_raw else None,
                tags=list(dict.fromkeys(item_tags)),
                metrics={"aihot_score": score_val},
            )
        )
        if len(articles) >= max_items:
            break

    return articles


def parse_aihot_markdown(md_text: str, source: dict[str, Any]) -> list[Article]:
    """Fallback parser if an AI Hot `/api/v1/agent/*` endpoint returns Markdown."""
    source_id = source["id"]
    source_name = source.get("name", "AI Hot")
    default_cat = source.get("category", "portals")
    base_tags = list(source.get("tags", []))
    max_items = int(source.get("max_items", 20))

    seen: set[str] = set()
    articles: list[Article] = []
    for line in md_text.splitlines():
        m = _MD_LINK_RE.search(line)
        if not m:
            continue
        title = m.group(1).strip()
        url = m.group(2).strip()
        if url in seen or "aihot.news/api" in url:
            continue
        seen.add(url)
        clean_line = _MD_LINK_RE.sub(title, line).lstrip("#-*0123456789. ").strip()
        articles.append(
            Article(
                source_id=source_id,
                source_name=source_name,
                category=default_cat,
                title=title[:220],
                url=url,
                summary=clean_line[:360],
                tags=list(base_tags),
            )
        )
        if len(articles) >= max_items:
            break
    return articles


def fetch_aihot_source(source: dict[str, Any], http: HTTPClient) -> list[Article]:
    urls: list[str] = []
    if source.get("url"):
        urls.append(str(source["url"]))
    for fb in source.get("fallback_urls") or []:
        if fb and fb not in urls:
            urls.append(str(fb))
    if not urls:
        urls = list(DEFAULT_AIHOT_URLS)

    errors: list[str] = []
    for raw_url in urls:
        target = _build_query_url(raw_url, source)
        try:
            resp = http.get(target, headers={"Accept": "application/json, text/markdown, text/plain"})
            text = resp.text().strip()
            if not text:
                continue
            if text.startswith(("{", "[")):
                articles = parse_aihot_json(resp.json(), source)
            else:
                articles = parse_aihot_markdown(text, source)
            if articles:
                return articles
        except Exception as exc:
            errors.append(f"{target}: {exc}")

    if errors:
        raise RuntimeError("AI Hot endpoints failed: " + " | ".join(errors))
    return []
