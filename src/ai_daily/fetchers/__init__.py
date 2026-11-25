"""Concurrent source dispatcher aggregating AI portals, official blogs, APIs, and web monitors."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ai_daily.fetchers.aihot import fetch_aihot_source
from ai_daily.fetchers.github_trending import fetch_github_trending_source
from ai_daily.fetchers.hackernews import fetch_hackernews_source
from ai_daily.fetchers.hf_papers import fetch_hf_papers_source
from ai_daily.fetchers.hf_spaces import fetch_hf_spaces_source
from ai_daily.fetchers.rss import fetch_rss_source
from ai_daily.fetchers.web_watch import fetch_web_watch_source
from ai_daily.http_client import HTTPClient
from ai_daily.models import SourceFetchResult


def fetch_single_source(
    source: dict[str, Any], http: HTTPClient, window_hours: int = 36
) -> SourceFetchResult:
    sid = source["id"]
    sname = source.get("name", sid)
    stype = source.get("type", "rss")
    t0 = time.monotonic()
    try:
        if stype == "rss":
            articles = fetch_rss_source(source, http)
        elif stype == "aihot":
            articles = fetch_aihot_source(source, http)
        elif stype == "hackernews":
            articles = fetch_hackernews_source(source, http, window_hours=window_hours)
        elif stype == "hf_papers":
            articles = fetch_hf_papers_source(source, http)
        elif stype == "hf_spaces":
            articles = fetch_hf_spaces_source(source, http)
        elif stype == "github_trending":
            articles = fetch_github_trending_source(source, http)
        elif stype == "web_watch":
            articles = fetch_web_watch_source(source, http)
        else:
            raise ValueError(f"Unsupported source type: {stype}")
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return SourceFetchResult(
            source_id=sid,
            source_name=sname,
            ok=True,
            articles=articles,
            elapsed_ms=elapsed_ms,
        )
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return SourceFetchResult(
            source_id=sid,
            source_name=sname,
            ok=False,
            articles=[],
            error=str(exc),
            elapsed_ms=elapsed_ms,
        )


def fetch_all_sources(
    sources: list[dict[str, Any]],
    http: HTTPClient,
    *,
    window_hours: int = 36,
    max_workers: int = 8,
) -> list[SourceFetchResult]:
    enabled_sources = [s for s in sources if s.get("enabled", True)]
    if not enabled_sources:
        return []

    results_by_id: dict[str, SourceFetchResult] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(enabled_sources))) as pool:
        future_map = {
            pool.submit(fetch_single_source, src, http, window_hours): src["id"]
            for src in enabled_sources
        }
        for future in as_completed(future_map):
            sid = future_map[future]
            try:
                results_by_id[sid] = future.result()
            except Exception as exc:
                results_by_id[sid] = SourceFetchResult(
                    source_id=sid, source_name=sid, ok=False, error=str(exc)
                )

    return [results_by_id[s["id"]] for s in enabled_sources if s["id"] in results_by_id]
