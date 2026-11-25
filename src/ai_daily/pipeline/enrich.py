"""Stage 1.5 Deep Enrichment: fetches full article text, geek anchors, and HN community comments."""

from __future__ import annotations

import html
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from typing import Any

from ai_daily.fetchers.rss import parse_datetime, strip_html
from ai_daily.http_client import HTTPClient
from ai_daily.models import Article

_WS_RE = re.compile(r"\s+")
_META_PUBDATE_RE = re.compile(
    r'(?:property|name)\s*=\s*["\'](?:article:published_time|og:published_time|pubdate|publishdate|datePublished|dc\.date)["\'][^>]*?content\s*=\s*["\']([^"\']+)["\']|'
    r'content\s*=\s*["\']([^"\']+)["\'][^>]*?(?:property|name)\s*=\s*["\'](?:article:published_time|og:published_time|pubdate|publishdate|datePublished|dc\.date)["\']|'
    r'"datePublished"\s*:\s*"([^"]+)"',
    re.IGNORECASE,
)


def extract_meta_published_date(html_text: str) -> str:
    """Extract canonical page publish date (YYYY-MM-DD) from HTML meta tags or JSON-LD."""
    head = html_text[:25000]
    m = _META_PUBDATE_RE.search(head)
    if not m:
        return ""
    raw = next((g for g in m.groups() if g), "").strip()
    dt = parse_datetime(raw)
    return dt.strftime("%Y-%m-%d") if dt else ""
_SKIP_DOMAINS = (
    "news.google.com",
    "mp.weixin.qq.com",
    "x.com",
    "twitter.com",
    "youtube.com",
    "youtu.be",
    "t.co",
)
_INSTALL_CMD_RE = re.compile(
    r"\b(?:uv\s+tool\s+install|uv\s+pip\s+install|pip\s+install|npm\s+install\s+-g|npx|brew\s+install|ollama\s+run)\s+[A-Za-z0-9@._/:~-]+(?:\s+[A-Za-z0-9@._/:~-]+)?",
    re.IGNORECASE,
)
_REPO_URL_RE = re.compile(
    r"https?://(?:github\.com|huggingface\.co)/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
)
_HN_ITEM_ID_RE = re.compile(r"item\?id=(\d+)")


class _ArticleBodyExtractor(HTMLParser):
    """Extracts readable text prioritizing <article>/<main>, stripping nav/header/footer/scripts."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._in_main_depth = 0
        self._main_chunks: list[str] = []
        self._all_chunks: list[str] = []
        self._in_pre_code = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        low = tag.lower()
        if low in ("script", "style", "noscript", "svg", "nav", "footer", "header", "aside"):
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if low in ("article", "main"):
            self._in_main_depth += 1
        elif low in ("pre", "code"):
            self._in_pre_code = True

    def handle_endtag(self, tag: str) -> None:
        low = tag.lower()
        if low in ("script", "style", "noscript", "svg", "nav", "footer", "header", "aside"):
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return
        if low in ("article", "main") and self._in_main_depth > 0:
            self._in_main_depth -= 1
        elif low in ("pre", "code"):
            self._in_pre_code = False
        elif low in ("p", "div", "li", "h1", "h2", "h3", "h4", "tr", "br"):
            sep = "\n"
            self._all_chunks.append(sep)
            if self._in_main_depth > 0:
                self._main_chunks.append(sep)

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = data if self._in_pre_code else _WS_RE.sub(" ", data)
        if not text.strip():
            return
        self._all_chunks.append(text)
        if self._in_main_depth > 0:
            self._main_chunks.append(text)

    def get_text(self, max_chars: int = 2600) -> str:
        chosen = self._main_chunks if len("".join(self._main_chunks).strip()) >= 350 else self._all_chunks
        raw = "".join(chosen)
        lines = [_WS_RE.sub(" ", ln).strip() for ln in raw.splitlines()]
        compact = "\n".join(ln for ln in lines if len(ln) >= 2)
        if len(compact) > max_chars:
            compact = compact[: max_chars - 1].rstrip() + "…"
        return compact


def extract_article_body_and_anchors(
    html_text: str, max_chars: int = 2600
) -> tuple[str, list[str]]:
    unescaped = html.unescape(html_text)
    parser = _ArticleBodyExtractor()
    try:
        parser.feed(unescaped)
        body = parser.get_text(max_chars=max_chars)
    except Exception:
        body = strip_html(html_text, max_chars=max_chars)

    anchors: list[str] = []
    for m in _INSTALL_CMD_RE.findall(body):
        cmd = _WS_RE.sub(" ", m).strip()
        if cmd not in anchors:
            anchors.append(cmd)
        if len(anchors) >= 3:
            break
    for m in _REPO_URL_RE.findall(html_text):
        clean = m.rstrip(".,;:)\"'")
        if any(bad in clean.lower() for bad in ("/issues", "/pull", "/blob", "/tree", "/releases")):
            continue
        if clean not in anchors:
            anchors.append(clean)
        if len(anchors) >= 5:
            break
    return body, anchors


def fetch_hn_top_comments(
    hn_url: str, http: HTTPClient, max_comments: int = 4
) -> list[str]:
    match = _HN_ITEM_ID_RE.search(hn_url or "")
    if not match:
        return []
    item_id = match.group(1)
    resp = http.get(f"https://hn.algolia.com/api/v1/items/{item_id}", timeout=8)
    data = resp.json()
    children = data.get("children") or []
    comments: list[str] = []
    for child in children:
        if not isinstance(child, dict):
            continue
        txt = strip_html(child.get("text") or "", max_chars=240)
        if len(txt) < 30:
            continue
        author = child.get("author") or "anon"
        comments.append(f"@{author}: {txt}")
        if len(comments) >= max_comments:
            break
    return comments


def _should_fetch_body(url: str) -> bool:
    low = (url or "").lower()
    if not low.startswith(("http://", "https://")):
        return False
    return not any(dom in low for dom in _SKIP_DOMAINS)


def _enrich_single(article: Article, http: HTTPClient, max_chars: int = 2600) -> None:
    parts: list[str] = []
    if article.full_text:
        parts.append(article.full_text[:max_chars])
    elif _should_fetch_body(article.url):
        try:
            resp = http.get(article.url, timeout=8)
            ctype = resp.headers.get("content-type", "").lower()
            if "html" in ctype or "text" in ctype or not ctype:
                raw_html = resp.text()
                page_pub = extract_meta_published_date(raw_html)
                if page_pub:
                    article.metrics["page_published_at"] = page_pub
                body, anchors = extract_article_body_and_anchors(raw_html, max_chars=max_chars)
                if len(body) > len(article.summary) + 80:
                    parts.append(body)
                if anchors:
                    article.metrics["extracted_anchors"] = anchors
        except Exception:
            pass

    hn_url = str(article.metrics.get("hn_url") or "")
    if hn_url:
        try:
            comments = fetch_hn_top_comments(hn_url, http, max_comments=4)
            if comments:
                parts.append("【HN 一线开发者热评】\n" + "\n".join(f"- {c}" for c in comments))
        except Exception:
            pass

    if parts:
        article.full_text = "\n\n".join(parts)[: max_chars + 900]


def enrich_candidates(
    candidates: list[Article],
    http: HTTPClient,
    *,
    max_enrich: int = 18,
    max_chars: int = 2400,
    max_workers: int = 6,
) -> int:
    """Concurrently enrich top candidates with full article text and HN comments (best-effort)."""
    targets = candidates[:max_enrich]
    if not targets:
        return 0

    fast_http = HTTPClient(timeout=8, max_retries=0) if http is None else http
    enriched_count = 0
    with ThreadPoolExecutor(max_workers=min(max_workers, len(targets))) as pool:
        futures = [
            pool.submit(_enrich_single, art, fast_http, max_chars)
            for art in targets
        ]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception:
                pass

    for art in targets:
        if art.full_text:
            enriched_count += 1
    return enriched_count
