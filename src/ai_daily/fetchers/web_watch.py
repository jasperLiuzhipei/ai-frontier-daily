"""Lightweight HTML monitor for official changelogs and AI portals without native RSS."""

from __future__ import annotations

import hashlib
import re
import urllib.parse
from html.parser import HTMLParser
from typing import Any

from ai_daily.fetchers.rss import parse_datetime
from ai_daily.http_client import HTTPClient
from ai_daily.models import Article

_WS_RE = re.compile(r"\s+")
_DATE_IN_TEXT_RE = re.compile(
    r"(20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+20\d{2})",
    re.IGNORECASE,
)


class _LinkListParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._in_a = False
        self._href = ""
        self._buf: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        low = tag.lower()
        if low in ("script", "style", "noscript"):
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and low == "a":
            attr_map = dict(attrs)
            href = (attr_map.get("href") or "").strip()
            if href:
                self._in_a = True
                self._href = href
                self._buf = []

    def handle_endtag(self, tag: str) -> None:
        low = tag.lower()
        if low in ("script", "style", "noscript") and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if low == "a" and self._in_a:
            text = _WS_RE.sub(" ", "".join(self._buf)).strip()
            if text and self._href:
                self.links.append((self._href, text))
            self._in_a = False
            self._href = ""
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and self._in_a:
            self._buf.append(data)


class _ChangelogParser(HTMLParser):
    """Extracts heading-delimited changelog sections (h2/h3 + nearby <time>/date + p/li text)."""

    def __init__(self) -> None:
        super().__init__()
        self.sections: list[tuple[str, str, str, str]] = []  # (heading, anchor_id, body, date_hint)
        self._skip_depth = 0
        self._in_heading = False
        self._in_body_block = False
        self._in_time = False
        self._cur_heading_buf: list[str] = []
        self._cur_anchor = ""
        self._cur_heading = ""
        self._cur_body_parts: list[str] = []
        self._cur_date_hint = ""
        self._pending_date_hint = ""
        self._time_buf: list[str] = []

    def _flush(self) -> None:
        if self._cur_heading:
            body = _WS_RE.sub(" ", " ".join(self._cur_body_parts)).strip()
            self.sections.append((self._cur_heading, self._cur_anchor, body, self._cur_date_hint))
        self._cur_heading = ""
        self._cur_anchor = ""
        self._cur_body_parts = []
        self._cur_date_hint = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        low = tag.lower()
        if low in ("script", "style", "noscript", "nav", "footer", "header"):
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        attr_map = dict(attrs)
        if low == "time":
            self._in_time = True
            self._time_buf = []
            dt_attr = (attr_map.get("datetime") or "").strip()
            if dt_attr:
                if self._cur_heading and not self._cur_date_hint:
                    self._cur_date_hint = dt_attr
                else:
                    self._pending_date_hint = dt_attr
        elif low in ("h2", "h3"):
            self._flush()
            self._in_heading = True
            self._cur_heading_buf = []
            self._cur_anchor = (attr_map.get("id") or "").strip()
            self._cur_date_hint = self._pending_date_hint
            self._pending_date_hint = ""
        elif low in ("p", "li") and self._cur_heading:
            self._in_body_block = True

    def handle_endtag(self, tag: str) -> None:
        low = tag.lower()
        if low in ("script", "style", "noscript", "nav", "footer", "header") and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return
        if low == "time" and self._in_time:
            txt = _WS_RE.sub(" ", "".join(self._time_buf)).strip()
            if txt:
                if self._cur_heading and not self._cur_date_hint:
                    self._cur_date_hint = txt
                elif not self._pending_date_hint:
                    self._pending_date_hint = txt
            self._in_time = False
        elif low in ("h2", "h3") and self._in_heading:
            self._cur_heading = _WS_RE.sub(" ", "".join(self._cur_heading_buf)).strip()
            self._in_heading = False
        elif low in ("p", "li") and self._in_body_block:
            self._in_body_block = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_time:
            self._time_buf.append(data)
        elif self._in_heading:
            self._cur_heading_buf.append(data)
        elif self._in_body_block and self._cur_heading:
            txt = data.strip()
            if txt:
                self._cur_body_parts.append(txt)
        else:
            # Capture standalone date labels rendered just above changelog headings (e.g. "Nov 13, 2026")
            txt = data.strip()
            if 6 <= len(txt) <= 30:
                m = _DATE_IN_TEXT_RE.search(txt)
                if m:
                    self._pending_date_hint = m.group(1)

    def close(self) -> None:
        super().close()
        self._flush()


def parse_web_watch_html(html_text: str, source: dict[str, Any]) -> list[Article]:
    mode = source.get("mode", "link_list")
    base_url = source["url"]
    max_items = int(source.get("max_items", 10))
    source_id = source["id"]
    source_name = source.get("name", source_id)
    category = source.get("category", "portals")
    tags = list(source.get("tags", []))

    articles: list[Article] = []

    if mode == "changelog":
        parser = _ChangelogParser()
        parser.feed(html_text)
        parser.close()
        for heading, anchor_id, body, date_hint in parser.sections:
            if len(heading) < 4:
                continue
            if not body and not _DATE_IN_TEXT_RE.search(heading):
                continue
            slug = anchor_id or hashlib.sha1(heading.encode("utf-8")).hexdigest()[:10]
            entry_url = f"{base_url.split('#')[0]}#{slug}"
            pub_dt = parse_datetime(date_hint) if date_hint else None
            if pub_dt is None:
                date_match = _DATE_IN_TEXT_RE.search(f"{heading} {body[:260]}")
                pub_dt = parse_datetime(date_match.group(1)) if date_match else None
            articles.append(
                Article(
                    source_id=source_id,
                    source_name=source_name,
                    category=category,
                    title=f"[{source_name}] {heading[:160]}",
                    url=entry_url,
                    summary=body[:480] if body else heading,
                    full_text=body[:2600] if len(body) > 480 else "",
                    published_at=pub_dt,
                    tags=list(tags),
                )
            )
            if len(articles) >= max_items:
                break
        return articles

    # Default: link_list mode
    pattern = re.compile(source.get("link_pattern") or r".+")
    parser = _LinkListParser()
    parser.feed(html_text)
    seen_urls: set[str] = set()
    for href, text in parser.links:
        if len(text) < 8:
            continue
        full_url = urllib.parse.urljoin(base_url, href)
        if not full_url.startswith(("http://", "https://")):
            continue
        if not pattern.search(full_url):
            continue
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)
        date_match = _DATE_IN_TEXT_RE.search(text)
        pub_dt = parse_datetime(date_match.group(1)) if date_match else None
        articles.append(
            Article(
                source_id=source_id,
                source_name=source_name,
                category=category,
                title=text[:200],
                url=full_url,
                summary=text[:360],
                published_at=pub_dt,
                tags=list(tags),
            )
        )
        if len(articles) >= max_items:
            break
    return articles


def fetch_web_watch_source(source: dict[str, Any], http: HTTPClient) -> list[Article]:
    resp = http.get(source["url"])
    return parse_web_watch_html(resp.text(), source)
