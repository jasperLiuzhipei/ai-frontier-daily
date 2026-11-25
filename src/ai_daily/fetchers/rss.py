"""Zero-dependency RSS 2.0 / Atom 1.0 / RDF parser with HTML cleaning and XML recovery."""

from __future__ import annotations

import email.utils
import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any

from ai_daily.http_client import HTTPClient
from ai_daily.models import Article

_BARE_AMP_RE = re.compile(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)")
_WS_RE = re.compile(r"\s+")


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in ("script", "style", "noscript"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in ("script", "style", "noscript") and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag.lower() in ("p", "div", "br", "li", "h1", "h2", "h3", "h4"):
            self._chunks.append(" ")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def get_text(self) -> str:
        return _WS_RE.sub(" ", "".join(self._chunks)).strip()


def strip_html(raw: str, max_chars: int = 420) -> str:
    if not raw:
        return ""
    unescaped = html.unescape(raw)
    parser = _HTMLTextExtractor()
    try:
        parser.feed(unescaped)
        text = parser.get_text()
    except Exception:
        text = re.sub(r"<[^>]+>", " ", unescaped)
        text = _WS_RE.sub(" ", text).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def parse_datetime(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    # RFC 2822 (RSS pubDate)
    try:
        dt = email.utils.parsedate_to_datetime(text)
        if dt is not None:
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass

    # ISO-8601 (Atom / dc:date / APIs)
    iso_text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso_text)
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass

    # Common date-only formats
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y"):
        try:
            dt = datetime.strptime(text[:25], fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1].lower()
    return tag.lower()


def _child_text(elem: ET.Element, names: tuple[str, ...]) -> str:
    for child in elem:
        if _local_name(child.tag) in names:
            txt = "".join(child.itertext()).strip()
            if txt:
                return txt
    return ""


def _extract_atom_link(entry: ET.Element) -> str:
    fallback = ""
    for child in entry:
        if _local_name(child.tag) == "link":
            href = (child.attrib.get("href") or "").strip()
            rel = (child.attrib.get("rel") or "alternate").strip().lower()
            if href and rel == "alternate":
                return href
            if href and not fallback:
                fallback = href
            txt = (child.text or "").strip()
            if txt and not fallback:
                fallback = txt
    return fallback


def parse_feed_xml(xml_text: str, source: dict[str, Any]) -> list[Article]:
    """Parse RSS 2.0, Atom 1.0, or RDF XML into Article objects."""
    cleaned = xml_text.lstrip("\ufeff").strip()
    if not cleaned:
        return []
    try:
        root = ET.fromstring(cleaned)
    except ET.ParseError:
        sanitized = _BARE_AMP_RE.sub("&amp;", cleaned)
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", sanitized)
        root = ET.fromstring(sanitized)

    root_tag = _local_name(root.tag)
    source_id = source["id"]
    source_name = source.get("name", source_id)
    category = source.get("category", "portals")
    base_tags = list(source.get("tags", []))
    max_items = int(source.get("max_items", 15))

    articles: list[Article] = []

    if root_tag == "feed":
        # Atom 1.0
        for entry in root:
            if _local_name(entry.tag) != "entry":
                continue
            title = strip_html(_child_text(entry, ("title",)), max_chars=240)
            url = _extract_atom_link(entry)
            if not title or not url:
                continue
            summary_raw = _child_text(entry, ("summary", "content", "description"))
            pub_raw = _child_text(entry, ("published", "updated", "date"))
            author = _child_text(entry, ("author", "creator"))
            articles.append(
                Article(
                    source_id=source_id,
                    source_name=source_name,
                    category=category,
                    title=title,
                    url=url,
                    summary=strip_html(summary_raw),
                    published_at=parse_datetime(pub_raw),
                    author=strip_html(author, max_chars=80),
                    tags=list(base_tags),
                )
            )
            if len(articles) >= max_items:
                break
    else:
        # RSS 2.0 or RDF
        items: list[ET.Element] = []
        for elem in root.iter():
            if _local_name(elem.tag) == "item":
                items.append(elem)
        for item in items:
            title = strip_html(_child_text(item, ("title",)), max_chars=240)
            url = _child_text(item, ("link", "guid"))
            if not title or not url or not url.startswith(("http://", "https://")):
                continue
            summary_raw = _child_text(item, ("description", "encoded", "summary", "content"))
            pub_raw = _child_text(item, ("pubdate", "date", "published", "updated"))
            author = _child_text(item, ("creator", "author"))
            articles.append(
                Article(
                    source_id=source_id,
                    source_name=source_name,
                    category=category,
                    title=title,
                    url=url.strip(),
                    summary=strip_html(summary_raw),
                    published_at=parse_datetime(pub_raw),
                    author=strip_html(author, max_chars=80),
                    tags=list(base_tags),
                )
            )
            if len(articles) >= max_items:
                break

    return articles


def fetch_rss_source(source: dict[str, Any], http: HTTPClient) -> list[Article]:
    resp = http.get(source["url"])
    return parse_feed_xml(resp.text(), source)
