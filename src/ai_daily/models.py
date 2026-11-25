"""Core data models for AI Daily."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

TRACK_ORDER: list[tuple[str, str]] = [
    ("harness", "🛠️ Harness 与 AI 产品演进"),
    ("models", "🚀 模型发布与能力边界"),
    ("systems", "⚙️ 大模型训推与系统优化"),
    ("geek", "🧪 极客实测与硬核玩具"),
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


@dataclass
class Article:
    source_id: str
    source_name: str
    category: str
    title: str
    url: str
    summary: str = ""
    full_text: str = ""
    published_at: datetime | None = None
    author: str = ""
    tags: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    canonical_url: str = ""
    url_hash: str = ""
    pre_score: float = 0.0
    matched_keywords: list[str] = field(default_factory=list)


@dataclass
class SourceFetchResult:
    source_id: str
    source_name: str
    ok: bool
    articles: list[Article] = field(default_factory=list)
    error: str = ""
    elapsed_ms: int = 0


@dataclass
class CuratedItem:
    candidate_id: int
    article: Article
    related_articles: list[Article]
    track: str
    score: float
    zh_title: str
    one_liner: str
    why_it_matters: str
    tech_depth: str = ""
    geek_action: str = ""
    section: str = ""
    tech_score: float = 0.0
    fun_score: float = 0.0
    impact_score: float = 0.0
    tags: list[str] = field(default_factory=list)


def format_related_links(
    primary: Article, related_articles: list[Article], max_links: int = 3
) -> list[tuple[str, str]]:
    """Deduplicate related links by canonical URL and guarantee distinct human-readable labels."""
    seen_urls = {primary.canonical_url or primary.url}
    used_labels: set[str] = set()
    raw_pairs: list[tuple[str, str, Article]] = []

    for ra in related_articles:
        u = ra.canonical_url or ra.url
        if not ra.url or u in seen_urls:
            continue
        seen_urls.add(u)
        raw_pairs.append((ra.source_name, ra.url, ra))
        if len(raw_pairs) >= max_links:
            break

    # Count source_name occurrences so duplicates (e.g. 2 Simon Willison posts) each show specific titles
    source_counts: dict[str, int] = {}
    for sname, _, _ in raw_pairs:
        source_counts[sname] = source_counts.get(sname, 0) + 1

    result: list[tuple[str, str]] = []
    for sname, url, ra in raw_pairs:
        short_title = ra.title.split("]", 1)[-1].strip()
        if len(short_title) > 32:
            short_title = short_title[:31].rstrip() + "…"
        if ra.source_id == primary.source_id or sname == primary.source_name:
            label = short_title or sname
        elif source_counts.get(sname, 0) > 1:
            short_source = sname.split("（", 1)[0].split("(", 1)[0].strip()
            label = f"{short_source}: {short_title}" if short_title else sname
        else:
            label = sname
        if label in used_labels:
            label = f"{label} ({len(result) + 1})"
        used_labels.add(label)
        result.append((label, url))
    return result


@dataclass
class EmergingSignal:
    keyword: str
    track: str
    reason: str


@dataclass
class CurateResult:
    headline_summary: list[str]
    selected: list[CuratedItem]
    emerging_signals: list[EmergingSignal]
    model_used: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    used_fallback_rule_mode: bool = False
    degrade_reason: str = ""


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelProbeResult:
    model_id: str
    ok: bool
    latency_ms: int = 0
    error: str = ""
    version_key: tuple[Any, ...] = field(default_factory=tuple)
