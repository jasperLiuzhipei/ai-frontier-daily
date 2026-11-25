"""URL/title deduplication, freshness filtering, and heuristic pre-scoring."""

from __future__ import annotations

import hashlib
import re
import urllib.parse
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any

from ai_daily.history import article_matches_history
from ai_daily.models import Article

_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "ref",
    "ref_src",
    "source",
    "fbclid",
    "gclid",
    "spm",
    "from",
    "isappinstalled",
}

_NON_ALNUM_RE = re.compile(r"[^\w\u4e00-\u9fff]+")
_SNAPSHOT_MMDD_RE = re.compile(
    r"(?:flash|pro|preview|exp|v\d+(?:\.\d+)?)[-_](\d{2})(\d{2})(?:\b|[^0-9])",
    re.IGNORECASE,
)
_FULL_DATE_RE = re.compile(r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_URL_PATH_DATE_RE = re.compile(r"/(20\d{2})/(0[1-9]|1[0-2])(?:/(0[1-9]|[12]\d|3[01]))?(?:/|$)")
_ARXIV_ID_RE = re.compile(r"(?:arxiv(?:\.org/(?:abs|pdf)/|[:\s]+))(\d{2})(0[1-9]|1[0-2])\.\d{4,5}", re.IGNORECASE)


def infer_embedded_release_date(text: str, ref_now: datetime) -> datetime | None:
    """Detect explicit release dates or -MMDD model snapshot stamps (e.g. Flash-1017)."""
    full_m = _FULL_DATE_RE.search(text)
    if full_m:
        try:
            y, m, d = int(full_m.group(1)), int(full_m.group(2)), int(full_m.group(3))
            dt = datetime(y, m, d, tzinfo=timezone.utc)
            if dt <= ref_now + timedelta(days=2):
                return dt
        except ValueError:
            pass

    mmdd_m = _SNAPSHOT_MMDD_RE.search(text)
    if mmdd_m:
        mm, dd = int(mmdd_m.group(1)), int(mmdd_m.group(2))
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            year = ref_now.year
            try:
                candidate = datetime(year, mm, dd, tzinfo=timezone.utc)
                if candidate > ref_now + timedelta(days=30):
                    candidate = datetime(year - 1, mm, dd, tzinfo=timezone.utc)
                return candidate
            except ValueError:
                pass
    return None


def infer_origin_date(url: str, text: str, ref_now: datetime) -> tuple[datetime | None, str]:
    """Penetrate repost/tweet wrappers to find true event origin dates (URL path / arXiv YYMM)."""
    url_m = _URL_PATH_DATE_RE.search(url or "")
    if url_m:
        y, m = int(url_m.group(1)), int(url_m.group(2))
        d = int(url_m.group(3)) if url_m.group(3) else 28
        try:
            dt = datetime(y, m, min(d, 28 if not url_m.group(3) else d), tzinfo=timezone.utc)
            if dt <= ref_now + timedelta(days=2):
                return (dt, f"url:{y:04d}-{m:02d}" + (f"-{int(url_m.group(3)):02d}" if url_m.group(3) else ""))
        except ValueError:
            pass

    arxiv_m = _ARXIV_ID_RE.search(f"{url} {text}")
    if arxiv_m:
        yy, mm = int(arxiv_m.group(1)), int(arxiv_m.group(2))
        year = 2000 + yy
        try:
            # Conservative end-of-month estimate for arXiv YYMM so only genuinely older months are flagged
            next_month = datetime(year + (1 if mm == 12 else 0), 1 if mm == 12 else mm + 1, 1, tzinfo=timezone.utc)
            end_of_month = next_month - timedelta(days=1)
            if end_of_month < ref_now:
                return (end_of_month, f"arxiv:{year:04d}-{mm:02d}")
        except ValueError:
            pass

    return (None, "")


def canonicalize_url(url: str, *, keep_fragment: bool = False) -> str:
    parsed = urllib.parse.urlsplit(url.strip())
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/") or "/"
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=False)
    filtered_q = [
        (k, v)
        for k, v in query_pairs
        if k.lower() not in _TRACKING_PARAMS and not k.lower().startswith("utm_")
    ]
    filtered_q.sort()
    query = urllib.parse.urlencode(filtered_q)
    fragment = parsed.fragment if keep_fragment else ""
    return urllib.parse.urlunsplit((scheme, netloc, path, query, fragment))


def hash_url(canonical_url: str) -> str:
    return hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:24]


def normalize_title(title: str) -> str:
    return _NON_ALNUM_RE.sub("", title.lower()).strip()


def _titles_too_similar(norm_a: str, norm_b: str) -> bool:
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True
    if min(len(norm_a), len(norm_b)) < 10:
        return False
    return SequenceMatcher(None, norm_a, norm_b).ratio() >= 0.86


def _match_keywords(text_low: str, keywords: list[str]) -> list[str]:
    hits: list[str] = []
    for kw in keywords:
        k = kw.strip().lower()
        if not k:
            continue
        if k in text_low:
            hits.append(kw)
    return hits


def pre_score_article(
    article: Article,
    profile: dict[str, Any],
    source_weights: dict[str, float],
    source_hit_rates: dict[str, float],
) -> tuple[float, list[str], bool]:
    """Compute a fast local relevance score before sending candidates to the LLM."""
    # Match ONLY against actual article content (never source-inherited tags) to prevent
    # targeted news queries from getting artificial boost scores on irrelevant hits.
    text_low = f"{article.title} {article.summary}".lower()

    exclude_hits = _match_keywords(text_low, profile.get("exclude_keywords", []))
    if exclude_hits:
        return (-10.0, [], True)

    src_weight = source_weights.get(article.source_id, 1.0)
    score = 2.0 * src_weight
    matched: list[str] = []

    best_track_bonus = 0.0
    for track in profile.get("tracks", []):
        hits = _match_keywords(text_low, track.get("keywords", []))
        if hits:
            matched.extend(hits)
            t_weight = float(track.get("weight", 1.0))
            bonus = min(len(hits), 3) * 0.9 * t_weight
            best_track_bonus = max(best_track_bonus, bonus)
    score += best_track_bonus

    # Keep keyword boost moderate so high-tech / fun general AI stories aren't drowned out
    boost_hits = _match_keywords(text_low, profile.get("boost_keywords", []))
    if boost_hits:
        matched.extend(boost_hits)
        score += min(len(boost_hits), 2) * 1.25

    # Strong independent signal boost for community heat, geek builds, and curated portals
    points = int(article.metrics.get("points") or 0)
    upvotes = int(article.metrics.get("upvotes") or 0)
    stars = int(article.metrics.get("stars") or 0)
    aihot_score = float(article.metrics.get("aihot_score") or 0.0)
    trending_score = float(article.metrics.get("trending_score") or 0.0)
    if points >= 200:
        score += 4.0
    elif points >= 80:
        score += 2.6
    elif points >= 40:
        score += 1.4
    if upvotes >= 50:
        score += 3.0
    elif upvotes >= 20:
        score += 1.8
    if trending_score >= 20:
        score += 2.2
    if stars >= 500:
        score += 2.0
    elif stars >= 100:
        score += 1.0
    if aihot_score >= 70.0 or (8.0 <= aihot_score <= 10.0):
        score += 2.8
    elif aihot_score >= 55.0 or (6.5 <= aihot_score < 8.0):
        score += 1.5
    if article.title.lower().startswith("show hn:"):
        score += 1.4

    # Historical hit-rate bonus from self-evolution telemetry
    hit_rate = source_hit_rates.get(article.source_id, 0.0)
    if hit_rate > 0:
        score += min(hit_rate * 2.0, 1.2)

    # Slight freshness bonus for last 12h
    if article.published_at is not None:
        age_hours = (datetime.now(timezone.utc) - article.published_at).total_seconds() / 3600.0
        if 0 <= age_hours <= 12:
            score += 0.6

    dedup_matched = list(dict.fromkeys(matched))
    return (round(score, 2), dedup_matched, False)


def prepare_candidates(
    articles: list[Article],
    *,
    profile: dict[str, Any],
    sources: list[dict[str, Any]],
    pushed_hashes: set[str],
    pushed_norm_titles: set[str],
    history_lookup: tuple[set[str], set[str], set[tuple[str, str]], dict[str, str]] | None = None,
    source_hit_rates: dict[str, float] | None = None,
    window_hours: int | None = None,
    max_candidates: int | None = None,
) -> tuple[list[Article], dict[str, int]]:
    """Filter by freshness, origin-date penetration, and 7d event history, then pre-score."""
    hours = window_hours if window_hours is not None else int(profile.get("window_hours", 36))
    hard_max_hours = max(hours, int(profile.get("hard_max_age_hours", 72)))
    origin_max_days = int(profile.get("origin_max_age_days", 14))
    limit = (
        max_candidates
        if max_candidates is not None
        else int(profile.get("max_candidates_for_llm", 65))
    )
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(hours=hours)
    hard_cutoff = now_utc - timedelta(hours=hard_max_hours)
    origin_cutoff = now_utc - timedelta(days=origin_max_days)

    source_map = {s["id"]: s for s in sources}
    source_weights = {s["id"]: float(s.get("weight", 1.0)) for s in sources}
    hit_rates = source_hit_rates or {}
    hist_urls, hist_repos, hist_ev, repo_notes = history_lookup or (set(), set(), set(), {})

    seen_batch_hashes: set[str] = set()
    seen_batch_titles: list[str] = []
    undated_per_source: dict[str, int] = {}
    new_counts: dict[str, int] = {}
    filtered: list[Article] = []

    for art in articles:
        src = source_map.get(art.source_id, {})
        keep_frag = src.get("type") == "web_watch" and src.get("mode") == "changelog"
        art.canonical_url = canonicalize_url(art.url, keep_fragment=keep_frag)
        art.url_hash = hash_url(art.canonical_url)
        norm_title = normalize_title(art.title)

        # Freshness check 1: explicit published_at
        if art.published_at is not None and art.published_at < cutoff:
            continue

        # Freshness check 2: infer embedded release/snapshot date stamps (e.g. Flash-1017)
        if art.published_at is None:
            inferred = infer_embedded_release_date(f"{art.title} {art.summary[:160]}", now_utc)
            if inferred is not None:
                art.published_at = inferred
                if inferred < hard_cutoff:
                    continue
            elif src.get("type") == "web_watch":
                undated_seen = undated_per_source.get(art.source_id, 0)
                if undated_seen >= 2:
                    continue
                undated_per_source[art.source_id] = undated_seen + 1

        # Freshness check 3: origin-date penetration (URL /2026/08/ path or old arXiv YYMM)
        origin_dt, origin_tag = infer_origin_date(art.url, f"{art.title} {art.summary}", now_utc)
        if origin_dt is not None and origin_dt < origin_cutoff:
            if origin_tag.startswith("url:"):
                # Old article URL from weeks/months ago resurfaced by search -> hard drop
                continue
            # Old arXiv paper reshared on X/blogs -> drop if >21 days old, else flag for LLM
            if origin_dt < now_utc - timedelta(days=21):
                continue
            art.metrics["origin_date_warning"] = f"底层论文/原始来源时间 {origin_tag}"

        # Already pushed in SQLite or 7-day Git sent_history.json
        if art.url_hash in pushed_hashes or norm_title in pushed_norm_titles:
            continue

        hard_dup, soft_hist_note = article_matches_history(
            art,
            hist_urls=hist_urls,
            hist_repos=hist_repos,
            hist_entity_versions=hist_ev,
            repo_notes=repo_notes,
        )
        if hard_dup:
            continue
        if soft_hist_note:
            art.metrics["history_hit"] = soft_hist_note

        # Intra-batch deduplication
        if art.url_hash in seen_batch_hashes:
            continue
        if any(_titles_too_similar(norm_title, existing) for existing in seen_batch_titles):
            continue

        score, matched, excluded = pre_score_article(art, profile, source_weights, hit_rates)
        if excluded:
            continue
        if soft_hist_note:
            score = round(max(0.5, score - 2.5), 2)

        seen_batch_hashes.add(art.url_hash)
        seen_batch_titles.append(norm_title)
        art.pre_score = score
        art.matched_keywords = matched
        new_counts[art.source_id] = new_counts.get(art.source_id, 0) + 1
        filtered.append(art)

    filtered.sort(key=lambda a: a.pre_score, reverse=True)

    if len(filtered) <= limit:
        return filtered, new_counts

    # Balanced selection: ensure each track/category gets reserved slots before filling globally
    by_cat: dict[str, list[Article]] = {}
    for art in filtered:
        by_cat.setdefault(art.category, []).append(art)

    chosen: list[Article] = []
    chosen_hashes: set[str] = set()
    per_cat_reserve = max(6, limit // max(len(by_cat), 1) // 2)

    for cat_items in by_cat.values():
        for art in cat_items[:per_cat_reserve]:
            if art.url_hash not in chosen_hashes:
                chosen.append(art)
                chosen_hashes.add(art.url_hash)

    for art in filtered:
        if len(chosen) >= limit:
            break
        if art.url_hash not in chosen_hashes:
            chosen.append(art)
            chosen_hashes.add(art.url_hash)

    chosen.sort(key=lambda a: a.pre_score, reverse=True)
    return chosen[:limit], new_counts
