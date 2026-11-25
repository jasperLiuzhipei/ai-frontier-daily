"""Persistent 7-14 day sent-event memory store committed to Git for cross-day deduplication."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ai_daily.models import Article, CuratedItem

_GITHUB_REPO_RE = re.compile(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", re.IGNORECASE)
_HF_REPO_RE = re.compile(r"huggingface\.co/(?:spaces/)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", re.IGNORECASE)
_PRODUCT_VERSION_RE = re.compile(
    r"\b(claude\s*code|codex|antigravity|browser[- ]use|vllm|sglang|deepseek[- ]?v?\d(?:\.\d)?|qwen\s*\d(?:\.\d)?|gemini\s*\d(?:\.\d)?|seedance|inkling)\b"
    r"[^\n]{0,24}?\b(v?\d+\.\d+(?:\.\d+)?(?:-[a-z0-9.]+)?)\b",
    re.IGNORECASE,
)


def extract_repo_slug(text: str) -> str:
    m = _GITHUB_REPO_RE.search(text or "")
    if m:
        slug = m.group(1).rstrip(".,;:)\"'/").lower()
        if not slug.startswith(("topics/", "orgs/", "search/")):
            return f"github.com/{slug}"
    m2 = _HF_REPO_RE.search(text or "")
    if m2:
        slug = m2.group(1).rstrip(".,;:)\"'/").lower()
        if not slug.startswith(("papers/", "blog/", "docs/")):
            return f"huggingface.co/{slug}"
    return ""


def extract_entity_version(text: str) -> tuple[str, str]:
    m = _PRODUCT_VERSION_RE.search(text or "")
    if not m:
        return ("", "")
    entity = re.sub(r"\s+", " ", m.group(1).strip().lower())
    ver = m.group(2).lstrip("vV").lower()
    return (entity, ver)


class SentHistoryStore:
    """Tracks sent stories across days in a Git-friendly JSON file (data/sent_history.json)."""

    def __init__(self, path: Path, retain_days: int = 14) -> None:
        self.path = path
        self.retain_days = retain_days

    def load(self, within_days: int | None = None) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []
        entries = data.get("items", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        if within_days is None:
            return [e for e in entries if isinstance(e, dict)]

        cutoff = (datetime.now(timezone.utc) - timedelta(days=within_days)).strftime("%Y-%m-%d")
        return [
            e for e in entries
            if isinstance(e, dict) and str(e.get("date") or "") >= cutoff
        ]

    def record_items(self, run_date: str, items: list[CuratedItem]) -> None:
        existing = self.load(within_days=self.retain_days)
        seen_keys = {
            (str(e.get("canonical_url") or ""), str(e.get("entity") or ""), str(e.get("version") or ""))
            for e in existing
        }

        for item in items:
            blob = f"{item.zh_title} {item.article.title} {item.article.url} {item.one_liner} {item.geek_action}"
            repo = extract_repo_slug(blob)
            entity, version = extract_entity_version(blob)
            if not entity and item.tags:
                entity = item.tags[0].lower()
            canon = item.article.canonical_url or item.article.url
            key = (canon, entity, version)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            existing.append(
                {
                    "date": run_date,
                    "canonical_url": canon,
                    "repo": repo,
                    "entity": entity,
                    "version": version,
                    "title": item.zh_title[:120],
                    "event_fingerprint": item.one_liner[:180],
                }
            )

        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retain_days)).strftime("%Y-%m-%d")
        pruned = [e for e in existing if str(e.get("date") or "") >= cutoff]
        pruned.sort(key=lambda e: str(e.get("date") or ""), reverse=True)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"schema_version": 1, "items": pruned[:200]}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def build_prompt_context(self, within_days: int = 7, max_items: int = 70) -> str:
        recent = self.load(within_days=within_days)[:max_items]
        if not recent:
            return "（过去 7 天暂无已发历史记录）"
        lines: list[str] = []
        for e in recent:
            dt = e.get("date", "")
            ident_parts = [p for p in (e.get("entity"), e.get("version"), e.get("repo")) if p]
            ident = " / ".join(ident_parts) if ident_parts else e.get("title", "")[:40]
            fp = e.get("event_fingerprint") or e.get("title") or ""
            lines.append(f"- {dt} | {ident} | {fp}")
        return "\n".join(lines)

    def build_lookup_sets(
        self, within_days: int = 7
    ) -> tuple[set[str], set[str], set[tuple[str, str]], dict[str, str]]:
        """Return (urls, repos, entity_versions, repo_to_note) for deterministic filtering."""
        recent = self.load(within_days=within_days)
        urls: set[str] = set()
        repos: set[str] = set()
        entity_versions: set[tuple[str, str]] = set()
        repo_notes: dict[str, str] = {}
        for e in recent:
            u = str(e.get("canonical_url") or "").strip()
            if u:
                urls.add(u)
            repo = str(e.get("repo") or "").strip().lower()
            if repo:
                repos.add(repo)
                repo_notes[repo] = f"{e.get('date', '')}: {e.get('title', '')[:60]}"
            ent = str(e.get("entity") or "").strip().lower()
            ver = str(e.get("version") or "").strip().lower()
            if ent and ver:
                entity_versions.add((ent, ver))
        return urls, repos, entity_versions, repo_notes


def article_matches_history(
    article: Article,
    *,
    hist_urls: set[str],
    hist_repos: set[str],
    hist_entity_versions: set[tuple[str, str]],
    repo_notes: dict[str, str],
) -> tuple[bool, str]:
    """Check if an article duplicates a recently sent story.

    Returns:
        (hard_drop, soft_history_note)
        - hard_drop=True: exact canonical URL or exact (product, version) already sent in last 7d.
        - soft_history_note: non-empty if the same GitHub/HF repo appeared in last 7d (warn LLM).
    """
    canon = article.canonical_url or article.url
    if canon in hist_urls:
        return (True, "canonical_url_dup")

    blob = f"{article.title} {article.summary} {article.url}"
    ent, ver = extract_entity_version(blob)
    if ent and ver and (ent, ver) in hist_entity_versions:
        return (True, f"entity_version_dup:{ent}@{ver}")

    repo = extract_repo_slug(blob)
    if repo and repo in hist_repos:
        # Exact GitHub repo root link already pushed -> hard drop; blog discussing it -> soft flag for LLM
        if canon.rstrip("/").lower().endswith(repo):
            return (True, f"repo_root_dup:{repo}")
        return (False, f"近7天已报道同项目 {repo}（{repo_notes.get(repo, '')}），无重大新变故请勿重复推送")

    return (False, "")
