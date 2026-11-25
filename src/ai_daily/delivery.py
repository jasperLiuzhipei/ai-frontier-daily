"""Persistent per-day draft artifacts and per-channel delivery receipt state store."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ai_daily.models import Article, CuratedItem, CurateResult, utc_now_iso
from ai_daily.notifiers.feishu import make_delivery_uuid


def current_local_now() -> datetime:
    """Return timezone-aware local time (respects AI_DAILY_TZ when set, else system/TZ)."""
    tz_name = os.environ.get("AI_DAILY_TZ", "").strip()
    if tz_name:
        try:
            return datetime.now(ZoneInfo(tz_name))
        except Exception:
            pass
    return datetime.now().astimezone()


def local_now_iso() -> str:
    return current_local_now().isoformat(timespec="seconds")


def compute_draft_sha256(
    *, title: str, markdown: str, feishu_card: dict[str, Any]
) -> str:
    payload = json.dumps(
        {"title": title, "markdown": markdown, "feishu_card": feishu_card},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _serialize_article(art: Article) -> dict[str, Any]:
    return {
        "source_id": art.source_id,
        "source_name": art.source_name,
        "category": art.category,
        "title": art.title,
        "url": art.url,
        "summary": art.summary,
        "canonical_url": art.canonical_url or art.url,
        "url_hash": art.url_hash,
        "tags": list(art.tags),
    }


def _deserialize_article(row: dict[str, Any]) -> Article:
    url = str(row.get("url") or "")
    canon = str(row.get("canonical_url") or url)
    url_hash = str(row.get("url_hash") or hashlib.sha256(canon.encode("utf-8")).hexdigest()[:24])
    return Article(
        source_id=str(row.get("source_id") or ""),
        source_name=str(row.get("source_name") or ""),
        category=str(row.get("category") or "geek"),
        title=str(row.get("title") or ""),
        url=url,
        summary=str(row.get("summary") or ""),
        canonical_url=canon,
        url_hash=url_hash,
        tags=[str(t) for t in (row.get("tags") or [])],
    )


def serialize_curated_items(items: list[CuratedItem]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        rows.append(
            {
                "candidate_id": item.candidate_id,
                "article": _serialize_article(item.article),
                "related_articles": [_serialize_article(ra) for ra in item.related_articles],
                "track": item.track,
                "score": item.score,
                "zh_title": item.zh_title,
                "one_liner": item.one_liner,
                "why_it_matters": item.why_it_matters,
                "tech_depth": item.tech_depth,
                "geek_action": item.geek_action,
                "section": item.section,
                "tech_score": item.tech_score,
                "fun_score": item.fun_score,
                "impact_score": item.impact_score,
                "tags": list(item.tags),
            }
        )
    return rows


def deserialize_curated_items(rows: list[dict[str, Any]]) -> list[CuratedItem]:
    items: list[CuratedItem] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("article"), dict):
            continue
        items.append(
            CuratedItem(
                candidate_id=int(row.get("candidate_id") or 0),
                article=_deserialize_article(row["article"]),
                related_articles=[
                    _deserialize_article(ra)
                    for ra in (row.get("related_articles") or [])
                    if isinstance(ra, dict)
                ],
                track=str(row.get("track") or "geek"),
                score=float(row.get("score") or 0.0),
                zh_title=str(row.get("zh_title") or ""),
                one_liner=str(row.get("one_liner") or ""),
                why_it_matters=str(row.get("why_it_matters") or ""),
                tech_depth=str(row.get("tech_depth") or ""),
                geek_action=str(row.get("geek_action") or ""),
                section=str(row.get("section") or ""),
                tech_score=float(row.get("tech_score") or 0.0),
                fun_score=float(row.get("fun_score") or 0.0),
                impact_score=float(row.get("impact_score") or 0.0),
                tags=[str(t) for t in (row.get("tags") or [])],
            )
        )
    return items


def build_draft_payload(
    *,
    run_id: str,
    run_date: str,
    curate_res: CurateResult,
    title: str,
    short_preview: str,
    markdown: str,
    feishu_card: dict[str, Any],
    raw_count: int,
    candidate_count: int,
    enriched_count: int,
    sources_ok: int,
    sources_total: int,
    failed_sources: list[dict[str, Any]],
    evolution_notes: list[str],
) -> dict[str, Any]:
    sha = compute_draft_sha256(title=title, markdown=markdown, feishu_card=feishu_card)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "run_date": run_date,
        "created_at": utc_now_iso(),
        "created_at_local": local_now_iso(),
        "draft_sha256": sha,
        "title": title,
        "short_preview": short_preview,
        "markdown": markdown,
        "feishu_card": feishu_card,
        "model_used": curate_res.model_used,
        "prompt_tokens": curate_res.prompt_tokens,
        "completion_tokens": curate_res.completion_tokens,
        "degraded": bool(curate_res.used_fallback_rule_mode and curate_res.degrade_reason),
        "degrade_reason": curate_res.degrade_reason,
        "raw_count": raw_count,
        "candidate_count": candidate_count,
        "enriched_count": enriched_count,
        "selected_count": len(curate_res.selected),
        "sources_ok": sources_ok,
        "sources_total": sources_total,
        "failed_sources": failed_sources,
        "emerging_signals": [
            {"keyword": s.keyword, "track": s.track, "reason": s.reason}
            for s in curate_res.emerging_signals
        ],
        "evolution_notes": list(evolution_notes),
        "selected_records": serialize_curated_items(curate_res.selected),
    }


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


class DeliveryStore:
    """Stores immutable daily drafts and per-channel delivery receipts under a directory."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir

    def day_dir(self, run_date: str) -> Path:
        return self.state_dir / run_date

    def draft_path(self, run_date: str) -> Path:
        return self.day_dir(run_date) / "draft.json"

    def receipt_path(self, run_date: str) -> Path:
        return self.day_dir(run_date) / "receipt.json"

    def load_draft(self, run_date: str) -> dict[str, Any] | None:
        path = self.draft_path(run_date)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def save_draft(self, run_date: str, draft: dict[str, Any]) -> Path:
        path = self.draft_path(run_date)
        _write_json_atomic(path, draft)
        return path

    def load_receipt(self, run_date: str) -> dict[str, Any] | None:
        path = self.receipt_path(run_date)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def save_receipt(self, run_date: str, receipt: dict[str, Any]) -> Path:
        receipt["updated_at"] = utc_now_iso()
        receipt["updated_at_local"] = local_now_iso()
        path = self.receipt_path(run_date)
        _write_json_atomic(path, receipt)
        return path

    def ensure_receipt_for_draft(
        self,
        run_date: str,
        draft: dict[str, Any],
        targets: list[Any],
        *,
        force: bool = False,
        force_nonce: str = "",
    ) -> dict[str, Any]:
        """Initialize or refresh receipt.json for a fixed draft, preserving already-sent channels unless forced."""
        existing = self.load_receipt(run_date) or {}
        old_channels = existing.get("channels") if isinstance(existing.get("channels"), dict) else {}
        draft_sha = str(draft.get("draft_sha256") or "")
        same_draft = (existing.get("draft_sha256") == draft_sha) and not force

        channels: dict[str, Any] = dict(old_channels) if same_draft else {}
        for target in targets:
            ch_name = getattr(target, "name", "notifier")
            mode = getattr(target, "delivery_mode", ch_name)
            prev = channels.get(ch_name) if isinstance(channels.get(ch_name), dict) else {}
            if same_draft and prev.get("status") == "sent":
                continue
            idem_key = (
                str(prev.get("idempotency_key") or "")
                if same_draft and prev.get("idempotency_key")
                else make_delivery_uuid(run_date, ch_name, draft_sha, nonce=force_nonce)
            )
            channels[ch_name] = {
                "status": prev.get("status") if (same_draft and prev.get("status") in ("pending", "failed", "unknown")) else "pending",
                "mode": mode,
                "idempotency_key": idem_key,
                "attempts": int(prev.get("attempts") or 0) if same_draft else 0,
                "message_id": str(prev.get("message_id") or "") if same_draft else "",
                "last_error": str(prev.get("last_error") or "") if same_draft else "",
                "sent_at": str(prev.get("sent_at") or "") if same_draft else "",
                "sent_at_local": str(prev.get("sent_at_local") or "") if same_draft else "",
                "updated_at": utc_now_iso(),
            }

        receipt = {
            "schema_version": 1,
            "run_date": run_date,
            "draft_sha256": draft_sha,
            "degraded": bool(draft.get("degraded")),
            "degrade_reason": str(draft.get("degrade_reason") or ""),
            "model_used": str(draft.get("model_used") or ""),
            "selected_count": int(draft.get("selected_count") or 0),
            "created_at": existing.get("created_at") if same_draft and existing.get("created_at") else utc_now_iso(),
            "delivered_at_local": existing.get("delivered_at_local", "") if same_draft else "",
            "fetch_error": "",
            "llm_error": "",
            "history_committed": bool(existing.get("history_committed")) if same_draft else False,
            "archive_committed": bool(existing.get("archive_committed")) if same_draft else False,
            "channels": channels,
        }
        self.save_receipt(run_date, receipt)
        return receipt

    def record_stage_failure(
        self,
        run_date: str,
        *,
        kind: str,
        error: str,
    ) -> dict[str, Any]:
        """Persist fetch_failed or llm_failed status without writing a draft."""
        receipt = self.load_receipt(run_date) or {
            "schema_version": 1,
            "run_date": run_date,
            "draft_sha256": "",
            "degraded": False,
            "degrade_reason": "",
            "model_used": "",
            "selected_count": 0,
            "created_at": utc_now_iso(),
            "delivered_at_local": "",
            "fetch_error": "",
            "llm_error": "",
            "history_committed": False,
            "archive_committed": False,
            "channels": {},
        }
        if kind == "fetch_failed":
            receipt["fetch_error"] = error[:400]
            receipt["fetch_attempts"] = int(receipt.get("fetch_attempts") or 0) + 1
        elif kind == "llm_failed":
            receipt["llm_error"] = error[:400]
            receipt["llm_attempts"] = int(receipt.get("llm_attempts") or 0) + 1
        self.save_receipt(run_date, receipt)
        return receipt

    @staticmethod
    def all_channels_sent(receipt: dict[str, Any] | None, target_names: list[str]) -> bool:
        if not receipt or not target_names:
            return False
        channels = receipt.get("channels") or {}
        if not isinstance(channels, dict):
            return False
        return all(
            isinstance(channels.get(name), dict) and channels[name].get("status") == "sent"
            for name in target_names
        )

    def summarize_metrics(self, days: int = 14) -> dict[str, Any]:
        """Summarize recent delivery reliability metrics for unattended soak evaluation."""
        if not self.state_dir.is_dir():
            return {"days_checked": 0, "delivered": 0, "failed_or_pending": 0, "degraded": 0, "retried": 0, "rows": []}

        receipt_files = sorted(self.state_dir.glob("*/receipt.json"), reverse=True)[: max(days, 1)]
        rows: list[dict[str, Any]] = []
        delivered = 0
        failed_or_pending = 0
        degraded = 0
        retried = 0

        for path in receipt_files:
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            channels = rec.get("channels") or {}
            ch_statuses = {k: v.get("status", "unknown") for k, v in channels.items() if isinstance(v, dict)}
            max_attempts = max((int(v.get("attempts") or 0) for v in channels.values() if isinstance(v, dict)), default=0)
            is_sent = bool(ch_statuses) and all(st == "sent" for st in ch_statuses.values())
            if is_sent:
                delivered += 1
            else:
                failed_or_pending += 1
            if rec.get("degraded"):
                degraded += 1
            if max_attempts > 1:
                retried += 1
            rows.append(
                {
                    "run_date": rec.get("run_date") or path.parent.name,
                    "sent": is_sent,
                    "delivered_at_local": rec.get("delivered_at_local") or "",
                    "model_used": rec.get("model_used") or "",
                    "degraded": bool(rec.get("degraded")),
                    "attempts": max_attempts,
                    "channels": ch_statuses,
                    "fetch_error": rec.get("fetch_error") or "",
                    "llm_error": rec.get("llm_error") or "",
                }
            )

        return {
            "days_checked": len(rows),
            "delivered": delivered,
            "failed_or_pending": failed_or_pending,
            "degraded": degraded,
            "retried": retried,
            "rows": rows,
        }
