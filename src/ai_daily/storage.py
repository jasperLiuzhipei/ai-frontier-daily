"""SQLite state store for deduplication, source health metrics, feedback, and self-evolution."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ai_daily.models import Article, CuratedItem, EmergingSignal, SourceFetchResult, utc_now_iso

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS seen_articles (
    url_hash TEXT PRIMARY KEY,
    canonical_url TEXT NOT NULL,
    norm_title TEXT NOT NULL,
    title TEXT NOT NULL,
    source_id TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    pushed_at TEXT,
    llm_score REAL,
    track TEXT
);

CREATE INDEX IF NOT EXISTS idx_seen_norm_title ON seen_articles(norm_title);
CREATE INDEX IF NOT EXISTS idx_seen_pushed_at ON seen_articles(pushed_at);

CREATE TABLE IF NOT EXISTS digest_runs (
    run_id TEXT PRIMARY KEY,
    run_date TEXT NOT NULL,
    model_used TEXT,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    raw_count INTEGER DEFAULT 0,
    candidate_count INTEGER DEFAULT 0,
    selected_count INTEGER DEFAULT 0,
    output_path TEXT,
    pushed INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    status TEXT NOT NULL,
    fetched_count INTEGER DEFAULT 0,
    new_count INTEGER DEFAULT 0,
    selected_count INTEGER DEFAULT 0,
    avg_score REAL DEFAULT 0.0,
    elapsed_ms INTEGER DEFAULT 0,
    error_message TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_source_runs_source_id ON source_runs(source_id, created_at DESC);

CREATE TABLE IF NOT EXISTS emerging_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT NOT NULL,
    track TEXT NOT NULL,
    reason TEXT DEFAULT '',
    seen_count INTEGER DEFAULT 1,
    last_seen_at TEXT NOT NULL,
    applied INTEGER DEFAULT 0,
    UNIQUE(keyword, track)
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    target TEXT NOT NULL,
    detail TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    applied INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS evolution_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    summary TEXT NOT NULL,
    backup_path TEXT DEFAULT '',
    diff_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);
"""


class StateStore:
    """Persistent SQLite storage for deduplication and autonomous evolution telemetry."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    def get_pushed_hashes_and_titles(self, limit: int = 2000) -> tuple[set[str], set[str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT url_hash, norm_title
                FROM seen_articles
                WHERE pushed_at IS NOT NULL
                ORDER BY pushed_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return (
            {row["url_hash"] for row in rows},
            {row["norm_title"] for row in rows if row["norm_title"]},
        )

    def record_seen_articles(
        self, articles: list[Article], norm_title_fn: Any
    ) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO seen_articles
                    (url_hash, canonical_url, norm_title, title, source_id, first_seen_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        a.url_hash,
                        a.canonical_url or a.url,
                        norm_title_fn(a.title),
                        a.title,
                        a.source_id,
                        now,
                    )
                    for a in articles
                ],
            )

    def mark_curated_as_pushed(
        self, items: list[CuratedItem], norm_title_fn: Any
    ) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            for item in items:
                all_articles = [item.article, *item.related_articles]
                for idx, art in enumerate(all_articles):
                    conn.execute(
                        """
                        INSERT INTO seen_articles
                            (url_hash, canonical_url, norm_title, title, source_id,
                             first_seen_at, pushed_at, llm_score, track)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(url_hash) DO UPDATE SET
                            pushed_at = excluded.pushed_at,
                            llm_score = CASE WHEN ? = 0 THEN excluded.llm_score ELSE seen_articles.llm_score END,
                            track = excluded.track
                        """,
                        (
                            art.url_hash,
                            art.canonical_url or art.url,
                            norm_title_fn(art.title),
                            art.title,
                            art.source_id,
                            now,
                            now,
                            item.score,
                            item.track,
                            idx,
                        ),
                    )

    def record_digest_run(
        self,
        *,
        run_id: str,
        run_date: str,
        model_used: str,
        prompt_tokens: int,
        completion_tokens: int,
        raw_count: int,
        candidate_count: int,
        selected_count: int,
        output_path: str,
        pushed: bool,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO digest_runs
                    (run_id, run_date, model_used, prompt_tokens, completion_tokens,
                     raw_count, candidate_count, selected_count, output_path, pushed, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    run_date,
                    model_used,
                    prompt_tokens,
                    completion_tokens,
                    raw_count,
                    candidate_count,
                    selected_count,
                    output_path,
                    1 if pushed else 0,
                    utc_now_iso(),
                ),
            )

    def record_source_runs(
        self,
        run_id: str,
        fetch_results: list[SourceFetchResult],
        new_counts: dict[str, int],
        selected_items: list[CuratedItem],
    ) -> None:
        selected_by_source: dict[str, list[float]] = {}
        for item in selected_items:
            selected_by_source.setdefault(item.article.source_id, []).append(item.score)

        now = utc_now_iso()
        with self._connect() as conn:
            for res in fetch_results:
                scores = selected_by_source.get(res.source_id, [])
                avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
                conn.execute(
                    """
                    INSERT INTO source_runs
                        (run_id, source_id, status, fetched_count, new_count,
                         selected_count, avg_score, elapsed_ms, error_message, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        res.source_id,
                        "ok" if res.ok else "error",
                        len(res.articles),
                        new_counts.get(res.source_id, 0),
                        len(scores),
                        avg_score,
                        res.elapsed_ms,
                        res.error[:500],
                        now,
                    ),
                )

    def record_emerging_signals(self, signals: list[EmergingSignal]) -> None:
        if not signals:
            return
        now = utc_now_iso()
        with self._connect() as conn:
            for sig in signals:
                kw = sig.keyword.strip()
                if not kw:
                    continue
                conn.execute(
                    """
                    INSERT INTO emerging_signals (keyword, track, reason, seen_count, last_seen_at, applied)
                    VALUES (?, ?, ?, 1, ?, 0)
                    ON CONFLICT(keyword, track) DO UPDATE SET
                        seen_count = emerging_signals.seen_count + 1,
                        reason = excluded.reason,
                        last_seen_at = excluded.last_seen_at
                    """,
                    (kw, sig.track or "geek", sig.reason[:300], now),
                )

    def get_pending_emerging_signals(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, keyword, track, reason, seen_count, last_seen_at
                FROM emerging_signals
                WHERE applied = 0
                ORDER BY seen_count DESC, last_seen_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_signals_applied(self, keywords: list[str]) -> None:
        if not keywords:
            return
        lowered = [k.lower() for k in keywords]
        with self._connect() as conn:
            for kw in lowered:
                conn.execute(
                    "UPDATE emerging_signals SET applied = 1 WHERE lower(keyword) = ?",
                    (kw,),
                )

    def add_feedback(self, kind: str, target: str, detail: str = "") -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO feedback (kind, target, detail, created_at, applied)
                VALUES (?, ?, ?, ?, 0)
                """,
                (kind, target.strip(), detail.strip(), utc_now_iso()),
            )
            return int(cur.lastrowid or 0)

    def get_feedback(self, only_pending: bool = True, limit: int = 30) -> list[dict[str, Any]]:
        where = "WHERE applied = 0" if only_pending else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, kind, target, detail, created_at, applied
                FROM feedback
                {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_feedback_applied(self, ids: list[int]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE feedback SET applied = 1 WHERE id IN ({placeholders})",
                ids,
            )

    def get_source_stats(self, lookback_runs: int = 20) -> list[dict[str, Any]]:
        """Aggregate health, consecutive failures, and digest hit-rate per source."""
        with self._connect() as conn:
            source_ids = [
                r["source_id"]
                for r in conn.execute("SELECT DISTINCT source_id FROM source_runs").fetchall()
            ]
            stats: list[dict[str, Any]] = []
            for sid in source_ids:
                rows = conn.execute(
                    """
                    SELECT status, fetched_count, new_count, selected_count, avg_score, error_message, created_at
                    FROM source_runs
                    WHERE source_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (sid, lookback_runs),
                ).fetchall()
                if not rows:
                    continue
                total = len(rows)
                ok_runs = sum(1 for r in rows if r["status"] == "ok")
                consecutive_failures = 0
                for r in rows:
                    if r["status"] != "ok":
                        consecutive_failures += 1
                    else:
                        break
                total_fetched = sum(r["fetched_count"] for r in rows)
                total_new = sum(r["new_count"] for r in rows)
                total_selected = sum(r["selected_count"] for r in rows)
                score_rows = [r["avg_score"] for r in rows if r["selected_count"] > 0]
                mean_score = round(sum(score_rows) / len(score_rows), 2) if score_rows else 0.0
                hit_rate = round(total_selected / max(total_new, 1), 3)
                stats.append(
                    {
                        "source_id": sid,
                        "runs": total,
                        "ok_runs": ok_runs,
                        "success_rate": round(ok_runs / total, 3),
                        "consecutive_failures": consecutive_failures,
                        "total_fetched": total_fetched,
                        "total_new": total_new,
                        "total_selected": total_selected,
                        "hit_rate": hit_rate,
                        "mean_selected_score": mean_score,
                        "last_error": rows[0]["error_message"] if rows[0]["status"] != "ok" else "",
                    }
                )
        stats.sort(key=lambda s: (s["total_selected"], s["mean_selected_score"]), reverse=True)
        return stats

    def get_total_digest_runs(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM digest_runs").fetchone()
        return int(row["cnt"]) if row else 0

    def record_evolution_log(
        self,
        kind: str,
        summary: str,
        backup_path: str = "",
        diff_obj: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evolution_logs (kind, summary, backup_path, diff_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    summary,
                    backup_path,
                    json.dumps(diff_obj or {}, ensure_ascii=False),
                    utc_now_iso(),
                ),
            )

    def get_recent_evolution_logs(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, kind, summary, backup_path, diff_json, created_at
                FROM evolution_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
