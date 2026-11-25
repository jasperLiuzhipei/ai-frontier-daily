"""End-to-end daily briefing runner with persistent drafts, delivery receipts, and safe retries."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ai_daily.config import ConfigManager
from ai_daily.delivery import (
    DeliveryStore,
    build_draft_payload,
    current_local_now,
    deserialize_curated_items,
    local_now_iso,
)
from ai_daily.fetchers import fetch_all_sources
from ai_daily.history import SentHistoryStore
from ai_daily.http_client import HTTPClient
from ai_daily.llm.client import OpenAICompatClient
from ai_daily.llm.registry import ModelRegistry
from ai_daily.models import utc_now_iso
from ai_daily.notifiers import (
    DeliveryUnknownError,
    FeishuAppNotifier,
    FeishuNotifier,
    build_digest_card,
    resolve_notifiers,
)
from ai_daily.notifiers.feishu import _trim_card_to_budget
from ai_daily.pipeline.enrich import enrich_candidates
from ai_daily.pipeline.filter import infer_origin_date, normalize_title, prepare_candidates
from ai_daily.pipeline.generator import (
    LLMCurationError,
    curate_candidates,
    render_digest_markdown,
)
from ai_daily.storage import StateStore


def build_llm_client(config_mgr: ConfigManager, http: HTTPClient | None = None) -> OpenAICompatClient:
    llm_cfg = config_mgr.load_llm()
    api_key = config_mgr.resolve_api_key(llm_cfg)
    auto_cfg = llm_cfg.get("auto_iteration") or {}
    return OpenAICompatClient(
        base_url=llm_cfg["base_url"],
        api_key=api_key,
        primary_model=llm_cfg["primary_model"],
        fallback_models=llm_cfg.get("fallback_models") or [],
        temperature=float(llm_cfg.get("temperature", 0.2)),
        max_tokens=llm_cfg.get("max_tokens"),
        extra_body=llm_cfg.get("extra_body") or {},
        timeout_seconds=int(llm_cfg.get("timeout_seconds", 120)),
        max_retries=int(llm_cfg.get("max_retries", 2)),
        auto_discover_on_error=bool(auto_cfg.get("auto_discover_on_error", True)),
        http=http,
    )


def _is_feishu_notifier(target: Any) -> bool:
    return isinstance(target, (FeishuAppNotifier, FeishuNotifier))


def _base_summary(
    *,
    run_id: str,
    run_date: str,
    stage: str,
    output_path: Path,
    state_dir: Path,
) -> dict[str, Any]:
    return {
        "ok": False,
        "stage": stage,
        "run_id": run_id,
        "run_date": run_date,
        "skipped": False,
        "already_sent": False,
        "reused_draft": False,
        "degraded": False,
        "degrade_reason": "",
        "delivery_status": "pending",
        "sources_total": 0,
        "sources_ok": 0,
        "failed_sources": [],
        "raw_count": 0,
        "candidate_count": 0,
        "enriched_count": 0,
        "selected_count": 0,
        "model_used": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "output_path": str(output_path),
        "state_dir": str(state_dir),
        "title": "",
        "pushed": False,
        "pushed_channels": [],
        "channel_statuses": {},
        "push_error": "",
        "emerging_signals": [],
        "evolution_summary": None,
    }


def _populate_summary_from_draft(summary: dict[str, Any], draft: dict[str, Any]) -> None:
    summary["sources_total"] = int(draft.get("sources_total") or 0)
    summary["sources_ok"] = int(draft.get("sources_ok") or 0)
    summary["failed_sources"] = list(draft.get("failed_sources") or [])
    summary["raw_count"] = int(draft.get("raw_count") or 0)
    summary["candidate_count"] = int(draft.get("candidate_count") or 0)
    summary["enriched_count"] = int(draft.get("enriched_count") or 0)
    summary["selected_count"] = int(draft.get("selected_count") or 0)
    summary["model_used"] = str(draft.get("model_used") or "")
    summary["prompt_tokens"] = int(draft.get("prompt_tokens") or 0)
    summary["completion_tokens"] = int(draft.get("completion_tokens") or 0)
    summary["title"] = str(draft.get("title") or "")
    summary["degraded"] = bool(draft.get("degraded"))
    summary["degrade_reason"] = str(draft.get("degrade_reason") or "")
    summary["emerging_signals"] = [
        str(s.get("keyword") or "")
        for s in (draft.get("emerging_signals") or [])
        if isinstance(s, dict) and s.get("keyword")
    ]


def _resolve_targets(
    *,
    notifier: Any | None,
    notifiers: list[Any] | None,
    notify_channel: str | None,
    http_client: HTTPClient,
) -> list[Any]:
    if notifiers is not None:
        return list(notifiers)
    if notifier is not None:
        return [notifier]
    return resolve_notifiers(channel=notify_channel, http=http_client)


def _commit_confirmed_delivery(
    *,
    config_mgr: ConfigManager,
    store: StateStore,
    history_store: SentHistoryStore,
    delivery_store: DeliveryStore,
    run_id: str,
    run_date: str,
    output_path: Path,
    draft: dict[str, Any],
    receipt: dict[str, Any],
    write_archive: bool,
) -> None:
    """Persist archive Markdown and sent-history dedup records ONLY after delivery is confirmed."""
    if not receipt.get("history_committed"):
        curated_items = deserialize_curated_items(draft.get("selected_records") or [])
        if curated_items:
            store.mark_curated_as_pushed(curated_items, normalize_title)
            history_store.record_items(run_date, curated_items)
        receipt["history_committed"] = True

    if write_archive:
        config_mgr.output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(str(draft.get("markdown") or ""), encoding="utf-8")
        receipt["archive_committed"] = True
        receipt["delivered_at_local"] = receipt.get("delivered_at_local") or local_now_iso()
        store.record_digest_run(
            run_id=str(draft.get("run_id") or run_id),
            run_date=run_date,
            model_used=str(draft.get("model_used") or ""),
            prompt_tokens=int(draft.get("prompt_tokens") or 0),
            completion_tokens=int(draft.get("completion_tokens") or 0),
            raw_count=int(draft.get("raw_count") or 0),
            candidate_count=int(draft.get("candidate_count") or 0),
            selected_count=int(draft.get("selected_count") or 0),
            output_path=str(output_path),
            pushed=True,
        )

    delivery_store.save_receipt(run_date, receipt)


def run_daily_digest(
    config_mgr: ConfigManager,
    *,
    dry_run: bool = False,
    no_push: bool = False,
    no_llm: bool = False,
    window_hours: int | None = None,
    max_candidates: int | None = None,
    check_models: bool = False,
    auto_evolve: bool = False,
    save_candidates_path: Path | None = None,
    notify_channel: str | None = None,
    skip_if_today_exists: bool = False,
    stage: str = "all",
    state_dir: Path | None = None,
    force: bool = False,
    degrade_on_llm_error: bool = True,
    retry_unknown: bool = True,
    run_date_override: str | None = None,
    http: HTTPClient | None = None,
    llm_client: OpenAICompatClient | None = None,
    notifier: Any | None = None,
    notifiers: list[Any] | None = None,
) -> dict[str, Any]:
    """Execute the daily AI briefing workflow with durable draft & receipt persistence.

    Stages:
        - "prepare": check receipts, reuse pending draft or generate & persist a fixed draft (no push).
        - "deliver": deliver persisted draft only to unconfirmed channels, save receipts, then commit archive/history.
        - "all": run prepare + deliver in one invocation (default for local runs).
    """
    eff_stage = (stage or "all").strip().lower()
    if eff_stage not in ("all", "prepare", "deliver"):
        raise ValueError(f"Unsupported stage '{stage}'. Expected one of: all, prepare, deliver")

    http_client = http or HTTPClient()
    store = StateStore(config_mgr.data_dir / "state.db")
    history_store = SentHistoryStore(config_mgr.data_dir / "sent_history.json")
    eff_state_dir = state_dir or (config_mgr.data_dir / "delivery")
    delivery_store = DeliveryStore(eff_state_dir)

    profile = config_mgr.load_profile()
    sources_cfg = config_mgr.load_sources()
    prompts = config_mgr.load_prompts()
    sources = sources_cfg.get("sources", [])

    now_local = current_local_now()
    run_date = (run_date_override or now_local.strftime("%Y-%m-%d")).strip()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path: Path = config_mgr.output_dir / f"{run_date}.md"

    summary = _base_summary(
        run_id=run_id,
        run_date=run_date,
        stage=eff_stage,
        output_path=output_path,
        state_dir=eff_state_dir,
    )

    targets = _resolve_targets(
        notifier=notifier,
        notifiers=notifiers,
        notify_channel=notify_channel,
        http_client=http_client,
    )
    target_names = [getattr(t, "name", "notifier") for t in targets]

    # 0. Receipt-first idempotency check (shared across schedule / repository_dispatch / workflow_dispatch)
    if not dry_run and not no_push and not force:
        existing_receipt = delivery_store.load_receipt(run_date)
        if existing_receipt is not None:
            if delivery_store.all_channels_sent(existing_receipt, target_names):
                existing_draft = delivery_store.load_draft(run_date)
                if existing_draft:
                    _populate_summary_from_draft(summary, existing_draft)
                summary["ok"] = True
                summary["skipped"] = True
                summary["already_sent"] = True
                summary["delivery_status"] = "sent"
                summary["channel_statuses"] = {
                    k: v.get("status", "")
                    for k, v in (existing_receipt.get("channels") or {}).items()
                    if isinstance(v, dict)
                }
                summary["push_error"] = f"今日日报 ({run_date}) 已确认送达所有目标渠道，幂等跳过重复推送"
                return summary
        elif skip_if_today_exists and output_path.is_file():
            # Legacy compatibility when an archive exists from before delivery receipts were introduced
            summary["ok"] = True
            summary["skipped"] = True
            summary["already_sent"] = True
            summary["delivery_status"] = "sent"
            summary["push_error"] = f"今日日报 {output_path.name} 已归档，幂等跳过重复推送"
            return summary

    draft: dict[str, Any] | None = None
    receipt: dict[str, Any] | None = None
    client: OpenAICompatClient | None = None

    # =========================================================================
    # STAGE A: PREPARE (reuse fixed draft if pending/failed, else fetch & curate)
    # =========================================================================
    if eff_stage in ("all", "prepare"):
        existing_draft = (
            delivery_store.load_draft(run_date)
            if (not dry_run and not no_push and not force)
            else None
        )
        if existing_draft is not None:
            draft = existing_draft
            summary["reused_draft"] = True
            _populate_summary_from_draft(summary, draft)
            receipt = delivery_store.ensure_receipt_for_draft(
                run_date,
                draft,
                targets,
                force=False,
            )
            summary["channel_statuses"] = {
                k: v.get("status", "")
                for k, v in (receipt.get("channels") or {}).items()
                if isinstance(v, dict)
            }
            if eff_stage == "prepare":
                summary["ok"] = True
                summary["delivery_status"] = "pending"
                return summary
        else:
            client = llm_client or build_llm_client(config_mgr, http=http_client)
            evolution_notes: list[str] = []

            # Optional model version probe (kept off daily CI critical path)
            if check_models and client.configured and not no_llm:
                try:
                    reg = ModelRegistry(config_mgr, client, store=store)
                    sync_report = reg.sync(apply=False, probe=True)
                    if sync_report.get("upgraded"):
                        evolution_notes.append(
                            f"模型已自动迭代升级至 `{sync_report['recommended_upgrade']}`"
                        )
                        client = build_llm_client(config_mgr, http=http_client)
                    elif sync_report.get("recommended_upgrade"):
                        evolution_notes.append(
                            f"探测到更新模型 `{sync_report['recommended_upgrade']}`，可运行 `ai_daily models sync --apply` 升级"
                        )
                except Exception as exc:
                    evolution_notes.append(f"模型版本探测跳过：{exc}")

            # 1. Fetch all enabled sources concurrently
            eff_hours = window_hours if window_hours is not None else int(profile.get("window_hours", 36))
            fetch_results = fetch_all_sources(sources, http_client, window_hours=eff_hours)
            all_articles = [art for res in fetch_results for art in res.articles]
            sources_ok = sum(1 for res in fetch_results if res.ok)
            failed_sources = [res for res in fetch_results if not res.ok]
            failed_sources_payload = [{"id": r.source_id, "error": r.error} for r in failed_sources]

            summary["sources_total"] = len(fetch_results)
            summary["sources_ok"] = sources_ok
            summary["failed_sources"] = failed_sources_payload
            summary["raw_count"] = len(all_articles)

            # All sources failed -> DO NOT write output/ or draft; preserve failed state for next retry
            if sources_ok == 0 and len(fetch_results) > 0:
                err_msg = "全部信源抓取失败，未生成归档与待发稿，保留失败状态交由下一轮重试"
                summary["ok"] = False
                summary["delivery_status"] = "fetch_failed"
                summary["push_error"] = err_msg
                if not dry_run:
                    delivery_store.record_stage_failure(
                        run_date, kind="fetch_failed", error=err_msg
                    )
                return summary

            if failed_sources:
                failed_names = "、".join(r.source_id for r in failed_sources[:4])
                more = f" 等 {len(failed_sources)} 个" if len(failed_sources) > 4 else ""
                evolution_notes.append(f"本次有信源抓取异常：{failed_names}{more}（已自动记录健康状态）")

            # 2. Dedup (SQLite + 7-day Git sent_history.json) + pre-score
            history_lookup = history_store.build_lookup_sets(within_days=7)
            pushed_hashes, pushed_titles = store.get_pushed_hashes_and_titles()
            source_stats = store.get_source_stats(lookback_runs=15)
            hit_rates = {s["source_id"]: float(s["hit_rate"]) for s in source_stats}

            candidates, new_counts = prepare_candidates(
                all_articles,
                profile=profile,
                sources=sources,
                pushed_hashes=pushed_hashes,
                pushed_norm_titles=pushed_titles,
                history_lookup=history_lookup,
                source_hit_rates=hit_rates,
                window_hours=eff_hours,
                max_candidates=max_candidates,
            )

            if save_candidates_path is not None:
                save_candidates_path.parent.mkdir(parents=True, exist_ok=True)
                dump_rows = [
                    {
                        "id": idx,
                        "source_id": a.source_id,
                        "source_name": a.source_name,
                        "category": a.category,
                        "title": a.title,
                        "url": a.url,
                        "summary": a.summary,
                        "published_at": a.published_at.isoformat() if a.published_at else None,
                        "pre_score": a.pre_score,
                        "matched_keywords": a.matched_keywords,
                        "metrics": a.metrics,
                    }
                    for idx, a in enumerate(candidates, start=1)
                ]
                save_candidates_path.write_text(
                    json.dumps(dump_rows, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

            # 3. Enrich top candidates with full article text, meta publish date + HN comments
            enriched_count = 0
            if not no_llm and candidates:
                enrich_top_n = int(profile.get("enrich_top_n", 18))
                enriched_count = enrich_candidates(
                    candidates,
                    http_client,
                    max_enrich=enrich_top_n,
                )
                origin_max_days = int(profile.get("origin_max_age_days", 14))
                now_utc = datetime.now(timezone.utc)
                origin_cutoff_str = (now_utc - timedelta(days=origin_max_days)).strftime("%Y-%m-%d")
                fresh_candidates = []
                for art in candidates:
                    page_pub = str(art.metrics.get("page_published_at") or "")
                    if page_pub and page_pub < origin_cutoff_str:
                        continue
                    if art.full_text and not art.metrics.get("origin_date_warning"):
                        orig_dt, orig_tag = infer_origin_date(art.url, art.full_text[:1200], now_utc)
                        if orig_dt is not None and orig_dt < now_utc - timedelta(days=21):
                            art.metrics["origin_date_warning"] = f"正文提及原始论文/来源时间 {orig_tag}"
                    fresh_candidates.append(art)
                candidates = fresh_candidates

            summary["candidate_count"] = len(candidates)
            summary["enriched_count"] = enriched_count

            # 4. Curate via LLM (with model fallback and Strategy-B controlled degradation)
            pending_feedback = store.get_feedback(only_pending=True, limit=12)
            recent_7d_history = history_store.build_prompt_context(within_days=7)
            try:
                curate_res = curate_candidates(
                    candidates,
                    run_date=run_date,
                    profile=profile,
                    prompts=prompts,
                    llm_client=client,
                    feedback_rows=pending_feedback,
                    recent_7d_history=recent_7d_history,
                    use_llm=not no_llm,
                    allow_degraded_fallback=degrade_on_llm_error,
                )
            except LLMCurationError as exc:
                err_msg = f"主备大模型调用均失败（按策略保留失败状态，交由兜底轮重试）：{exc}"
                summary["ok"] = False
                summary["delivery_status"] = "llm_failed"
                summary["push_error"] = err_msg
                if not dry_run:
                    delivery_store.record_stage_failure(
                        run_date, kind="llm_failed", error=str(exc)
                    )
                return summary

            # 5. Render fixed Markdown & Feishu Interactive Card
            title, short_preview, markdown = render_digest_markdown(
                curate_res,
                run_date=run_date,
                raw_count=len(all_articles),
                sources_ok=sources_ok,
                sources_total=len(fetch_results),
                evolution_notes=evolution_notes,
            )
            feishu_card = _trim_card_to_budget(
                build_digest_card(
                    curate_res,
                    run_date=run_date,
                    raw_count=len(all_articles),
                    sources_ok=sources_ok,
                    sources_total=len(fetch_results),
                    evolution_notes=evolution_notes,
                )
            )

            draft = build_draft_payload(
                run_id=run_id,
                run_date=run_date,
                curate_res=curate_res,
                title=title,
                short_preview=short_preview,
                markdown=markdown,
                feishu_card=feishu_card,
                raw_count=len(all_articles),
                candidate_count=len(candidates),
                enriched_count=enriched_count,
                sources_ok=sources_ok,
                sources_total=len(fetch_results),
                failed_sources=failed_sources_payload,
                evolution_notes=evolution_notes,
            )
            _populate_summary_from_draft(summary, draft)

            # Dry-run or --no-push: write local preview file and stop before delivery state
            if dry_run or no_push:
                config_mgr.output_dir.mkdir(parents=True, exist_ok=True)
                output_path.write_text(markdown, encoding="utf-8")
                if no_push and not dry_run:
                    store.record_seen_articles(candidates, normalize_title)
                    store.record_source_runs(run_id, fetch_results, new_counts, curate_res.selected)
                    store.record_emerging_signals(curate_res.emerging_signals)
                    store.record_digest_run(
                        run_id=run_id,
                        run_date=run_date,
                        model_used=curate_res.model_used,
                        prompt_tokens=curate_res.prompt_tokens,
                        completion_tokens=curate_res.completion_tokens,
                        raw_count=len(all_articles),
                        candidate_count=len(candidates),
                        selected_count=len(curate_res.selected),
                        output_path=str(output_path),
                        pushed=False,
                    )
                summary["ok"] = True
                summary["delivery_status"] = "dry_run" if dry_run else "no_push"
                return summary

            # Persist non-pushed source telemetry locally (never mark pushed_at or sent_history yet!)
            store.record_seen_articles(candidates, normalize_title)
            store.record_source_runs(run_id, fetch_results, new_counts, curate_res.selected)
            store.record_emerging_signals(curate_res.emerging_signals)

            # Save fixed draft + pending per-channel receipt BEFORE calling any notifier
            delivery_store.save_draft(run_date, draft)
            receipt = delivery_store.ensure_receipt_for_draft(
                run_date,
                draft,
                targets,
                force=force,
                force_nonce=run_id if force else "",
            )
            summary["channel_statuses"] = {
                k: v.get("status", "")
                for k, v in (receipt.get("channels") or {}).items()
                if isinstance(v, dict)
            }

            if eff_stage == "prepare":
                summary["ok"] = True
                summary["delivery_status"] = "pending"
                return summary

    # =========================================================================
    # STAGE B: DELIVER (send only unconfirmed channels, save receipt, then commit history & archive)
    # =========================================================================
    if eff_stage in ("all", "deliver") and not dry_run and not no_push:
        if draft is None:
            draft = delivery_store.load_draft(run_date)
        if draft is None:
            summary["ok"] = False
            summary["delivery_status"] = "missing_draft"
            summary["push_error"] = f"未找到 {run_date} 的待发成稿 (draft.json)，无法执行补发"
            return summary
        _populate_summary_from_draft(summary, draft)

        if receipt is None:
            receipt = delivery_store.ensure_receipt_for_draft(
                run_date,
                draft,
                targets,
                force=force,
                force_nonce=run_id if force else "",
            )

        channels_map: dict[str, Any] = receipt.setdefault("channels", {})
        pushed_channels: list[str] = []
        errors: list[str] = []
        has_unknown = False

        for target in targets:
            ch_name = getattr(target, "name", "notifier")
            ch_rec = channels_map.get(ch_name)
            if not isinstance(ch_rec, dict):
                receipt = delivery_store.ensure_receipt_for_draft(run_date, draft, targets, force=False)
                channels_map = receipt["channels"]
                ch_rec = channels_map[ch_name]

            if ch_rec.get("status") == "sent" and not force:
                continue

            supports_idem = bool(getattr(target, "supports_idempotent_retry", False))
            if ch_rec.get("status") == "unknown" and not supports_idem:
                if not retry_unknown or int(ch_rec.get("attempts") or 0) >= 2:
                    has_unknown = True
                    errors.append(
                        f"{ch_name}: 上次发送结果未知且渠道无幂等键，已停止自动重发以防刷屏，请人工核对"
                    )
                    continue

            if not getattr(target, "configured", False):
                err_txt = f"{ch_name} 未配置"
                ch_rec["status"] = "failed"
                ch_rec["last_error"] = err_txt
                ch_rec["updated_at"] = utc_now_iso()
                delivery_store.save_receipt(run_date, receipt)
                errors.append(err_txt)
                continue

            ch_rec["attempts"] = int(ch_rec.get("attempts") or 0) + 1
            ch_rec["mode"] = getattr(target, "delivery_mode", ch_rec.get("mode", ch_name))
            idem_key = str(ch_rec.get("idempotency_key") or "")

            try:
                if _is_feishu_notifier(target) or hasattr(target, "send_card"):
                    resp_data = target.send_card(draft["feishu_card"], idempotency_key=idem_key)
                else:
                    resp_data = target.send(
                        draft["title"],
                        draft["markdown"],
                    )
                ch_rec["status"] = "sent"
                ch_rec["message_id"] = str((resp_data or {}).get("message_id") or "")
                ch_rec["last_error"] = ""
                ch_rec["sent_at"] = utc_now_iso()
                ch_rec["sent_at_local"] = local_now_iso()
                ch_rec["updated_at"] = utc_now_iso()
                pushed_channels.append(ch_name)
            except DeliveryUnknownError as exc:
                has_unknown = True
                ch_rec["status"] = "unknown"
                ch_rec["last_error"] = str(exc)
                ch_rec["updated_at"] = utc_now_iso()
                errors.append(f"{ch_name} [结果未知]: {exc}")
            except Exception as exc:
                ch_rec["status"] = "failed"
                ch_rec["last_error"] = str(exc)
                ch_rec["updated_at"] = utc_now_iso()
                errors.append(f"{ch_name}: {exc}")

            delivery_store.save_receipt(run_date, receipt)

        all_sent = delivery_store.all_channels_sent(receipt, target_names)
        any_sent = any(
            isinstance(channels_map.get(name), dict) and channels_map[name].get("status") == "sent"
            for name in target_names
        )

        # Persist dedup history once any channel confirmed; persist output archive once ALL target channels confirmed
        if any_sent:
            _commit_confirmed_delivery(
                config_mgr=config_mgr,
                store=store,
                history_store=history_store,
                delivery_store=delivery_store,
                run_id=run_id,
                run_date=run_date,
                output_path=output_path,
                draft=draft,
                receipt=receipt,
                write_archive=all_sent,
            )

        summary["pushed"] = all_sent
        summary["pushed_channels"] = pushed_channels
        summary["channel_statuses"] = {
            name: str((channels_map.get(name) or {}).get("status") or "")
            for name in target_names
        }
        if errors:
            summary["push_error"] = "；".join(errors)

        if all_sent:
            summary["ok"] = True
            summary["delivery_status"] = "sent"
        elif has_unknown:
            summary["ok"] = False
            summary["delivery_status"] = "unknown"
        else:
            summary["ok"] = False
            summary["delivery_status"] = "failed"

        # Optional self-evolution ONLY when explicitly requested AND isolated from delivery outcome
        if auto_evolve and all_sent:
            try:
                every_n = int((profile.get("evolution") or {}).get("auto_evolve_every_n_runs", 5))
                total_runs = store.get_total_digest_runs()
                if total_runs > 0 and total_runs % max(every_n, 1) == 0:
                    from ai_daily.evolution.engine import EvolutionEngine

                    evo_client = client or llm_client or build_llm_client(config_mgr, http=http_client)
                    engine = EvolutionEngine(config_mgr, store, evo_client if not no_llm else None)
                    summary["evolution_summary"] = engine.run_cycle(apply=False)
            except Exception as exc:
                summary["evolution_summary"] = {"error": f"自演进提案生成失败（不影响日报推送）: {exc}"}

    return summary
