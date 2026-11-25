"""Autonomous self-iteration engine combining telemetry guardrails with LLM reflection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_daily.config import ConfigManager
from ai_daily.llm.client import OpenAICompatClient
from ai_daily.models import utc_now_iso
from ai_daily.storage import StateStore

VALID_TRACKS = {"harness", "models", "systems", "geek"}


def _merge_rule_proposals(
    profile: dict[str, Any],
    sources_cfg: dict[str, Any],
    source_stats: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    feedback: list[dict[str, Any]],
) -> dict[str, Any]:
    evo_cfg = profile.get("evolution") or {}
    min_runs = int(evo_cfg.get("min_runs_before_pruning", 7))
    max_fail = int(evo_cfg.get("max_consecutive_failures", 5))
    max_new_kw = int(evo_cfg.get("max_auto_added_keywords_per_cycle", 5))

    source_map = {s["id"]: s for s in sources_cfg.get("sources", [])}
    source_adjustments: list[dict[str, Any]] = []

    for stat in source_stats:
        sid = stat["source_id"]
        src = source_map.get(sid)
        if not src or not src.get("enabled", True):
            continue
        cur_weight = float(src.get("weight", 1.0))
        if stat["consecutive_failures"] >= max_fail:
            source_adjustments.append(
                {
                    "source_id": sid,
                    "action": "disable",
                    "new_weight": cur_weight,
                    "reason": f"连续 {stat['consecutive_failures']} 次抓取失败（最近错误：{stat['last_error'][:100]}）",
                }
            )
        elif stat["runs"] >= min_runs and stat["total_new"] >= 20 and stat["total_selected"] == 0:
            new_w = round(max(0.5, cur_weight * 0.8), 2)
            if new_w < cur_weight:
                source_adjustments.append(
                    {
                        "source_id": sid,
                        "action": "demote",
                        "new_weight": new_w,
                        "reason": f"近 {stat['runs']} 期抓取 {stat['total_new']} 条新内容但 0 条入选，自动降权",
                    }
                )
        elif stat["runs"] >= 5 and stat["hit_rate"] >= 0.25 and stat["mean_selected_score"] >= 8.0:
            new_w = round(min(1.8, cur_weight * 1.1), 2)
            if new_w > cur_weight:
                source_adjustments.append(
                    {
                        "source_id": sid,
                        "action": "boost",
                        "new_weight": new_w,
                        "reason": f"近 {stat['runs']} 期入选率 {stat['hit_rate']:.0%} 且均分 {stat['mean_selected_score']}，自动提权",
                    }
                )

    add_keywords: list[dict[str, Any]] = []
    for sig in signals:
        if len(add_keywords) >= max_new_kw:
            break
        if int(sig.get("seen_count", 1)) >= 2:
            track = sig.get("track", "geek")
            if track not in VALID_TRACKS:
                track = "geek"
            add_keywords.append(
                {
                    "track": track,
                    "keyword": sig["keyword"],
                    "reason": f"近期跨期出现 {sig['seen_count']} 次：{sig.get('reason', '')}",
                }
            )

    add_boost: list[dict[str, Any]] = []
    add_exclude: list[dict[str, Any]] = []
    user_notes: list[str] = []
    for fb in feedback:
        kind = fb.get("kind")
        target = str(fb.get("target") or "").strip()
        detail = str(fb.get("detail") or "").strip()
        if not target:
            continue
        if kind in ("topic", "like"):
            add_boost.append({"keyword": target, "reason": detail or f"来自用户 {kind} 反馈"})
        elif kind == "dislike":
            add_exclude.append({"keyword": target, "reason": detail or "来自用户 dislike 反馈"})
        elif kind == "note":
            user_notes.append(target if not detail else f"{target}：{detail}")

    return {
        "summary": "基于运行遥测数据与用户反馈生成的自主迭代提案。",
        "add_keywords": add_keywords,
        "add_boost_keywords": add_boost,
        "add_exclude_keywords": add_exclude,
        "source_adjustments": source_adjustments,
        "candidate_new_sources": [],
        "user_note_additions": user_notes,
    }


def _dedup_proposals(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "summary": extra.get("summary") or base.get("summary") or "",
        "add_keywords": [],
        "add_boost_keywords": [],
        "add_exclude_keywords": [],
        "source_adjustments": [],
        "candidate_new_sources": list(extra.get("candidate_new_sources") or []),
        "user_note_additions": [],
    }

    seen_kw: set[tuple[str, str]] = set()
    for item in [*(base.get("add_keywords") or []), *(extra.get("add_keywords") or [])]:
        if not isinstance(item, dict):
            continue
        kw = str(item.get("keyword") or "").strip()
        track = str(item.get("track") or "geek").strip().lower()
        if kw and track in VALID_TRACKS and (track, kw.lower()) not in seen_kw:
            seen_kw.add((track, kw.lower()))
            merged["add_keywords"].append(
                {"track": track, "keyword": kw, "reason": str(item.get("reason") or "")}
            )

    for key in ("add_boost_keywords", "add_exclude_keywords"):
        seen_k: set[str] = set()
        for item in [*(base.get(key) or []), *(extra.get(key) or [])]:
            if not isinstance(item, dict):
                continue
            kw = str(item.get("keyword") or "").strip()
            if kw and kw.lower() not in seen_k:
                seen_k.add(kw.lower())
                merged[key].append({"keyword": kw, "reason": str(item.get("reason") or "")})

    seen_sid: set[str] = set()
    for adj in [*(base.get("source_adjustments") or []), *(extra.get("source_adjustments") or [])]:
        if not isinstance(adj, dict):
            continue
        sid = str(adj.get("source_id") or "").strip()
        action = str(adj.get("action") or "").strip().lower()
        if sid and action in ("boost", "demote", "disable") and sid not in seen_sid:
            seen_sid.add(sid)
            merged["source_adjustments"].append(adj)

    seen_notes: set[str] = set()
    for note in [*(base.get("user_note_additions") or []), *(extra.get("user_note_additions") or [])]:
        txt = str(note).strip()
        if txt and txt not in seen_notes:
            seen_notes.add(txt)
            merged["user_note_additions"].append(txt)

    return merged


class EvolutionEngine:
    """Analyzes telemetry, signals, and feedback to safely evolve profile.json and sources.json."""

    def __init__(
        self,
        config_mgr: ConfigManager,
        store: StateStore,
        llm_client: OpenAICompatClient | None = None,
    ) -> None:
        self.config_mgr = config_mgr
        self.store = store
        self.llm_client = llm_client
        self.proposal_path: Path = config_mgr.data_dir / "evolution_proposals.json"

    def generate_proposal(self, *, use_llm: bool = True) -> dict[str, Any]:
        profile = self.config_mgr.load_profile()
        sources_cfg = self.config_mgr.load_sources()
        prompts = self.config_mgr.load_prompts()

        source_stats = self.store.get_source_stats(lookback_runs=20)
        signals = self.store.get_pending_emerging_signals(limit=25)
        feedback = self.store.get_feedback(only_pending=True, limit=25)

        rule_proposal = _merge_rule_proposals(
            profile, sources_cfg, source_stats, signals, feedback
        )
        llm_proposal: dict[str, Any] = {}
        model_used = "rule-only"

        if (
            use_llm
            and self.llm_client is not None
            and self.llm_client.configured
            and (source_stats or signals or feedback)
        ):
            user_prompt = prompts["evolve_user_template"].format(
                profile_json=json.dumps(
                    {
                        "tracks": profile.get("tracks", []),
                        "boost_keywords": profile.get("boost_keywords", []),
                        "exclude_keywords": profile.get("exclude_keywords", []),
                        "user_notes": profile.get("user_notes", []),
                    },
                    ensure_ascii=False,
                ),
                source_stats_json=json.dumps(source_stats, ensure_ascii=False),
                emerging_signals_json=json.dumps(signals, ensure_ascii=False),
                feedback_json=json.dumps(feedback, ensure_ascii=False),
            )
            llm_proposal, resp = self.llm_client.chat_json(
                prompts["evolve_system"], user_prompt
            )
            model_used = resp.model

        merged = _dedup_proposals(rule_proposal, llm_proposal)
        merged["generated_at"] = utc_now_iso()
        merged["model_used"] = model_used
        merged["feedback_ids"] = [int(f["id"]) for f in feedback]

        self.proposal_path.parent.mkdir(parents=True, exist_ok=True)
        self.proposal_path.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return merged

    def apply_proposal(self, proposal: dict[str, Any]) -> dict[str, Any]:
        profile = self.config_mgr.load_profile()
        sources_cfg = self.config_mgr.load_sources()
        evo_cfg = profile.get("evolution") or {}
        max_new_kw = int(evo_cfg.get("max_auto_added_keywords_per_cycle", 5))

        profile_changed = False
        sources_changed = False
        applied_changes: dict[str, Any] = {
            "added_track_keywords": [],
            "added_boost_keywords": [],
            "added_exclude_keywords": [],
            "added_user_notes": [],
            "source_updates": [],
            "backups": {},
        }

        # 1. Track keywords
        track_map = {t["id"]: t for t in profile.get("tracks", [])}
        applied_signal_kws: list[str] = []
        for item in (proposal.get("add_keywords") or [])[:max_new_kw]:
            tid = item.get("track")
            kw = str(item.get("keyword") or "").strip()
            if not kw or tid not in track_map:
                continue
            existing_low = {k.lower() for k in track_map[tid].get("keywords", [])}
            if kw.lower() not in existing_low:
                track_map[tid].setdefault("keywords", []).append(kw)
                applied_changes["added_track_keywords"].append({"track": tid, "keyword": kw})
                applied_signal_kws.append(kw)
                profile_changed = True

        # 2. Boost keywords
        boost_list = profile.setdefault("boost_keywords", [])
        boost_low = {k.lower() for k in boost_list}
        for item in (proposal.get("add_boost_keywords") or [])[:max_new_kw]:
            kw = str(item.get("keyword") or "").strip()
            if kw and kw.lower() not in boost_low:
                boost_list.append(kw)
                boost_low.add(kw.lower())
                applied_changes["added_boost_keywords"].append(kw)
                applied_signal_kws.append(kw)
                profile_changed = True

        # 3. Exclude keywords
        exclude_list = profile.setdefault("exclude_keywords", [])
        exclude_low = {k.lower() for k in exclude_list}
        for item in (proposal.get("add_exclude_keywords") or [])[:max_new_kw]:
            kw = str(item.get("keyword") or "").strip()
            if kw and kw.lower() not in exclude_low:
                exclude_list.append(kw)
                exclude_low.add(kw.lower())
                applied_changes["added_exclude_keywords"].append(kw)
                profile_changed = True

        # 4. User notes
        notes_list = profile.setdefault("user_notes", [])
        for note in (proposal.get("user_note_additions") or [])[:3]:
            txt = str(note).strip()
            if txt and txt not in notes_list:
                notes_list.append(txt)
                applied_changes["added_user_notes"].append(txt)
                profile_changed = True

        # 5. Source weight / disable adjustments
        src_map = {s["id"]: s for s in sources_cfg.get("sources", [])}
        for adj in proposal.get("source_adjustments") or []:
            sid = adj.get("source_id")
            action = adj.get("action")
            src = src_map.get(sid)
            if not src:
                continue
            if action == "disable" and src.get("enabled", True):
                src["enabled"] = False
                applied_changes["source_updates"].append(
                    {"source_id": sid, "action": "disable", "reason": adj.get("reason", "")}
                )
                sources_changed = True
            elif action in ("boost", "demote"):
                cur_w = float(src.get("weight", 1.0))
                raw_new = adj.get("new_weight")
                try:
                    new_w = float(raw_new) if raw_new is not None else (
                        cur_w * 1.1 if action == "boost" else cur_w * 0.85
                    )
                except (TypeError, ValueError):
                    new_w = cur_w
                new_w = round(min(2.0, max(0.4, new_w)), 2)
                if new_w != cur_w:
                    src["weight"] = new_w
                    applied_changes["source_updates"].append(
                        {
                            "source_id": sid,
                            "action": action,
                            "old_weight": cur_w,
                            "new_weight": new_w,
                            "reason": adj.get("reason", ""),
                        }
                    )
                    sources_changed = True

        if profile_changed:
            bpath = self.config_mgr.save_with_backup("profile", profile, tag="evolve")
            applied_changes["backups"]["profile"] = str(bpath) if bpath else ""
        if sources_changed:
            bpath = self.config_mgr.save_with_backup("sources", sources_cfg, tag="evolve")
            applied_changes["backups"]["sources"] = str(bpath) if bpath else ""

        if applied_signal_kws:
            self.store.mark_signals_applied(applied_signal_kws)
        fb_ids = proposal.get("feedback_ids") or []
        if fb_ids:
            self.store.mark_feedback_applied(fb_ids)

        if profile_changed or sources_changed:
            backup_summary = ", ".join(
                f"{k}:{v}" for k, v in applied_changes["backups"].items() if v
            )
            self.store.record_evolution_log(
                kind="config_evolve",
                summary=proposal.get("summary") or "Applied autonomous evolution updates",
                backup_path=backup_summary,
                diff_obj=applied_changes,
            )

        return {
            "applied": profile_changed or sources_changed,
            "changes": applied_changes,
            "candidate_new_sources": proposal.get("candidate_new_sources") or [],
        }

    def run_cycle(self, *, apply: bool = False, use_llm: bool = True) -> dict[str, Any]:
        proposal = self.generate_proposal(use_llm=use_llm)
        result: dict[str, Any] = {"proposal": proposal, "apply_result": None}
        if apply:
            result["apply_result"] = self.apply_proposal(proposal)
        return result
