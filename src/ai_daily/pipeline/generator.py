"""LLM curation + verified-link Markdown digest rendering (with deterministic rule fallback)."""

from __future__ import annotations

import json
import re
from typing import Any

from ai_daily.llm.client import OpenAICompatClient
from ai_daily.models import (
    TRACK_ORDER,
    Article,
    CuratedItem,
    CurateResult,
    EmergingSignal,
    format_related_links,
)

_LEADING_DECOR_RE = re.compile(r"^(?:[🔬💡⚡🛠️🚀⚙️🧪🎲]+\s*|(?:原文|注)[:：]\s*)+")
TRACK_HEADING_MAP = dict(TRACK_ORDER)


class LLMCurationError(RuntimeError):
    """Raised when all configured LLM models fail curation and degraded fallback is disabled."""


def compute_or_gate_score(tech: float, fun: float, impact: float) -> float:
    """Single-peak OR-gate score: 0.6 * max + 0.4 * avg(other two).

    Ensures a wildly fun OR deeply technical item sails through even if its other scores are modest.
    """
    vals = sorted(
        [max(0.0, min(10.0, float(v))) for v in (tech, fun, impact)],
        reverse=True,
    )
    return round(vals[0] * 0.6 + ((vals[1] + vals[2]) / 2.0) * 0.4, 1)


def _clean_field(val: Any) -> str:
    return _LEADING_DECOR_RE.sub("", str(val or "").strip()).strip()

VALID_TRACKS = {t[0] for t in TRACK_ORDER}
LEGACY_TRACK_MAP = {
    "portals": "geek",
    "tech_share": "systems",
}


def _normalize_track(raw_track: str, fallback: str = "geek") -> str:
    t = (raw_track or "").strip().lower()
    t = LEGACY_TRACK_MAP.get(t, t)
    if t in VALID_TRACKS:
        return t
    fb = LEGACY_TRACK_MAP.get(fallback, fallback)
    return fb if fb in VALID_TRACKS else "geek"


def _build_profile_summary(profile: dict[str, Any]) -> str:
    lines: list[str] = []
    for t in profile.get("tracks", []):
        kws = ", ".join(t.get("keywords", [])[:12])
        lines.append(
            f"- [{t['id']}] {t['label']} (权重 {t.get('weight', 1.0)}): {t.get('description', '')} | 关键词: {kws}"
        )
    boost = ", ".join(profile.get("boost_keywords", []))
    if boost:
        lines.append(f"- 重点加权词: {boost}")
    for note in profile.get("user_notes", []):
        lines.append(f"- 主编备注: {note}")
    return "\n".join(lines)


def _build_feedback_notes(feedback_rows: list[dict[str, Any]]) -> str:
    if not feedback_rows:
        return "（暂无额外近期反馈）"
    lines: list[str] = []
    for row in feedback_rows[:12]:
        detail = f"（{row['detail']}）" if row.get("detail") else ""
        lines.append(f"- [{row['kind']}] {row['target']}{detail}")
    return "\n".join(lines)


def _curate_rule_fallback(
    candidates: list[Article],
    profile: dict[str, Any],
    *,
    degrade_reason: str = "",
) -> CurateResult:
    raw_target = int(profile.get("target_digest_items", 12))
    target = min(raw_target, 8) if degrade_reason else raw_target
    picked = candidates[:target]
    items: list[CuratedItem] = []
    for idx, art in enumerate(picked, start=1):
        track = _normalize_track(art.category)
        norm_score = min(9.5, max(6.0, round(5.5 + art.pre_score * 0.45, 1)))
        why_text = (
            "（LLM 服务临时不可用，本条按信源权重与关键词规则降级保底，已通过时效与 7 天去重校验）"
            if degrade_reason
            else "（离线规则模式生成；配置 DeepSeek API Key 后将由大模型提供深度中文解读与事件聚合）"
        )
        items.append(
            CuratedItem(
                candidate_id=idx,
                article=art,
                related_articles=[],
                track=track,
                score=norm_score,
                zh_title=art.title,
                one_liner=art.summary[:160] if art.summary else art.title,
                why_it_matters=why_text,
                tags=art.matched_keywords[:4] or art.tags[:3],
            )
        )
    headlines = [item.zh_title[:45] for item in items[:3]]
    return CurateResult(
        headline_summary=headlines,
        selected=items,
        emerging_signals=[],
        model_used="rule-fallback(degraded)" if degrade_reason else "rule-fallback",
        used_fallback_rule_mode=True,
        degrade_reason=degrade_reason,
    )


def curate_candidates(
    candidates: list[Article],
    *,
    run_date: str,
    profile: dict[str, Any],
    prompts: dict[str, Any],
    llm_client: OpenAICompatClient | None,
    feedback_rows: list[dict[str, Any]] | None = None,
    recent_7d_history: str = "（过去 7 天暂无已发历史记录）",
    use_llm: bool = True,
    allow_degraded_fallback: bool = True,
) -> CurateResult:
    if not candidates:
        return CurateResult(headline_summary=[], selected=[], emerging_signals=[])

    if not use_llm or llm_client is None or not llm_client.configured:
        return _curate_rule_fallback(candidates, profile)

    id_to_article: dict[int, Article] = {}
    compact_candidates: list[dict[str, Any]] = []
    for idx, art in enumerate(candidates, start=1):
        id_to_article[idx] = art
        entry: dict[str, Any] = {
            "id": idx,
            "source": art.source_name,
            "category": art.category,
            "title": art.title,
            "summary": art.summary[:340],
            "published_at": art.published_at.strftime("%Y-%m-%d %H:%M UTC")
            if art.published_at
            else "unknown",
            "pre_score": art.pre_score,
            "matched": art.matched_keywords[:6],
            "url": art.url,
        }
        if art.full_text:
            entry["full_text"] = art.full_text[:2200]
        if art.metrics.get("page_published_at"):
            entry["page_published_at"] = art.metrics["page_published_at"]
        if art.metrics.get("origin_date_warning"):
            entry["origin_date_warning"] = art.metrics["origin_date_warning"]
        if art.metrics.get("history_hit"):
            entry["history_hit"] = art.metrics["history_hit"]
        anchors = art.metrics.get("extracted_anchors") or []
        hn_url = art.metrics.get("hn_url")
        if hn_url and hn_url not in anchors:
            anchors = [*anchors, hn_url]
        if anchors:
            entry["anchors"] = anchors[:5]
        compact_candidates.append(entry)

    max_per_track = int(profile.get("max_per_track", 4))
    quotas = profile.get("track_quotas") or {
        "harness": "2-3",
        "models": "2-3",
        "systems": "2-3",
        "geek": "1-2",
    }
    quotas_str = " / ".join(f"{k}:{v}条" for k, v in quotas.items())

    user_prompt = prompts["curate_user_template"].format(
        date=run_date,
        target_items=int(profile.get("target_digest_items", 10)),
        min_score=float(profile.get("min_score", 6.8)),
        track_quotas=quotas_str,
        max_per_track=max_per_track,
        recent_7d_history=recent_7d_history,
        profile_summary=_build_profile_summary(profile),
        feedback_notes=_build_feedback_notes(feedback_rows or []),
        candidates_json=json.dumps(compact_candidates, ensure_ascii=False),
    )

    try:
        parsed, resp = llm_client.chat_json(
            prompts["curate_system"],
            user_prompt,
            validator=lambda obj: isinstance(obj.get("selected"), list),
        )
    except Exception as exc:
        reason = " ".join(str(exc).split())[:240]
        if allow_degraded_fallback:
            return _curate_rule_fallback(candidates, profile, degrade_reason=reason)
        raise LLMCurationError(reason) from exc

    min_score = float(profile.get("min_score", 6.8))
    selected: list[CuratedItem] = []
    used_ids: set[int] = set()
    section_counts: dict[str, int] = {}

    raw_selected = [
        x for x in (parsed.get("selected") or []) if isinstance(x, dict)
    ]

    def _safe_float(d: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            return float(d.get(key, default))
        except (TypeError, ValueError):
            return default

    def _effective_score(d: dict[str, Any]) -> float:
        has_dims = any(k in d for k in ("tech_score", "fun_score", "impact_score"))
        if has_dims:
            return compute_or_gate_score(
                _safe_float(d, "tech_score", 0.0),
                _safe_float(d, "fun_score", 0.0),
                _safe_float(d, "impact_score", 0.0),
            )
        return round(_safe_float(d, "score", 0.0), 1)

    raw_selected.sort(key=_effective_score, reverse=True)

    for raw_item in raw_selected:
        try:
            cid = int(raw_item.get("id"))
        except (TypeError, ValueError):
            continue
        art = id_to_article.get(cid)
        if art is None or cid in used_ids:
            continue
        score = _effective_score(raw_item)
        if score < min_score:
            continue

        track = _normalize_track(str(raw_item.get("track") or ""), fallback=art.category)
        section = str(raw_item.get("section") or "").strip() or TRACK_HEADING_MAP.get(
            track, "🔥 今日精选"
        )
        if section_counts.get(section, 0) >= max_per_track:
            continue

        used_ids.add(cid)
        section_counts[section] = section_counts.get(section, 0) + 1

        related: list[Article] = []
        for rid_raw in raw_item.get("related_ids") or []:
            try:
                rid = int(rid_raw)
            except (TypeError, ValueError):
                continue
            if rid != cid and rid in id_to_article and rid not in used_ids:
                related.append(id_to_article[rid])
                used_ids.add(rid)

        tags_raw = raw_item.get("tags") or []
        tags = [str(t).strip() for t in tags_raw if str(t).strip()][:5]

        selected.append(
            CuratedItem(
                candidate_id=cid,
                article=art,
                related_articles=related,
                track=track,
                score=score,
                zh_title=str(raw_item.get("zh_title") or art.title).strip(),
                one_liner=_clean_field(raw_item.get("one_liner") or art.summary[:160]),
                why_it_matters=_clean_field(raw_item.get("why_it_matters") or ""),
                tech_depth=_clean_field(raw_item.get("tech_depth") or ""),
                geek_action=_clean_field(raw_item.get("geek_action") or ""),
                section=section,
                tech_score=round(_safe_float(raw_item, "tech_score", 0.0), 1),
                fun_score=round(_safe_float(raw_item, "fun_score", 0.0), 1),
                impact_score=round(_safe_float(raw_item, "impact_score", 0.0), 1),
                tags=tags,
            )
        )

    selected.sort(key=lambda x: x.score, reverse=True)

    # Guardrail: if LLM returned overly strict empty list, fallback to top 5 candidates
    if not selected and candidates:
        return _curate_rule_fallback(candidates[:5], profile)

    headlines = [
        str(h).strip()
        for h in (parsed.get("headline_summary") or [])
        if str(h).strip()
    ][:4]
    if not headlines:
        headlines = [item.zh_title for item in selected[:3]]

    signals: list[EmergingSignal] = []
    for raw_sig in parsed.get("emerging_signals") or []:
        if not isinstance(raw_sig, dict):
            continue
        kw = str(raw_sig.get("keyword") or "").strip()
        if not kw:
            continue
        sig_track = _normalize_track(str(raw_sig.get("track") or "geek"))
        signals.append(
            EmergingSignal(
                keyword=kw,
                track=sig_track,
                reason=str(raw_sig.get("reason") or "").strip(),
            )
        )

    return CurateResult(
        headline_summary=headlines,
        selected=selected,
        emerging_signals=signals,
        model_used=resp.model,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        used_fallback_rule_mode=False,
    )


def render_digest_markdown(
    result: CurateResult,
    *,
    run_date: str,
    raw_count: int,
    sources_ok: int,
    sources_total: int,
    evolution_notes: list[str] | None = None,
) -> tuple[str, str, str]:
    """Render clean Markdown digest for local archiving and Feishu card fallback.

    Returns:
        (title, short_preview, markdown_body)
    """
    short_date = run_date[5:] if len(run_date) >= 10 else run_date
    is_degraded = bool(result.used_fallback_rule_mode and result.degrade_reason)
    title = (
        f"AI前沿日报 {short_date}｜降级短报·{len(result.selected)}条"
        if is_degraded
        else f"AI前沿日报 {short_date}｜共{len(result.selected)}条精选"
    )
    short_preview = (
        "；".join(result.headline_summary[:2])[:64]
        if result.headline_summary
        else f"{run_date} AI 前沿与 Harness 动态精选"
    )

    lines: list[str] = [
        f"# AI 前沿日报 · {run_date}" + ("（降级短报）" if is_degraded else ""),
        "",
    ]
    if is_degraded:
        lines.append(
            f"> ⚠️ **降级短报提示**：主备大模型调用均失败，以下为已通过时效与 7 天去重检查的规则精选候选（原因：{result.degrade_reason}）。"
        )
        lines.append("")
    lines.extend(
        [
            f"> 扫描信源 **{sources_ok}/{sources_total}** 个 · 原始候选 **{raw_count}** 条 · 精选 **{len(result.selected)}** 条 · 模型 `{result.model_used or 'N/A'}`",
            "",
        ]
    )

    if result.headline_summary:
        lines.append("## ⚡ 今日必看速览")
        lines.append("")
        for h in result.headline_summary:
            lines.append(f"- {h}")
        lines.append("")

    if not result.selected:
        lines.append("今日时间窗口内暂无满足阈值的新增前沿动态。")
        lines.append("")
    else:
        grouped: dict[str, list[CuratedItem]] = {}
        section_order: list[str] = []
        for item in result.selected:
            sec = item.section.strip() if item.section else TRACK_HEADING_MAP.get(item.track, "🔥 今日精选")
            if sec not in grouped:
                grouped[sec] = []
                section_order.append(sec)
            grouped[sec].append(item)

        for section_heading in section_order:
            items = grouped[section_heading]
            lines.append(f"## {section_heading}")
            lines.append("")
            for idx, item in enumerate(items, start=1):
                tag_str = " ".join(f"`{t}`" for t in item.tags[:4])
                dim_str = ""
                if item.tech_score or item.fun_score or item.impact_score:
                    dim_str = f" · 技{item.tech_score:.0f}/趣{item.fun_score:.0f}/影{item.impact_score:.0f}"
                lines.append(
                    f"### {idx}. [{item.zh_title}]({item.article.url})（{item.score:.1f}分{dim_str}）"
                )
                lines.append("")
                lines.append(f"- **核心要点**：{item.one_liner}")
                if item.tech_depth:
                    lines.append(f"- **🔬 技术与架构细节**：{item.tech_depth}")
                if item.why_it_matters:
                    lines.append(f"- **💡 启发与价值**：{item.why_it_matters}")
                if item.geek_action:
                    lines.append(f"- **⚡ 极客入口**：{item.geek_action}")
                meta = f"- **来源**：{item.article.source_name}"
                if tag_str:
                    meta += f" ｜ {tag_str}"
                lines.append(meta)
                rel_pairs = format_related_links(item.article, item.related_articles)
                if rel_pairs:
                    rel_links = "、".join(f"[{lbl}]({u})" for lbl, u in rel_pairs)
                    lines.append(f"- **关联报道**：{rel_links}")
                lines.append("")

    # Evolution & telemetry footer
    footer_notes = list(evolution_notes or [])
    if result.emerging_signals:
        sig_str = "、".join(f"`{s.keyword}`" for s in result.emerging_signals[:5])
        footer_notes.append(f"新发现潜在追踪信号：{sig_str}（已入库，可通过 `python3 run.py evolve` 纳入迭代）")
    if result.prompt_tokens or result.completion_tokens:
        footer_notes.append(
            f"本次 Token 消耗：输入 {result.prompt_tokens} / 输出 {result.completion_tokens}"
        )

    if footer_notes:
        lines.append("---")
        lines.append("### 🧬 系统自迭代与运行状态")
        lines.append("")
        for note in footer_notes:
            lines.append(f"- {note}")
        lines.append("")

    return title, short_preview, "\n".join(lines).strip() + "\n"
