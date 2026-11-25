"""Command-line interface for AI Daily."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Any

from ai_daily.config import ConfigManager, sanitize_secret
from ai_daily.delivery import DeliveryStore
from ai_daily.evolution.engine import EvolutionEngine
from ai_daily.llm.registry import ModelRegistry
from ai_daily.monitor import report_healthcheck
from ai_daily.notifiers import FeishuAppNotifier, FeishuNotifier, resolve_notifiers
from ai_daily.pipeline.runner import build_llm_client, run_daily_digest
from ai_daily.storage import StateStore


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def cmd_run(args: argparse.Namespace, cfg: ConfigManager) -> int:
    summary = run_daily_digest(
        cfg,
        dry_run=args.dry_run,
        no_push=args.no_push,
        no_llm=args.no_llm,
        window_hours=args.hours,
        max_candidates=args.limit,
        check_models=args.check_models,
        auto_evolve=args.auto_evolve,
        save_candidates_path=args.save_candidates,
        notify_channel=args.channel,
        skip_if_today_exists=args.skip_if_today_exists,
        stage=args.stage,
        state_dir=args.state_dir,
        force=args.force,
        degrade_on_llm_error=args.degrade_on_llm_error,
        retry_unknown=not args.no_retry_unknown,
        run_date_override=args.date,
    )
    if args.json_summary is not None:
        args.json_summary.parent.mkdir(parents=True, exist_ok=True)
        args.json_summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if summary.get("skipped"):
        print(f"=== AI 前沿日报 ({summary['run_date']}) 已跳过 ===")
        print(summary["push_error"])
        return 0

    stage_label = f" [stage={summary.get('stage', 'all')}]" if args.stage != "all" else ""
    print(f"=== AI 前沿日报运行完成 ({summary['run_date']}){stage_label} ===")
    if summary.get("reused_draft"):
        print("待发稿状态: 复用已持久化成稿（跳过重新抓取与模型调用）")
    if summary.get("sources_total"):
        print(f"信源状态: {summary['sources_ok']}/{summary['sources_total']} 成功")
    if summary.get("raw_count") or summary.get("selected_count"):
        print(
            f"漏斗统计: 原始抓取 {summary['raw_count']} -> 去重候选 {summary['candidate_count']} "
            f"(深抓正文 {summary.get('enriched_count', 0)} 条) -> 精选 {summary['selected_count']} 条"
        )
    if summary.get("model_used"):
        degrade_tag = " [降级短报]" if summary.get("degraded") else ""
        print(
            f"使用模型: {summary['model_used']}{degrade_tag} "
            f"(tokens: in={summary['prompt_tokens']}, out={summary['completion_tokens']})"
        )
    if summary.get("delivery_status") == "sent" or args.dry_run or args.no_push:
        print(f"本地归档: {summary['output_path']}")
    print(f"交付状态: {summary.get('delivery_status')} | 渠道回执: {summary.get('channel_statuses') or '{}'}")

    if summary["pushed"]:
        ch_str = ", ".join(summary.get("pushed_channels") or ["already-sent"])
        print(f"消息推送: 成功（{ch_str}）| 标题: {summary['title']}")
    elif args.stage == "prepare" and summary.get("ok"):
        print(f"待发成稿: 已持久化至 {summary['state_dir']}/{summary['run_date']}/draft.json")
    elif args.dry_run or args.no_push:
        print("消息推送: 已跳过（--dry-run / --no-push）")
    else:
        print(f"消息推送: 未完成（{summary['push_error']}）", file=sys.stderr)

    if summary["emerging_signals"]:
        print("新发现信号: " + ", ".join(summary["emerging_signals"]))
    if summary["failed_sources"]:
        print("异常信源:")
        for f in summary["failed_sources"]:
            print(f"  - {f['id']}: {f['error']}")

    if summary.get("ok"):
        return 0
    if summary.get("delivery_status") == "unknown":
        return 2
    return 1


def cmd_sources(args: argparse.Namespace, cfg: ConfigManager) -> int:
    sources_cfg = cfg.load_sources()
    sources = sources_cfg.get("sources", [])

    if args.sources_action == "list":
        for s in sources:
            flag = "ON " if s.get("enabled", True) else "OFF"
            print(
                f"[{flag}] {s['id']:<26} ({s['type']:<15} | {s['category']:<10} | w={s.get('weight', 1.0)})  {s['name']}"
            )
        return 0

    if args.sources_action == "stats":
        store = StateStore(cfg.data_dir / "state.db")
        stats = store.get_source_stats(lookback_runs=args.lookback)
        if not stats:
            print("暂无历史抓取统计，请先运行一次 `python3 -m ai_daily run`。")
            return 0
        print(
            f"{'SOURCE_ID':<26} {'RUNS':>4} {'OK%':>6} {'FAIL':>4} {'NEW':>5} {'SEL':>4} {'HIT%':>6} {'SCORE':>5}"
        )
        for st in stats:
            print(
                f"{st['source_id']:<26} {st['runs']:>4} {st['success_rate']*100:>5.0f}% "
                f"{st['consecutive_failures']:>4} {st['total_new']:>5} {st['total_selected']:>4} "
                f"{st['hit_rate']*100:>5.1f}% {st['mean_selected_score']:>5.1f}"
            )
        return 0

    if args.sources_action == "add":
        if any(s["id"] == args.id for s in sources):
            print(f"信源 ID '{args.id}' 已存在。", file=sys.stderr)
            return 1
        entry: dict[str, Any] = {
            "id": args.id,
            "name": args.name,
            "type": args.type,
            "category": args.category,
            "url": args.url,
            "tier": 2,
            "weight": args.weight,
            "max_items": 12,
            "enabled": True,
            "tags": [args.category, "custom"],
        }
        if args.type == "web_watch":
            entry["mode"] = args.mode
            if args.link_pattern:
                entry["link_pattern"] = args.link_pattern
        sources.append(entry)
        backup = cfg.save_with_backup("sources", sources_cfg, tag="add-source")
        print(f"已添加信源 '{args.id}'（备份：{backup}）")
        return 0

    if args.sources_action == "toggle":
        for s in sources:
            if s["id"] == args.id:
                s["enabled"] = not bool(s.get("enabled", True))
                backup = cfg.save_with_backup("sources", sources_cfg, tag="toggle-source")
                print(
                    f"已将信源 '{args.id}' 设为 {'启用' if s['enabled'] else '停用'}（备份：{backup}）"
                )
                return 0
        print(f"未找到信源 ID '{args.id}'。", file=sys.stderr)
        return 1

    return 1


def cmd_models(args: argparse.Namespace, cfg: ConfigManager) -> int:
    llm_cfg = cfg.load_llm()
    if args.models_action == "status":
        print(f"Base URL     : {llm_cfg['base_url']}")
        print(f"Primary Model: {llm_cfg['primary_model']}")
        print(f"Fallbacks    : {', '.join(llm_cfg.get('fallback_models', []))}")
        print(f"API Key Set  : {'Yes' if cfg.resolve_api_key(llm_cfg) else 'No'}")
        reg_path = cfg.data_dir / "model_registry.json"
        if reg_path.is_file():
            reg = json.loads(reg_path.read_text(encoding="utf-8"))
            print(f"Last Sync    : {reg.get('checked_at')}")
            print(f"Recommended  : {reg.get('recommended_upgrade') or '当前已是最新匹配模型'}")
        return 0

    if args.models_action == "sync":
        client = build_llm_client(cfg)
        store = StateStore(cfg.data_dir / "state.db")
        registry = ModelRegistry(cfg, client, store=store)
        report = registry.sync(apply=args.apply, probe=not args.no_probe)
        _print_json(report)
        return 0

    return 1


def cmd_feedback(args: argparse.Namespace, cfg: ConfigManager) -> int:
    store = StateStore(cfg.data_dir / "state.db")
    if args.list:
        rows = store.get_feedback(only_pending=False, limit=30)
        if not rows:
            print("暂无反馈记录。")
            return 0
        for r in rows:
            status = "已应用" if r["applied"] else "待迭代"
            print(f"#{r['id']} [{status}] ({r['kind']}) {r['target']} {r['detail']}")
        return 0

    added = 0
    for kw in args.like or []:
        store.add_feedback("like", kw, args.detail or "")
        added += 1
    for kw in args.dislike or []:
        store.add_feedback("dislike", kw, args.detail or "")
        added += 1
    for kw in args.topic or []:
        store.add_feedback("topic", kw, args.detail or "")
        added += 1
    if args.note:
        store.add_feedback("note", args.note, args.detail or "")
        added += 1

    if added == 0:
        print("请提供 --like / --dislike / --topic / --note 或 --list。")
        return 1
    print(f"已记录 {added} 条偏好反馈；下次运行 `python3 -m ai_daily evolve --apply` 将自动融入配置。")
    return 0


def cmd_evolve(args: argparse.Namespace, cfg: ConfigManager) -> int:
    store = StateStore(cfg.data_dir / "state.db")
    if args.history:
        logs = store.get_recent_evolution_logs(limit=20)
        if not logs:
            print("暂无自主迭代历史。")
            return 0
        for log in logs:
            print(f"[{log['created_at']}] ({log['kind']}) {log['summary']}")
            if log["backup_path"]:
                print(f"  backup: {log['backup_path']}")
        return 0

    client = None if args.no_llm else build_llm_client(cfg)
    engine = EvolutionEngine(cfg, store, client)
    result = engine.run_cycle(apply=args.apply, use_llm=not args.no_llm)
    _print_json(result)
    return 0


def cmd_rollback(args: argparse.Namespace, cfg: ConfigManager) -> int:
    if args.list:
        backups = cfg.list_backups(args.config_name)
        if not backups:
            print(f"配置 '{args.config_name}' 暂无历史备份。")
            return 0
        for b in backups:
            print(b.name)
        return 0
    restored = cfg.rollback(args.config_name, backup_file=args.backup)
    store = StateStore(cfg.data_dir / "state.db")
    store.record_evolution_log(
        kind="rollback",
        summary=f"Rolled back {args.config_name} to {restored.name}",
        backup_path=str(restored),
    )
    print(f"已将 config/{args.config_name}.json 回滚至 {restored.name}")
    return 0


def cmd_setup(args: argparse.Namespace, cfg: ConfigManager) -> int:
    """Interactively prompt for secrets with silent (no-echo) terminal input and save to .env."""
    env_path = cfg.root / ".env"
    existing: dict[str, str] = {}
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k, v = s.split("=", 1)
                existing[k.strip()] = v.strip()

    print("=== AI Daily 静默密钥配置（输入时不回显，仅写入本地 .env）===")
    cur_base = existing.get("LLM_BASE_URL") or "https://api.deepseek.com/v1"
    base_in = input(f"LLM_BASE_URL [回车默认 {cur_base}]: ").strip()
    base_url = base_in or cur_base

    ds_key = sanitize_secret(getpass.getpass("DEEPSEEK_API_KEY（静默输入，回车保持现有）: "))
    if not ds_key:
        ds_key = existing.get("DEEPSEEK_API_KEY", "")
    elif any(c.isspace() for c in ds_key):
        print("⚠️ 警告：检测到的 DEEPSEEK_API_KEY 含有空格，可能是误粘贴了命令，请核对 .env！", file=sys.stderr)
    else:
        print(f"  ✓ 已读取 DEEPSEEK_API_KEY（前缀 {ds_key[:4]}...，长度 {len(ds_key)}）")

    print("\n--- 飞书推送配置（推荐自建应用消息 API，支持 1 小时 uuid 幂等防重）---")
    cur_app_id = existing.get("FEISHU_APP_ID", "")
    app_id_in = sanitize_secret(input(f"FEISHU_APP_ID [回车默认 {cur_app_id or '跳过'}]: "))
    fs_app_id = app_id_in or cur_app_id

    fs_app_secret = existing.get("FEISHU_APP_SECRET", "")
    fs_chat_id = existing.get("FEISHU_CHAT_ID", "")
    if fs_app_id:
        sec_in = sanitize_secret(
            getpass.getpass("FEISHU_APP_SECRET（静默输入，回车保持现有）: ")
        )
        if sec_in:
            fs_app_secret = sec_in
            print(f"  ✓ 已读取 FEISHU_APP_SECRET（长度 {len(fs_app_secret)}）")

        if fs_app_secret:
            try:
                app_client = FeishuAppNotifier(
                    app_id=fs_app_id,
                    app_secret=fs_app_secret,
                    receive_id="oc_placeholder",
                )
                chats = app_client.list_chats()
                if chats:
                    print("  已自动查询到机器人所在的飞书群：")
                    for idx, ch in enumerate(chats, start=1):
                        mark = " (当前)" if ch["chat_id"] == fs_chat_id else ""
                        print(f"    [{idx}] {ch['name']}  ({ch['chat_id']}){mark}")
                    pick = input(
                        f"请选择接收日报的群编号或直接粘贴 chat_id [回车默认 {fs_chat_id or '1'}]: "
                    ).strip()
                    if not pick and not fs_chat_id:
                        fs_chat_id = chats[0]["chat_id"]
                    elif pick.isdigit() and 1 <= int(pick) <= len(chats):
                        fs_chat_id = chats[int(pick) - 1]["chat_id"]
                    elif pick:
                        fs_chat_id = sanitize_secret(pick)
                else:
                    print("  ⚠️ 尚未检测到机器人加入任何群聊，请先在飞书群设置里拉入机器人。")
                    chat_in = sanitize_secret(
                        input(f"FEISHU_CHAT_ID [回车默认 {fs_chat_id or '跳过'}]: ")
                    )
                    if chat_in:
                        fs_chat_id = chat_in
            except Exception as exc:
                print(f"  ⚠️ 自动拉取群列表失败（{exc}），可手动填入 FEISHU_CHAT_ID。", file=sys.stderr)
                chat_in = sanitize_secret(
                    input(f"FEISHU_CHAT_ID [回车默认 {fs_chat_id or '跳过'}]: ")
                )
                if chat_in:
                    fs_chat_id = chat_in

    fs_url = sanitize_secret(
        getpass.getpass("FEISHU_WEBHOOK_URL（可选旧版 Webhook 备用，静默输入，回车保持现有/跳过）: ")
    )
    if not fs_url:
        fs_url = existing.get("FEISHU_WEBHOOK_URL", "")
    elif not fs_url.startswith(("http://", "https://")):
        print("⚠️ 警告：FEISHU_WEBHOOK_URL 应以 https://open.feishu.cn/... 开头，请核对 .env！", file=sys.stderr)
    else:
        print(f"  ✓ 已读取 FEISHU_WEBHOOK_URL（长度 {len(fs_url)}）")

    fs_secret = existing.get("FEISHU_WEBHOOK_SECRET", "")
    if fs_url:
        sec_hook = sanitize_secret(
            getpass.getpass("FEISHU_WEBHOOK_SECRET（可选 Webhook 签名密钥，回车保持现有/跳过）: ")
        )
        if sec_hook:
            fs_secret = sec_hook

    hc_url = sanitize_secret(
        getpass.getpass("HEALTHCHECK_PING_URL（可选外部死信监控如 Healthchecks.io，回车保持现有/跳过）: ")
    )
    if not hc_url:
        hc_url = existing.get("HEALTHCHECK_PING_URL", "")
    else:
        print(f"  ✓ 已读取 HEALTHCHECK_PING_URL（长度 {len(hc_url)}）")

    model_in = input("LLM_MODEL（可选，回车默认跟随 config/llm.json 自动迭代）: ").strip()

    has_feishu = bool((fs_app_id and fs_app_secret and fs_chat_id) or fs_url)
    lines = [
        f"LLM_BASE_URL={base_url}",
        f"DEEPSEEK_API_KEY={ds_key}",
        f"NOTIFY_CHANNEL={'feishu' if has_feishu else existing.get('NOTIFY_CHANNEL', 'auto')}",
    ]
    if fs_app_id:
        lines.append(f"FEISHU_APP_ID={fs_app_id}")
    if fs_app_secret:
        lines.append(f"FEISHU_APP_SECRET={fs_app_secret}")
    if fs_chat_id:
        lines.append(f"FEISHU_CHAT_ID={fs_chat_id}")
    if fs_url:
        lines.append(f"FEISHU_WEBHOOK_URL={fs_url}")
    if fs_secret:
        lines.append(f"FEISHU_WEBHOOK_SECRET={fs_secret}")
    if hc_url:
        lines.append(f"HEALTHCHECK_PING_URL={hc_url}")
    if model_in:
        lines.append(f"LLM_MODEL={model_in}")

    managed = {
        "LLM_BASE_URL",
        "DEEPSEEK_API_KEY",
        "NOTIFY_CHANNEL",
        "FEISHU_APP_ID",
        "FEISHU_APP_SECRET",
        "FEISHU_CHAT_ID",
        "FEISHU_WEBHOOK_URL",
        "FEISHU_WEBHOOK_SECRET",
        "HEALTHCHECK_PING_URL",
        "LLM_MODEL",
    }
    for k, v in existing.items():
        if k not in managed:
            lines.append(f"{k}={v}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(env_path, 0o600)
    except OSError:
        pass
    print(f"已安全写入 {env_path}（权限 600，已加入 .gitignore）。")
    return 0


def cmd_feishu_chats(args: argparse.Namespace, cfg: ConfigManager) -> int:
    """List group chats that the Feishu custom app bot has joined."""
    app_notifier = FeishuAppNotifier()
    chats = app_notifier.list_chats()
    if args.json:
        _print_json(chats)
        return 0
    if not chats:
        print("未检测到机器人所在的飞书群聊。请先在飞书群「群设置 → 群机器人 → 添加机器人」中拉入应用机器人。")
        return 0
    print(f"共检测到 {len(chats)} 个机器人已加入的飞书群：")
    for idx, ch in enumerate(chats, start=1):
        print(f"  {idx}. {ch['name']}  |  FEISHU_CHAT_ID={ch['chat_id']}")
    return 0


def cmd_status(args: argparse.Namespace, cfg: ConfigManager) -> int:
    """Print recent delivery receipts and 14-day reliability metrics."""
    state_dir = args.state_dir or (cfg.data_dir / "delivery")
    dstore = DeliveryStore(state_dir)
    metrics = dstore.summarize_metrics(days=args.days)
    if args.json:
        _print_json(metrics)
        return 0
    print(
        f"=== AI Daily 交付观测统计（最近 {metrics['days_checked']} 期，"
        f"送达 {metrics['delivered']} / 异常 {metrics['failed_or_pending']} / "
        f"降级 {metrics['degraded']} / 触发补发 {metrics['retried']}）==="
    )
    if not metrics["rows"]:
        print(f"状态目录 {state_dir} 下暂无回执记录。")
        return 0
    print(f"{'DATE':<12} {'SENT':<5} {'DEGRADED':<8} {'ATT':>3} {'DELIVERED_AT_LOCAL':<26} {'MODEL':<22} {'CHANNELS'}")
    for row in metrics["rows"]:
        ch_str = ", ".join(f"{k}:{v}" for k, v in (row.get("channels") or {}).items()) or "-"
        if row.get("fetch_error"):
            ch_str += " [fetch_failed]"
        if row.get("llm_error"):
            ch_str += " [llm_failed]"
        print(
            f"{row['run_date']:<12} {'YES' if row['sent'] else 'NO':<5} "
            f"{'YES' if row['degraded'] else 'NO':<8} {row['attempts']:>3} "
            f"{(row['delivered_at_local'] or '-'):<26} {(row['model_used'] or '-'):<22} {ch_str}"
        )
    return 0


def cmd_ping_monitor(args: argparse.Namespace, cfg: ConfigManager) -> int:
    """Report run status to external dead-man's-switch monitor (HEALTHCHECK_PING_URL)."""
    ok, msg = report_healthcheck(
        ping_url=args.url,
        status=args.status,
        message=args.message,
    )
    print(msg)
    return 0 if ok or "未配置" in msg else 1


def cmd_push(args: argparse.Namespace, cfg: ConfigManager) -> int:
    """Push an existing archived Markdown digest directly to Feishu."""
    target_path: Path | None = args.file
    if target_path is None:
        md_files = sorted(cfg.output_dir.glob("*.md"), reverse=True)
        if not md_files:
            raise FileNotFoundError(f"在 {cfg.output_dir} 下未找到任何已生成的日报 Markdown。")
        target_path = md_files[0]
    if not target_path.is_file():
        raise FileNotFoundError(f"文件不存在: {target_path}")

    markdown = target_path.read_text(encoding="utf-8")
    first_line = next((ln.strip() for ln in markdown.splitlines() if ln.strip()), "")
    title = first_line.lstrip("# ").strip() or f"AI 前沿日报 · {target_path.stem}"

    notifiers = resolve_notifiers(channel=args.channel)
    for n in notifiers:
        res = n.send(title=title, markdown=markdown)
        mode = getattr(n, "delivery_mode", n.name)
        print(f"已将 {target_path.name} 推送至 {n.name} ({mode})！")
        _print_json(res)
    return 0


def cmd_test_notify(args: argparse.Namespace, cfg: ConfigManager) -> int:
    notifiers = resolve_notifiers(channel=args.channel)
    for n in notifiers:
        res = n.send(title=args.title, markdown=args.message)
        mode = getattr(n, "delivery_mode", n.name)
        print(f"{n.name} ({mode}) 测试推送成功！")
        _print_json(res)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai_daily",
        description="自迭代 AI 前沿日报（聚焦 Harness 优化、模型与训推优化，飞书原生交互卡片推送）",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="项目根目录路径（默认自动检测）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="抓取信源、筛选生成日报并推送（默认优先飞书自建应用 API）")
    p_run.add_argument("--dry-run", action="store_true", help="试运行：不推送且不写入去重状态")
    p_run.add_argument("--no-push", action="store_true", help="只生成本地 Markdown 归档，不发消息")
    p_run.add_argument("--no-llm", action="store_true", help="离线规则模式（不调 LLM API）")
    p_run.add_argument("--hours", type=int, default=None, help="覆盖抓取时间窗口（小时）")
    p_run.add_argument("--limit", type=int, default=None, help="覆盖送入 LLM 的最大候选条数")
    p_run.add_argument("--check-models", action="store_true", help="运行前先探测是否有更新的可用模型")
    p_run.add_argument("--auto-evolve", action="store_true", help="按周期生成配置自演进提案（隔离于推送之外）")
    p_run.add_argument("--save-candidates", type=Path, default=None, help="将去重预筛后的候选列表导出为 JSON")
    p_run.add_argument(
        "--channel",
        choices=["auto", "feishu", "feishu-app", "feishu-webhook"],
        default=None,
        help="指定推送渠道（默认读取 NOTIFY_CHANNEL 或 auto，优先飞书应用 API）",
    )
    p_run.add_argument(
        "--skip-if-today-exists",
        action="store_true",
        help="若今日已确认送达（或历史归档已存在）则幂等跳过",
    )
    p_run.add_argument(
        "--stage",
        choices=["all", "prepare", "deliver"],
        default="all",
        help="执行阶段：prepare（仅成稿并落盘待发状态）、deliver（仅按回执推送并落归档）、all（默认全流程）",
    )
    p_run.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="待发稿与发送回执持久化目录（默认 data/delivery，CI 中指向 delivery-state 分支目录）",
    )
    p_run.add_argument(
        "--force",
        action="store_true",
        help="强制忽略今日已送达回执，重新抓取生成并推送",
    )
    p_run.add_argument(
        "--degrade-on-llm-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="主备模型全部失败时是否允许生成标明降级的短报（10:23 主跑传 --no-degrade-on-llm-error，10:47 兜底默认开启）",
    )
    p_run.add_argument(
        "--no-retry-unknown",
        action="store_true",
        help="对无幂等键渠道（如旧 Webhook）遇到结果未知状态时禁止自动补发",
    )
    p_run.add_argument("--date", default=None, help="手动覆盖日报日期（YYYY-MM-DD，供测试/补跑）")
    p_run.add_argument("--json-summary", type=Path, default=None, help="将结构化执行摘要写入 JSON 文件")

    # sources
    p_src = sub.add_parser("sources", help="管理与查看 AI 门户及官方信源")
    src_sub = p_src.add_subparsers(dest="sources_action", required=True)
    src_sub.add_parser("list", help="列出所有信源")
    p_stats = src_sub.add_parser("stats", help="查看信源健康度与日报入选命中率")
    p_stats.add_argument("--lookback", type=int, default=20, help="统计最近 N 次运行")
    p_add = src_sub.add_parser("add", help="新增自定义 RSS 或网页监控信源")
    p_add.add_argument("--id", required=True, help="唯一信源 ID")
    p_add.add_argument("--name", required=True, help="信源展示名称")
    p_add.add_argument("--url", required=True, help="RSS 或网页 URL")
    p_add.add_argument("--type", choices=["rss", "web_watch"], default="rss")
    p_add.add_argument(
        "--category", choices=["harness", "models", "systems", "geek"], default="geek"
    )
    p_add.add_argument("--weight", type=float, default=1.1)
    p_add.add_argument("--mode", choices=["link_list", "changelog"], default="link_list")
    p_add.add_argument("--link-pattern", default="", help="web_watch link_list 模式下的 URL 正则")
    p_tog = src_sub.add_parser("toggle", help="启用/停用指定信源")
    p_tog.add_argument("id", help="信源 ID")

    # models
    p_mod = sub.add_parser("models", help="LLM API 模型状态与自动迭代升级")
    mod_sub = p_mod.add_subparsers(dest="models_action", required=True)
    mod_sub.add_parser("status", help="查看当前主备模型与最近探测结果")
    p_sync = mod_sub.add_parser("sync", help="从 /models 探测更新版本模型并探活")
    p_sync.add_argument("--apply", action="store_true", help="自动备份并升级 config/llm.json 主模型")
    p_sync.add_argument("--no-probe", action="store_true", help="仅列举排序，不发探活请求")

    # feedback
    p_fb = sub.add_parser("feedback", help="记录每日阅读反馈以驱动自进化")
    p_fb.add_argument("--like", action="append", help="想多看的方向/关键词（可多次指定）")
    p_fb.add_argument("--dislike", action="append", help="想屏蔽的噪音词（可多次指定）")
    p_fb.add_argument("--topic", action="append", help="新增重点追踪话题/产品名（可多次指定）")
    p_fb.add_argument("--note", default="", help="给 AI 主编的长期偏好备注")
    p_fb.add_argument("--detail", default="", help="补充原因说明")
    p_fb.add_argument("--list", action="store_true", help="查看历史反馈记录")

    # evolve
    p_evo = sub.add_parser("evolve", help="基于信源命中率、新信号与反馈执行自主迭代")
    p_evo.add_argument("--apply", action="store_true", help="应用演进提案（自动备份旧配置）")
    p_evo.add_argument("--no-llm", action="store_true", help="仅运行规则遥测演进，不调 LLM")
    p_evo.add_argument("--history", action="store_true", help="查看历史迭代日志")

    # rollback
    p_rb = sub.add_parser("rollback", help="将配置回滚到自主迭代前的历史版本")
    p_rb.add_argument("config_name", choices=["llm", "profile", "sources", "prompts"])
    p_rb.add_argument("--backup", default=None, help="指定备份文件名（默认回滚到最近一次）")
    p_rb.add_argument("--list", action="store_true", help="列出可用历史备份")

    # setup
    sub.add_parser("setup", help="终端静默录入 DeepSeek API Key、飞书自建应用凭证与群 chat_id 到 .env")

    # feishu-chats
    p_chats = sub.add_parser("feishu-chats", help="查询飞书自建应用机器人已加入的群聊及 chat_id")
    p_chats.add_argument("--json", action="store_true", help="以 JSON 格式输出")

    # status
    p_st = sub.add_parser("status", help="查看最近 N 天的日报送达回执与无人值守观测指标")
    p_st.add_argument("--days", type=int, default=14, help="统计最近 N 天（默认 14）")
    p_st.add_argument("--state-dir", type=Path, default=None, help="交付状态目录（默认 data/delivery）")
    p_st.add_argument("--json", action="store_true", help="以 JSON 格式输出")

    # ping-monitor
    p_pm = sub.add_parser("ping-monitor", help="向外部死信监控（HEALTHCHECK_PING_URL）上报运行状态")
    p_pm.add_argument("--status", choices=["ok", "fail", "start"], default="ok")
    p_pm.add_argument("--message", default="")
    p_pm.add_argument("--url", default=None, help="覆盖 HEALTHCHECK_PING_URL")

    # push
    p_push = sub.add_parser("push", help="将已生成的本地 Markdown 日报直接推送至飞书")
    p_push.add_argument("--file", type=Path, default=None, help="指定 Markdown 文件路径（默认推送最新一期）")
    p_push.add_argument(
        "--channel",
        choices=["auto", "feishu", "feishu-app", "feishu-webhook"],
        default=None,
        help="推送渠道（默认优先飞书应用 API）",
    )

    # test-notify
    p_tn = sub.add_parser("test-notify", help="发送一条飞书连通性测试卡片")
    p_tn.add_argument("--title", default="AI前沿日报｜飞书应用 API 卡片推送测试成功")
    p_tn.add_argument(
        "--message",
        default="**连通性测试通过**\n\n- 渠道：飞书交互卡片（零跳转直读，支持 1 小时 uuid 幂等防重）\n- 状态：已就绪，点击原文链接可直接跳转。",
    )
    p_tn.add_argument(
        "--channel",
        choices=["auto", "feishu", "feishu-app", "feishu-webhook"],
        default=None,
        help="测试渠道（默认优先飞书应用 API）",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = ConfigManager(root=args.root)

    dispatch = {
        "run": cmd_run,
        "sources": cmd_sources,
        "models": cmd_models,
        "feedback": cmd_feedback,
        "evolve": cmd_evolve,
        "rollback": cmd_rollback,
        "setup": cmd_setup,
        "feishu-chats": cmd_feishu_chats,
        "status": cmd_status,
        "ping-monitor": cmd_ping_monitor,
        "push": cmd_push,
        "test-notify": cmd_test_notify,
    }
    try:
        return dispatch[args.command](args, cfg)
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
